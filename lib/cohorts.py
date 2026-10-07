"""Deterministic cohort assignments for cohort-aware validation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class CohortAssignment:
    """A cohort assignment and the signal used to create it.

    ``test_ids`` is absent only for the random validation-only control.
    """

    strategy: str
    val_ids: np.ndarray
    test_ids: np.ndarray | None
    val_signal: np.ndarray
    test_signal: np.ndarray | None
    metadata: dict[str, Any]


@dataclass
class FrozenSelection:
    epoch: int
    val_mse: float
    val_predictions: np.ndarray
    test_predictions: np.ndarray


class CohortEarlyStoppingController:
    """Independent first-exhaustion early stopping for global and cohorts."""

    def __init__(self, *, patience: int, n_cohorts: int) -> None:
        if patience < 1:
            raise ValueError('patience must be positive')
        self.patience = patience
        self.n_cohorts = n_cohorts
        self.global_selection: FrozenSelection | None = None
        self.cohort_selections: list[FrozenSelection | None] = [None] * n_cohorts
        self._global_best: FrozenSelection | None = None
        self._cohort_best: list[FrozenSelection | None] = [None] * n_cohorts
        self._global_bad_epochs = 0
        self._cohort_bad_epochs = [0] * n_cohorts
        self._global_active = True
        self._cohort_active = [True] * n_cohorts
        self.global_stopped_epoch: int | None = None
        self.cohort_stopped_epochs: list[int | None] = [None] * n_cohorts
        self.history: list[dict[str, Any]] = []
        # Kept in checkpoint.pt, rather than as separate ambiguous epoch files.  Each
        # item contains complete, original-scale validation and test vectors.
        self.epoch_predictions: list[dict[str, np.ndarray | int]] = []

    @staticmethod
    def _selection(
        epoch: int,
        mse: float,
        val_predictions: np.ndarray,
        test_predictions: np.ndarray,
    ) -> FrozenSelection:
        return FrozenSelection(
            epoch=epoch,
            val_mse=float(mse),
            val_predictions=np.array(val_predictions, copy=True),
            test_predictions=np.array(test_predictions, copy=True),
        )

    def _update_one(
        self,
        *,
        active: bool,
        best: FrozenSelection | None,
        bad_epochs: int,
        epoch: int,
        mse: float,
        val_predictions: np.ndarray,
        test_predictions: np.ndarray,
    ) -> tuple[bool, FrozenSelection | None, int, FrozenSelection | None]:
        if not active:
            return active, best, bad_epochs, None
        if best is None or mse < best.val_mse:
            return (
                True,
                self._selection(epoch, mse, val_predictions, test_predictions),
                0,
                None,
            )
        bad_epochs += 1
        if bad_epochs >= self.patience:
            assert best is not None
            return False, best, bad_epochs, best
        return True, best, bad_epochs, None

    def update(
        self,
        epoch: int,
        global_val_mse: float,
        cohort_val_mse: list[float] | np.ndarray,
        val_predictions: np.ndarray,
        test_predictions: np.ndarray,
    ) -> None:
        values = np.asarray(cohort_val_mse, dtype=float)
        if values.shape != (self.n_cohorts,) or not np.isfinite(values).all():
            raise ValueError('cohort_val_mse must be finite with one value per cohort')
        if not np.isfinite(global_val_mse):
            raise ValueError('global_val_mse must be finite')
        (
            self._global_active,
            self._global_best,
            self._global_bad_epochs,
            frozen,
        ) = self._update_one(
            active=self._global_active,
            best=self._global_best,
            bad_epochs=self._global_bad_epochs,
            epoch=epoch,
            mse=global_val_mse,
            val_predictions=val_predictions,
            test_predictions=test_predictions,
        )
        if frozen is not None:
            self.global_selection = frozen
            self.global_stopped_epoch = epoch
        for index, mse in enumerate(values):
            active, best, bad_epochs, frozen = self._update_one(
                active=self._cohort_active[index],
                best=self._cohort_best[index],
                bad_epochs=self._cohort_bad_epochs[index],
                epoch=epoch,
                mse=float(mse),
                val_predictions=val_predictions,
                test_predictions=test_predictions,
            )
            self._cohort_active[index] = active
            self._cohort_best[index] = best
            self._cohort_bad_epochs[index] = bad_epochs
            if frozen is not None:
                self.cohort_selections[index] = frozen
                self.cohort_stopped_epochs[index] = epoch
        self.history.append(
            {
                'epoch': epoch,
                'global_val_mse': float(global_val_mse),
                'cohort_val_mse': values.tolist(),
            }
        )
        self.epoch_predictions.append(
            {
                'epoch': epoch,
                'val': np.array(val_predictions, copy=True),
                'test': np.array(test_predictions, copy=True),
            }
        )

    def finalize_global(self) -> FrozenSelection:
        """Return the frozen global choice, or freeze its last best epoch at exit."""

        if self.global_selection is None:
            if self._global_best is None:
                raise RuntimeError('global selection has no evaluated epoch')
            self.global_selection = self._global_best
        return self.global_selection

    @property
    def all_cohorts_stopped(self) -> bool:
        return not any(self._cohort_active)


def stitch_predictions(
    predictions_by_cohort: list[np.ndarray], cohort_ids: np.ndarray
) -> np.ndarray:
    """Select each object from its cohort's complete prediction vector."""

    ids = np.asarray(cohort_ids, dtype=int)
    if not predictions_by_cohort or len(predictions_by_cohort) <= ids.max():
        raise ValueError('cohort IDs do not match prediction vectors')
    stacked = np.stack(predictions_by_cohort)
    if stacked.shape[1] != len(ids):
        raise ValueError('prediction vectors do not match cohort_ids')
    return stacked[ids, np.arange(len(ids))]


def _as_signal(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f'{name} must be a non-empty one-dimensional array')
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must contain only finite values')
    return result


def _validate_val_ids(
    cohort_ids: np.ndarray, *, n_cohorts: int, thresholds: np.ndarray
) -> None:
    counts = np.bincount(cohort_ids, minlength=n_cohorts)
    if len(counts) != n_cohorts or (counts == 0).any():
        raise ValueError(
            'empty val cohort after threshold assignment; '
            f'{thresholds.tolist()=}, {counts.tolist()=}'
        )


def build_quantile_cohorts(
    val_signal: np.ndarray | list[float],
    test_signal: np.ndarray | list[float],
    *,
    n_cohorts: int,
    strategy: str = 'uncertainty',
    balance_ties_on_val: bool = False,
) -> CohortAssignment:
    """Build val-quantile cohorts and map test with val-derived thresholds."""

    if n_cohorts < 2:
        raise ValueError('n_cohorts must be at least two')
    val_signal = _as_signal(val_signal, name='val_signal')
    test_signal = _as_signal(test_signal, name='test_signal')
    if len(val_signal) < n_cohorts:
        raise ValueError('val_signal has fewer entries than n_cohorts')

    chunks = np.array_split(np.argsort(val_signal, kind='stable'), n_cohorts)
    thresholds = np.asarray([val_signal[chunk[-1]] for chunk in chunks[:-1]])
    # Equivalent to the notebook's `(signal[:, None] > thresholds).sum(1)`.
    val_ids = np.searchsorted(thresholds, val_signal, side='left')
    test_ids = np.searchsorted(thresholds, test_signal, side='left')
    assignment_source = 'val_thresholds'
    if balance_ties_on_val and np.any(np.bincount(val_ids, minlength=n_cohorts) == 0):
        # Repeated threshold values can otherwise empty a validation cohort.
        # Preserve uncertainty order and the intended array_split sizes.
        val_ids = np.empty(len(val_signal), dtype=int)
        for cohort, chunk in enumerate(chunks):
            val_ids[chunk] = cohort
        test_ids = np.searchsorted(thresholds, test_signal, side='right')
        assignment_source = 'stable_rank_for_tied_val'
    _validate_val_ids(val_ids, n_cohorts=n_cohorts, thresholds=thresholds)

    return CohortAssignment(
        strategy=strategy,
        val_ids=val_ids,
        test_ids=test_ids,
        val_signal=val_signal,
        test_signal=test_signal,
        metadata={
            'n_cohorts': n_cohorts,
            'thresholds': thresholds.tolist(),
            'assignment_source': assignment_source,
            'val_counts': np.bincount(val_ids, minlength=n_cohorts).tolist(),
            'test_counts': np.bincount(test_ids, minlength=n_cohorts).tolist(),
        },
    )


def build_random_cohorts(
    val_signal: np.ndarray | list[float],
    *,
    n_cohorts: int,
    rng: np.random.Generator,
) -> CohortAssignment:
    """Randomize only val cohort IDs while preserving quantile-group sizes."""

    val_signal = _as_signal(val_signal, name='val_signal')
    placeholder_test = np.arange(n_cohorts, dtype=float)
    base = build_quantile_cohorts(
        val_signal, placeholder_test, n_cohorts=n_cohorts, strategy='random'
    )
    val_ids = rng.permutation(base.val_ids)
    return CohortAssignment(
        strategy='random',
        val_ids=val_ids,
        test_ids=None,
        val_signal=val_signal,
        test_signal=None,
        metadata={
            'n_cohorts': n_cohorts,
            'val_counts': np.bincount(val_ids, minlength=n_cohorts).tolist(),
            'randomized_from_thresholds': base.metadata['thresholds'],
        },
    )


def build_balanced_tied_cohorts(
    val_signal: np.ndarray | list[float],
    test_signal: np.ndarray | list[float],
    *,
    n_cohorts: int,
) -> CohortAssignment:
    """Explicit balanced fallback for a genuinely tied uncertainty signal."""

    val_signal = _as_signal(val_signal, name='val_signal')
    test_signal = _as_signal(test_signal, name='test_signal')
    if len(val_signal) < n_cohorts or len(test_signal) < n_cohorts:
        raise ValueError('split has fewer entries than n_cohorts')
    val_ids = np.arange(len(val_signal)) % n_cohorts
    test_ids = np.arange(len(test_signal)) % n_cohorts
    return CohortAssignment(
        strategy='uncertainty_tied_balanced',
        val_ids=val_ids,
        test_ids=test_ids,
        val_signal=val_signal,
        test_signal=test_signal,
        metadata={
            'n_cohorts': n_cohorts,
            'thresholds': None,
            'val_counts': np.bincount(val_ids, minlength=n_cohorts).tolist(),
            'test_counts': np.bincount(test_ids, minlength=n_cohorts).tolist(),
            'assignment_source': 'balanced_index_for_tied_homoscedastic_signal',
        },
    )


def build_random_stitched_cohorts(
    val_signal: np.ndarray | list[float],
    test_signal: np.ndarray | list[float],
    *,
    n_cohorts: int,
    rng_seed: int,
    balance_ties_on_val: bool = False,
) -> CohortAssignment:
    """Randomize quantile-group IDs independently on val and test.

    This follows Kolya's random cohort control: val-derived uncertainty
    thresholds determine cohort sizes, then the IDs are independently
    permuted within each split with the same fixed seed.  The test IDs make a
    stitched oracle well-defined while removing uncertainty alignment.
    """

    val_signal = _as_signal(val_signal, name='val_signal')
    test_signal = _as_signal(test_signal, name='test_signal')
    # With homoscedastic ground truth every quantile threshold is tied.  For
    # the random control, use balanced labels instead of manufacturing empty
    # cohorts from arbitrary tie-breaking.
    if np.unique(val_signal).size < n_cohorts:
        if len(val_signal) < n_cohorts or len(test_signal) < n_cohorts:
            raise ValueError('split has fewer entries than n_cohorts')
        base_val_ids = np.arange(len(val_signal)) % n_cohorts
        base_test_ids = np.arange(len(test_signal)) % n_cohorts
        thresholds: list[float] | None = None
        assignment_source = 'balanced_random_for_tied_signal'
    else:
        base = build_quantile_cohorts(
            val_signal,
            test_signal,
            n_cohorts=n_cohorts,
            strategy='random_stitched',
            balance_ties_on_val=balance_ties_on_val,
        )
        base_val_ids, base_test_ids = base.val_ids, base.test_ids
        thresholds = base.metadata['thresholds']
        assignment_source = (
            'randomized_stable_rank_for_tied_val'
            if base.metadata['assignment_source'] == 'stable_rank_for_tied_val'
            else 'randomized_val_threshold_groups'
        )
    val_ids = np.random.default_rng(rng_seed).permutation(base_val_ids)
    test_ids = np.random.default_rng(rng_seed).permutation(base_test_ids)
    return CohortAssignment(
        strategy='random_stitched',
        val_ids=val_ids,
        test_ids=test_ids,
        val_signal=val_signal,
        test_signal=test_signal,
        metadata={
            'n_cohorts': n_cohorts,
            'val_counts': np.bincount(val_ids, minlength=n_cohorts).tolist(),
            'test_counts': np.bincount(test_ids, minlength=n_cohorts).tolist(),
            'randomized_from_thresholds': thresholds,
            'random_seed': rng_seed,
            'assignment_source': assignment_source,
        },
    )
@dataclass
class _FeatureTransformer:
    numeric: StandardScaler | None
    categorical: OneHotEncoder | None

    @classmethod
    def fit(
        cls, x_num: np.ndarray | None, x_cat: np.ndarray | None
    ) -> _FeatureTransformer:
        if x_num is None and x_cat is None:
            raise ValueError('at least one numeric or categorical feature is required')
        numeric = (
            None
            if x_num is None or x_num.shape[1] == 0
            else StandardScaler().fit(x_num)
        )
        categorical = (
            None
            if x_cat is None or x_cat.shape[1] == 0
            else OneHotEncoder(handle_unknown='ignore').fit(x_cat)
        )
        return cls(numeric=numeric, categorical=categorical)

    def transform(
        self, x_num: np.ndarray | None, x_cat: np.ndarray | None
    ) -> sparse.csr_matrix:
        parts: list[sparse.csr_matrix] = []
        if self.numeric is not None:
            if x_num is None:
                raise ValueError('numeric features are missing')
            parts.append(sparse.csr_matrix(self.numeric.transform(x_num)))
        if self.categorical is not None:
            if x_cat is None:
                raise ValueError('categorical features are missing')
            parts.append(self.categorical.transform(x_cat).tocsr())
        return sparse.hstack(parts, format='csr')

    def metadata(self) -> dict[str, Any]:
        return {
            'numeric_mean': None
            if self.numeric is None
            else self.numeric.mean_.tolist(),
            'numeric_scale': None
            if self.numeric is None
            else self.numeric.scale_.tolist(),
            'categorical_categories': None
            if self.categorical is None
            else [values.tolist() for values in self.categorical.categories_],
        }


def _fit_transform_features(
    x_train_num: np.ndarray | None,
    x_train_cat: np.ndarray | None,
    x_val_num: np.ndarray | None,
    x_val_cat: np.ndarray | None,
    x_test_num: np.ndarray | None,
    x_test_cat: np.ndarray | None,
) -> tuple[
    _FeatureTransformer, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix
]:
    transformer = _FeatureTransformer.fit(x_train_num, x_train_cat)
    return (
        transformer,
        transformer.transform(x_train_num, x_train_cat),
        transformer.transform(x_val_num, x_val_cat),
        transformer.transform(x_test_num, x_test_cat),
    )


def build_kmeans_cohorts(
    x_train_num: np.ndarray | None,
    x_train_cat: np.ndarray | None,
    x_val_num: np.ndarray | None,
    x_val_cat: np.ndarray | None,
    x_test_num: np.ndarray | None,
    x_test_cat: np.ndarray | None,
    *,
    n_cohorts: int,
    rng_seed: int,
) -> CohortAssignment:
    """Fit preprocessing on train and K-means centers exclusively on val."""

    transformer, _, x_val, x_test = _fit_transform_features(
        x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat
    )
    if x_val.shape[0] < n_cohorts:
        raise ValueError('val split has fewer entries than n_cohorts')
    # One deterministic initialization is deliberate here: cohorts are a fixed
    # research split, not an objective being optimized repeatedly.
    model = KMeans(n_clusters=n_cohorts, random_state=rng_seed, n_init=1).fit(x_val)
    val_ids = model.labels_.astype(int, copy=False)
    test_ids = model.predict(x_test).astype(int, copy=False)
    _validate_val_ids(val_ids, n_cohorts=n_cohorts, thresholds=np.empty(0))
    return CohortAssignment(
        strategy='kmeans',
        val_ids=val_ids,
        test_ids=test_ids,
        val_signal=np.zeros(len(val_ids)),
        test_signal=np.zeros(len(test_ids)),
        metadata={
            'n_cohorts': n_cohorts,
            'random_seed': rng_seed,
            'n_init': 1,
            'centers': model.cluster_centers_.tolist(),
            'inertia': float(model.inertia_),
            'preprocessing': transformer.metadata(),
            'val_counts': np.bincount(val_ids, minlength=n_cohorts).tolist(),
            'test_counts': np.bincount(test_ids, minlength=n_cohorts).tolist(),
        },
    )


def build_density_cohorts(
    x_train_num: np.ndarray | None,
    x_train_cat: np.ndarray | None,
    x_val_num: np.ndarray | None,
    x_val_cat: np.ndarray | None,
    x_test_num: np.ndarray | None,
    x_test_cat: np.ndarray | None,
    *,
    n_cohorts: int,
    density_k: int,
) -> CohortAssignment:
    """Group validation/test objects by mean distance to train neighbors."""

    transformer, x_train, x_val, x_test = _fit_transform_features(
        x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat
    )
    if density_k < 1 or density_k > x_train.shape[0]:
        raise ValueError(
            f'density_k must be in [1, {x_train.shape[0]}], got {density_k}'
        )
    neighbors = NearestNeighbors(n_neighbors=density_k, n_jobs=-1).fit(x_train)
    val_signal = neighbors.kneighbors(x_val, return_distance=True)[0].mean(axis=1)
    test_signal = neighbors.kneighbors(x_test, return_distance=True)[0].mean(axis=1)
    result = build_quantile_cohorts(
        val_signal, test_signal, n_cohorts=n_cohorts, strategy='density'
    )
    result.metadata.update(
        {'density_k': density_k, 'preprocessing': transformer.metadata()}
    )
    return result
