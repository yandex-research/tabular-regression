# Source: https://github.com/yandex-research/tabpack
# Paper: https://arxiv.org/abs/2607.05380
# Local benchmark adaptation; see VENDOR.md for modifications and provenance.

import abc
import contextlib
import itertools
import typing
from collections import OrderedDict
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.optim.optimizer
from torch import Tensor

from lib.types import KWArgs

# In many places, the implementation assumes the following values
# and is NOT ready for other values (e.g. in all places where any
# sort of indexing happens). Thus, these constants exist
# only for additional clarity where possible.
PACK_DIM = 0
BATCH_DIM = 1

# >>> Generic modules


class OneHotEncoding(nn.Module):
    # Input:  (*, n_cat_features=len(cardinalities))
    # Output: (*, sum(cardinalities))

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self._cardinalities = cardinalities

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim >= 1
        assert x.shape[-1] == len(self._cardinalities)

        return torch.cat(
            [
                # NOTE
                # This is a quick hack to support out-of-vocabulary categories.
                #
                # Recall that lib.data.transform_cat encodes categorical features
                # as follows:
                # - In-vocabulary values receive indices from `range(cardinality)`.
                # - All out-of-vocabulary values (i.e. new categories in validation
                #   and test data that are not presented in the training data)
                #   receive the index `cardinality`.
                #
                # As such, the line below will produce the standard one-hot encoding for
                # known categories, and the all-zeros encoding for unknown categories.
                # This may not be the best approach to deal with unknown values,
                # but should be enough for our purposes.
                nn.functional.one_hot(x[..., i], cardinality + 1)[..., :-1]
                for i, cardinality in enumerate(self._cardinalities)
            ],
            -1,
        )


# >>> TabPack modules


def get_pack_size(x: Tensor) -> int:
    """Get the pack dimension of a tensor pack."""
    return x.shape[PACK_DIM]


class PackView(nn.Module):
    """Turn a tensor to a valid input for a module pack."""

    def __init__(self, *, pack_size: int) -> None:
        super().__init__()
        self._pack_size = pack_size

    @property
    def pack_size(self) -> int:
        return self._pack_size

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        pack_size = self.pack_size if pack_idx is None else len(pack_idx)
        if x.ndim == 2:
            x = x.unsqueeze(PACK_DIM).expand(pack_size, -1, -1)
        else:
            assert x.ndim == 3
            assert self.training
            assert get_pack_size(x) == pack_size
        return x


@typing.overload
def _get_tensor_pack(x: None, pack_idx: None | Tensor) -> None: ...


@typing.overload
def _get_tensor_pack(x: Tensor, pack_idx: None | Tensor) -> Tensor: ...


def _get_tensor_pack(x, pack_idx):
    return None if x is None else x if pack_idx is None else x[pack_idx]


class ParameterPack(nn.Parameter):
    """Parameter pack.

    Module packs _must_ use this class to store their trainable parameters.
    """

    pass


class _BufferPackMeta(nn.parameter._BufferMeta):  # type: ignore
    """
    `BufferPack` relies on this metaclass for `isintance(..., BufferPack)` to work
    as one would expect. Otherwise, because of `torch.nn.Buffer` implementation details,
    the following (unexpected) behavior would take place:

    ```
    >>> class BufferPack(nn.Buffer):
    ...     pass
    ...
    >>> buffer_pack = BufferPack(torch.zeros(1))
    >>> isinstance(buffer_pack, BufferPack)
    False
    ```
    """

    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or (
            self is BufferPack
            and isinstance(instance, torch.Tensor)
            and getattr(instance, '_is_buffer_pack', False)
        )


class BufferPack(nn.Buffer, metaclass=_BufferPackMeta):  # ty:ignore[conflicting-metaclass]
    """Buffer pack.

    Module packs _must_ use this class to store their non-trainable tensors.
    """

    def __new__(cls, *args, **kwargs):
        t = nn.Buffer.__new__(cls, *args, **kwargs)
        t._is_buffer_pack = True
        return t


class _ModulePackBuffers(OrderedDict):
    """
    This class ensures that reassigning a module pack's field that is a buffer pack
    will result in a field that is also a buffer pack.
    """

    def __setitem__(self, key, value) -> None:
        current_value = self.get(key)
        if (
            current_value is not None
            # The following checks are the reason why `_BufferPackMeta` is needed.
            and isinstance(current_value, BufferPack)
            and not isinstance(value, BufferPack)
        ):
            value = BufferPack(value)
        return super().__setitem__(key, value)


class ModulePack(nn.Module, abc.ABC):
    """The base class for module packs.

    A module pack is a set of modules of the same family that may differ in their
    hyperparameters. Examples:

    * Linear layers with different input and output shapes.
    * Dropout layers with different dropout ratios.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffers = _ModulePackBuffers()

    @property
    @abc.abstractmethod
    def pack_size(self) -> int: ...

    def get_pack_size(self) -> None | int:
        """Get the pack size."""
        try:
            return self.pack_size
        except RuntimeError:
            return None


def _prepare_dimension_pack(
    d: int | list[int], d_max: None | int, pack_size: int, **factory_kwargs
) -> tuple[None | BufferPack, int]:
    if isinstance(d, list):
        # The dimensions are specified for all pack members separately.
        assert len(d) == pack_size
        assert d_max is not None
        assert all(0 < x <= d_max for x in d)
        d_buffer = BufferPack(torch.tensor(d, **factory_kwargs))
    else:
        # The same dimension for all pack members.
        assert d > 0
        if d_max is None:
            d_max = d
        else:
            assert d_max == d
        d_buffer = None

    return d_buffer, d_max


class LinearPack(ModulePack):
    """A pack of linear layers.

    Important implementation details:

    * The module assumes that the input is already properly masked according to
      `in_features` passed to the constructor. Therefore, `in_features` is only used in
      `reset_parameters`, but not in `forward`.
    * The module properly masks its output in `forward` according to `out_features`
      passed to the constructor.
    """

    def __init__(
        self,
        in_features: int | list[int],
        out_features: int | list[int],
        bias: bool = True,
        *,
        max_in_features: None | int = None,
        max_out_features: None | int = None,
        pack_size: int,
        loop: bool = False,
        dtype: None | torch.dtype = None,
        device: None | str | torch.device = None,
    ) -> None:
        assert pack_size > 0

        factory_kwargs: KWArgs = {'dtype': dtype, 'device': device}
        in_features_buffer, max_in_features = _prepare_dimension_pack(
            in_features, max_in_features, pack_size, **factory_kwargs
        )
        out_features_buffer, max_out_features = _prepare_dimension_pack(
            out_features, max_out_features, pack_size, **factory_kwargs
        )

        super().__init__()
        self.in_features = in_features_buffer
        self.out_features = out_features_buffer
        self.weight = ParameterPack(
            torch.empty(pack_size, max_out_features, max_in_features, **factory_kwargs)
        )
        self.bias = (
            ParameterPack(torch.empty(pack_size, max_out_features, **factory_kwargs))
            if bias
            else None
        )
        self._loop = loop

        self.reset_parameters()

    @property
    def max_in_features(self) -> int:
        return self.weight.shape[-1]

    @property
    def max_out_features(self) -> int:
        return self.weight.shape[-2]

    @property
    def pack_size(self) -> int:
        return get_pack_size(self.weight)

    def reset_parameters(self) -> None:
        if self.in_features is None:
            d_in_rsqrt = self.max_in_features**-0.5
            for p in (self.weight, self.bias):
                if p is not None:
                    nn.init.uniform_(p, -d_in_rsqrt, d_in_rsqrt)

        else:
            d_in_rsqrt = self.in_features.float().rsqrt()
            for p in (self.weight, self.bias):
                if p is not None:
                    single_shape = p.shape[1:]
                    pack_view_idx = (slice(None), *((None,) * (p.ndim - 1)))
                    p_init = torch.lerp(
                        -d_in_rsqrt[*pack_view_idx].expand(-1, *single_shape),
                        d_in_rsqrt[*pack_view_idx].expand(-1, *single_shape),
                        torch.rand(
                            self.pack_size,
                            *single_shape,
                            dtype=p.dtype,
                            device=p.device,
                        ),
                    )
                    p.data.copy_(p_init)

    def _do_forward(
        self,
        x: Tensor,
        *,
        weight: Tensor,
        bias: None | Tensor,
        out_features: None | Tensor,
    ) -> Tensor:
        x = (
            torch.bmm(x, weight.transpose(-2, -1))
            if bias is None
            else torch.baddbmm(bias.unsqueeze(BATCH_DIM), x, weight.transpose(-2, -1))
        )
        if out_features is not None:
            output_mask = (
                torch.arange(self.max_out_features, device=x.device)[None]
                < out_features[:, None]
            )
            x = x * output_mask.float().unsqueeze(BATCH_DIM)
        return x

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        assert x.ndim == 3  # (pack_size, batch_size, in_features)
        assert x.shape[-1] == self.max_in_features

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        out_features = _get_tensor_pack(self.out_features, pack_idx)

        if self._loop:
            # This implementation is very slow and is needed only for debugging;
            # in particular, for achieving reproducibility between running `k`
            # models with `LinearPack(..., pack_size=1)` and running one model
            # with `LinearPack(..., pack_size=k)`.
            return torch.cat(
                [
                    self._do_forward(
                        x[i : i + 1],
                        weight=weight[i : i + 1],
                        bias=(None if bias is None else bias[i : i + 1]),
                        out_features=(
                            None if out_features is None else out_features[i : i + 1]
                        ),
                    )
                    for i in range(get_pack_size(x))
                ]
            )
        else:
            return self._do_forward(
                x, weight=weight, bias=bias, out_features=out_features
            )


class DropoutPack(ModulePack):
    """A pack of dropout layers."""

    def __init__(self, p: float | list[float], *, pack_size: int) -> None:
        if isinstance(p, int | float):
            assert 0.0 <= p <= 1.0
        else:
            assert len(p) == pack_size
            assert all(0.0 <= x <= 1.0 for x in p)

        super().__init__()
        self.p = p if isinstance(p, float) else BufferPack(torch.tensor(p))

    @property
    def pack_size(self) -> int:
        if isinstance(self.p, int | float):
            raise RuntimeError('Cannot infer the pack size')
        return get_pack_size(self.p)

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        if isinstance(self.p, int | float):
            return nn.functional.dropout(x, p=self.p, training=self.training)

        p = _get_tensor_pack(self.p, pack_idx)

        if self.training:
            p_keep = 1.0 - p
            p_keep = p_keep[:, *((None,) * (x.ndim - 1))]
            return p_keep.expand_as(x).bernoulli().div_(p_keep) * x
        else:
            return x


class LeakyReLUPack(ModulePack):
    """A pack of leaky ReLU activations.

    This class is not used anywhere and exists only as one more illustration of how the
    "Module Pack" pattern can be used.
    """

    def __init__(self, *, negative_slope: float | list[float], pack_size: int) -> None:
        if isinstance(negative_slope, list):
            assert len(negative_slope) == pack_size
        if isinstance(negative_slope, int | float):
            assert negative_slope > 0
        else:
            assert all(x > 0 for x in negative_slope)

        super().__init__()
        self.negative_slope = (
            BufferPack(torch.tensor(negative_slope))
            if isinstance(negative_slope, list)
            else float(negative_slope)
        )

    @property
    def pack_size(self) -> int:
        if isinstance(self.negative_slope, int | float):
            raise RuntimeError('Cannot infer the pack size')
        return get_pack_size(self.negative_slope)

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        negative_slope = (
            self.negative_slope
            if isinstance(self.negative_slope, int | float)
            else _get_tensor_pack(self.negative_slope, pack_idx)
        )
        unsqueeze_idx = (slice(None), *((None,) * (x.ndim - 1)))

        x_negative = x * (
            negative_slope
            if isinstance(negative_slope, int | float)
            else negative_slope[unsqueeze_idx]
        )
        return torch.where(x >= 0, x, x_negative)


def _make_activation(type: str, **kwargs) -> nn.Module:
    cls = getattr(nn, type)
    kwargs.pop('pack_size', None)
    return cls(**kwargs)


class MLPBackbonePack(ModulePack):
    """A pack of multilayer perceptrons."""

    class Block(ModulePack):
        def __init__(
            self,
            *,
            dropout: float | list[float],
            activation: str | KWArgs,
            pack_size: int,
            **linear_kwargs,
        ) -> None:
            super().__init__()
            self.linear = LinearPack(pack_size=pack_size, **linear_kwargs)
            self.activation = _make_activation(
                pack_size=pack_size,
                **({'type': activation} if isinstance(activation, str) else activation),
            )
            self.dropout = DropoutPack(dropout, pack_size=pack_size)

        @property
        def pack_size(self) -> int:
            return self.linear.pack_size

        def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
            x = self.linear(x, pack_idx)
            x = (
                self.activation(x, pack_idx)
                if isinstance(self.activation, ModulePack)
                else self.activation(x)
            )
            x = self.dropout(x, pack_idx)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int | list[int],
        d_block: int | list[int],
        dropout: float | list[float],
        activation: str | KWArgs = 'ReLU',
        max_n_blocks: None | int = None,
        max_d_block: None | int = None,
        pack_size: int,
        loop: bool = False,
    ) -> None:
        if isinstance(n_blocks, int):
            assert max_n_blocks is None
            max_n_blocks = n_blocks
            n_blocks_buf = None
        else:
            assert max_n_blocks is not None
            assert all(1 <= x <= max_n_blocks for x in n_blocks)
            n_blocks_buf = BufferPack(torch.tensor(n_blocks))

        super().__init__()
        cls = type(self)
        self.blocks = nn.ModuleList(
            [
                cls.Block(
                    in_features=d_in if i == 0 else d_block,
                    out_features=d_block,
                    max_in_features=None if i == 0 else max_d_block,
                    max_out_features=max_d_block,
                    activation=activation,
                    dropout=dropout,
                    pack_size=pack_size,
                    loop=loop,
                )
                for i in range(max_n_blocks)
            ]
        )

        self._max_n_blocks = max_n_blocks
        self.n_blocks = n_blocks_buf
        self._block_idx_cache: dict[BufferPack, list[None | Tensor]] = {}

    @property
    def pack_size(self) -> int:
        return self.blocks[0].pack_size

    @property
    def max_n_blocks(self) -> int:
        return self._max_n_blocks

    def _iter_blocks(self) -> Iterator['MLPBackbonePack.Block']:
        return iter(self.blocks)

    def _compute_block_idx_list(self) -> list[None | Tensor]:
        assert self.n_blocks is not None

        # Consider the following example (pack_size=6):
        # n_blocks = [1, 2, 3, 1, 2, 3]
        #
        # The i-th row of `mask` shows what pack members are active in the i-th block:
        # mask = [
        #     True,  True,  True, True,  True,  True,
        #     False, True,  True, False, True,  True,
        #     False, False, True, False, False, True,
        # ]
        #
        # `counts` shows the number of pack members active in the i-th block:
        # counts = [6, 4, 2]
        #
        # `counts` is needed for using `nonzero_static` instead of `nonzero`,
        # which reduces the number of host-device synchronizations in this function
        # to one (triggered when computing `counts`).
        mask = (
            torch.arange(self.max_n_blocks, device=self.n_blocks.device)[:, None]
            < self.n_blocks[None]
        )
        counts = mask.long().sum(1).tolist()
        return [
            (
                None  # None means that all members are active in the given block.
                if counts[i_block] == self.pack_size
                else torch.nonzero_static(mask[i_block], size=counts[i_block])[
                    :, 0
                ].clone()  # Ensure that the result is a non-inference tensor.
            )
            for i_block in range(self.max_n_blocks)
        ]

    def _get_block_idx(self, i_block: int) -> None | Tensor:
        assert self.n_blocks is not None
        cache = self._block_idx_cache.get(self.n_blocks)
        if cache is None:
            cache = self._compute_block_idx_list()
            self._block_idx_cache[self.n_blocks] = cache
        return cache[i_block]

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        assert pack_idx is None, 'Currently, member indexing is not supported'

        for i_block, block in enumerate(self._iter_blocks()):
            if self.n_blocks is None:
                x = block(x)
            else:
                block_idx = self._get_block_idx(i_block)
                if block_idx is None:
                    x = block(x)
                elif block_idx.numel() > 0:
                    x = x.index_copy(
                        PACK_DIM, block_idx, block(x[block_idx], block_idx)
                    )
                else:
                    break
        return x


class LinearEmbeddingsPack(ModulePack):
    """A pack of linear feature embeddings."""

    def __init__(
        self,
        n_features: int,
        d_embedding: int | list[int],
        *,
        max_d_embedding: None | int = None,
        pack_size: int,
    ) -> None:
        d_embedding_buffer, max_d_embedding = _prepare_dimension_pack(
            d_embedding, max_d_embedding, pack_size
        )

        super().__init__()
        self.weight = ParameterPack(torch.empty(pack_size, n_features, max_d_embedding))
        self.bias = ParameterPack(torch.empty(pack_size, n_features, max_d_embedding))
        self.d_embedding = d_embedding_buffer
        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return get_pack_size(self.weight)

    @property
    def max_d_embedding(self) -> int:
        return self.weight.shape[-1]

    def get_output_shape(self) -> torch.Size:
        return self.weight.shape[1:]  # ty:ignore[invalid-return-type]

    def reset_parameters(self) -> None:
        if self.d_embedding is None:
            d_rsqrt = self.max_d_embedding**-0.5
            for p in (self.weight, self.bias):
                nn.init.uniform_(p, -d_rsqrt, d_rsqrt)

        else:
            d_rsqrt = self.d_embedding.float().rsqrt()
            for p in (self.weight, self.bias):
                single_shape = p.shape[1:]
                pack_view_idx = (slice(None), *((None,) * (p.ndim - 1)))
                p_init = torch.lerp(
                    -d_rsqrt[*pack_view_idx].expand(-1, *single_shape),
                    d_rsqrt[*pack_view_idx].expand(-1, *single_shape),
                    torch.rand(
                        self.pack_size,
                        *single_shape,
                        dtype=p.dtype,
                        device=p.device,
                    ),
                )
                p.data.copy_(p_init)

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        assert x.ndim == 3  # (pack_size, batch_size, n_features)
        assert x.shape[-1] == self.weight.shape[-2]

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        d_embedding = _get_tensor_pack(self.d_embedding, pack_idx)

        x = torch.addcmul(
            bias.unsqueeze(BATCH_DIM), weight.unsqueeze(BATCH_DIM), x[..., None]
        )
        if d_embedding is not None:
            output_mask = (
                torch.arange(self.max_d_embedding, device=x.device)[None]
                < d_embedding[:, None]
            )
            input_dim = -2
            x = x * output_mask.float().unsqueeze(BATCH_DIM).unsqueeze(input_dim)

        return x


class LinearReLUEmbeddingsPack(ModulePack):
    """A pack of Linear-ReLU feature embeddings."""

    def __init__(self, *args, concat_input: bool = False, **kwargs) -> None:
        super().__init__()
        if concat_input:
            linear_d_embedding = kwargs['d_embedding']
            linear_max_d_embedding = kwargs.get('max_d_embedding')
            if isinstance(linear_d_embedding, int):
                linear_d_embedding -= 1
            else:
                linear_d_embedding = [x - 1 for x in linear_d_embedding]
            if linear_max_d_embedding is not None:
                linear_max_d_embedding -= 1
            kwargs = kwargs | {
                'd_embedding': linear_d_embedding,
                'max_d_embedding': linear_max_d_embedding,
            }
        self.linear = LinearEmbeddingsPack(*args, **kwargs)  # ty:ignore[invalid-argument-type]
        self.activation = nn.ReLU()
        self._concat_input = concat_input

    @property
    def pack_size(self) -> int:
        return self.linear.pack_size

    @property
    def max_d_embedding(self) -> int:
        return self.linear.max_d_embedding

    def get_output_shape(self) -> torch.Size:
        return self.linear.get_output_shape()

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        x_input = x
        x = self.activation(self.linear(x, pack_idx))
        if self._concat_input:
            x = torch.cat([x_input[..., None], x], dim=-1)
        return x


class CosineEmbeddings(nn.Module):
    """Cosine feature embeddings."""

    def __init__(self, n_features: int, d_embedding: int, *, init_scale: float) -> None:
        # The original feature is concatenated to the trainable part of its embedding.
        # This is reflected in the following constant for better clarity of some
        # implementation details of this class.
        concat_input = True

        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n_features, d_embedding - int(concat_input))
        )
        self.bias = nn.Parameter(torch.empty(self.weight.shape))
        self.elementwise_affine_weight = nn.Parameter(
            torch.empty(
                *self.weight.shape[:-1], self.weight.shape[-1] + int(concat_input)
            )
        )
        self.elementwise_affine_bias = nn.Parameter(
            torch.empty(self.elementwise_affine_weight.shape)
        )

        self.d_embedding = d_embedding
        self.init_scale = init_scale
        self._concat_input = concat_input

        self.reset_parameters()

    def get_output_shape(self) -> torch.Size:
        return torch.Size(
            (self.weight.shape[0], self.weight.shape[1] + int(self._concat_input))
        )

    def reset_parameters(self) -> None:
        bound = 3.0
        nn.init.trunc_normal_(self.weight, a=-bound, b=bound)
        with torch.inference_mode():
            self.weight *= self.init_scale
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.elementwise_affine_weight)
        nn.init.zeros_(self.elementwise_affine_bias)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2  # (batch_size, n_features)
        assert x.shape[-1] == self.weight.shape[0]

        x_input = x
        # x: (B, n_features) -> (B, n_features, d_embedding - int(concat_input))
        x = (
            self.weight * x[..., None]
            if self.bias is None
            else torch.addcmul(self.bias, self.weight, x[..., None])
        )
        x = torch.cos(2 * torch.pi * x)
        if self._concat_input:
            # The input goes first to avoid the potential subsequent masking.
            x = torch.cat([x_input[..., None], x], dim=-1)
        x = torch.addcmul(
            self.elementwise_affine_bias, self.elementwise_affine_weight, x
        )

        return x


class CosineEmbeddingsPack(ModulePack):
    """A pack of cosine feature embeddings.

    These embeddings are proposed in the TabPack paper.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int | list[int],
        *,
        max_d_embedding: None | int = None,
        init_scale: float | list[float],
        pack_size: int,
    ) -> None:
        concat_input = True

        d_embedding_buffer, max_d_embedding = _prepare_dimension_pack(
            d_embedding, max_d_embedding, pack_size
        )

        super().__init__()
        self.weight = ParameterPack(
            torch.empty(pack_size, n_features, max_d_embedding - int(concat_input))
        )
        self.bias = ParameterPack(torch.empty(self.weight.shape))
        self.elementwise_affine_weight = ParameterPack(
            torch.empty(
                *self.weight.shape[:-1], self.weight.shape[-1] + int(concat_input)
            )
        )
        self.elementwise_affine_bias = ParameterPack(
            torch.empty(self.elementwise_affine_weight.shape)
        )

        self.d_embedding = d_embedding_buffer
        self.init_scale = (
            init_scale
            if isinstance(init_scale, float)
            else BufferPack(torch.tensor(init_scale))
        )

        self._concat_input = concat_input

        self.reset_parameters()

    @property
    def pack_size(self) -> int:
        return get_pack_size(self.weight)

    @property
    def max_d_embedding(self) -> int:
        return self.weight.shape[-1] + int(self._concat_input)

    def get_output_shape(self) -> torch.Size:
        return torch.Size(
            (self.weight.shape[1], self.weight.shape[2] + int(self._concat_input))
        )

    def reset_parameters(self) -> None:
        bound = 3.0
        nn.init.trunc_normal_(self.weight, a=-bound, b=bound)
        with torch.inference_mode():
            if isinstance(self.init_scale, int | float):
                self.weight *= self.init_scale
            else:
                self.weight *= self.init_scale[:, *((None,) * (self.weight.ndim - 1))]
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.elementwise_affine_weight)
        nn.init.zeros_(self.elementwise_affine_bias)

    def forward(self, x: Tensor, pack_idx: None | Tensor = None) -> Tensor:
        assert x.ndim == 3  # (pack_size, batch_size, n_features)
        assert x.shape[-1] == self.weight.shape[-2]

        weight = _get_tensor_pack(self.weight, pack_idx)
        bias = _get_tensor_pack(self.bias, pack_idx)
        elementwise_affine_weight = _get_tensor_pack(
            self.elementwise_affine_weight, pack_idx
        )
        elementwise_affine_bias = _get_tensor_pack(
            self.elementwise_affine_bias, pack_idx
        )
        d_embedding = _get_tensor_pack(self.d_embedding, pack_idx)

        x_input = x
        x = torch.addcmul(
            bias.unsqueeze(BATCH_DIM), weight.unsqueeze(BATCH_DIM), x[..., None]
        )
        x = torch.cos(2 * torch.pi * x)
        if self._concat_input:
            # The input goes first to avoid the potential subsequent masking.
            x = torch.cat([x_input[..., None], x], dim=-1)
        x = torch.addcmul(
            elementwise_affine_bias.unsqueeze(BATCH_DIM),
            elementwise_affine_weight.unsqueeze(BATCH_DIM),
            x,
        )

        if d_embedding is not None:
            output_mask = (
                torch.arange(self.max_d_embedding, device=x.device)[None]
                < d_embedding[:, None]
            )
            input_dim = -2
            x = x * output_mask.float().unsqueeze(BATCH_DIM).unsqueeze(input_dim)

        return x


# >>> TabPack utilities


def module_pack_load_state_dict(
    module: ModulePack,
    state_dict: dict[str, Tensor],
    *,
    pack_idx: Tensor,
    state_dict_idx: Tensor | None = None,
) -> None:
    """Load a state dict to a module pack."""
    state_dict = state_dict.copy()
    for name, x in itertools.chain(module.named_parameters(), module.named_buffers()):
        if isinstance(x, ParameterPack | BufferPack):
            x.data[pack_idx] = state_dict.pop(name)[
                pack_idx if state_dict_idx is None else state_dict_idx
            ]
    if state_dict:
        raise RuntimeError(f'Unused state dict keys: {", ".join(state_dict)}')


def make_keep_pack_idx(pack_size: int, remove_pack_idx: Tensor) -> Tensor:
    """Compute pack indices to keep from pack indices to remove."""
    device = remove_pack_idx.device
    mask = torch.ones(pack_size, dtype=torch.bool, device=device)
    mask[remove_pack_idx] = False
    return torch.arange(pack_size, device=device)[mask].clone()


def module_pack_remove(
    module: nn.Module, pack_idx: Tensor
) -> dict[ParameterPack, ParameterPack]:
    """Remove some members from a module pack."""
    assert len(pack_idx) > 0

    keep_pack_idx = None
    old_to_new = {}
    for name, x in itertools.chain(module.named_parameters(), module.named_buffers()):
        is_parameter_pack = isinstance(x, ParameterPack)
        is_buffer_pack = isinstance(x, BufferPack)
        if is_parameter_pack or is_buffer_pack:
            if keep_pack_idx is None:
                keep_pack_idx = make_keep_pack_idx(get_pack_size(x), pack_idx)
            # `type(x)` does not work because of how buffers work in PyTorch.
            new_x = (ParameterPack if is_parameter_pack else BufferPack)(
                x.data[keep_pack_idx].clone()
            )
            if '.' in name:
                submodule_name, attr = name.rsplit('.', 1)
                submodule = module.get_submodule(submodule_name)
            else:
                submodule = module
                attr = name
            setattr(submodule, attr, new_x)

            if isinstance(x, ParameterPack):
                assert isinstance(new_x, ParameterPack)
                old_to_new[x] = new_x

    for submodule in module.modules():
        if isinstance(submodule, PackView):
            submodule._pack_size -= len(pack_idx)

    return old_to_new


@contextlib.contextmanager
def module_pack_select(module: nn.Module, pack_idx: Tensor):
    """Temporarily use only some of the pack members, and ignore the rest."""
    for submodule in module.modules():
        assert not submodule.training, 'The function is not applicable during training'

    original_modules_and_tensors = []
    original_pack_views = []
    if pack_idx is not None:
        for name, x in itertools.chain(
            module.named_parameters(), module.named_buffers()
        ):
            is_parameter_pack = isinstance(x, ParameterPack)
            is_buffer_pack = isinstance(x, BufferPack)
            if is_parameter_pack or is_buffer_pack:
                if '.' in name:
                    submodule_name, attr = name.rsplit('.', 1)
                    submodule = module.get_submodule(submodule_name)
                else:
                    submodule = module
                    attr = name
                setattr(
                    submodule,
                    attr,
                    # NOTE: type(x) does not work because of how buffers work in PyTorch
                    (ParameterPack if is_parameter_pack else BufferPack)(
                        x.data[pack_idx]
                    ),
                )
                original_modules_and_tensors.append((submodule, attr, x))
        for submodule in module.modules():
            if isinstance(submodule, PackView):
                original_pack_views.append((submodule, submodule.pack_size))
                submodule._pack_size = len(pack_idx)

    try:
        yield

    finally:
        for submodule, attr, parameter in original_modules_and_tensors:
            setattr(submodule, attr, parameter)
        for submodule, pack_size in original_pack_views:
            submodule._pack_size = pack_size
