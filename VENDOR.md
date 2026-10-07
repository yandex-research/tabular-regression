# Third-party code and local adaptations

The vendored sources support the published experiment configurations. Their
original licenses apply to the corresponding files. Changes described below
document the implementation used for our benchmark; publication preparation
preserves its training computations.

## Sources

| Component | Retained source | Paper / upstream | License |
| --- | --- | --- | --- |
| HL-Gaussian | `vendor/hl_loss.py` | [Imani and White, 2018](https://arxiv.org/abs/1806.04613); implementation derived from [hl-gauss-pytorch](https://github.com/lucidrains/hl-gauss-pytorch) by Phil Wang | [MIT](vendor/licenses/hl-gauss-LICENSE) |
| Decoder vocabularies and numeric serialization | `vendor/decoder_vocabs.py`, `vendor/optformer/` | [Song and Bahri, Decoding-based Regression](https://arxiv.org/abs/2501.19383); [OptFormer source](https://github.com/google-research/optformer/tree/main/optformer/decoding_regression) | [Apache-2.0](vendor/optformer/LICENSE) |
| TabM reference | `vendor/tabm_reference.py` | [TabM paper](https://arxiv.org/abs/2410.24210); [reference implementation](https://github.com/yandex-research/tabm/blob/main/tabm_reference.py) | [Apache-2.0](vendor/licenses/tabm-LICENSE) |
| TabPack | `vendor/tabpack/` | [TabPack paper](https://arxiv.org/abs/2607.05380); [author repository](https://github.com/yandex-research/tabpack) | No standalone code license file in the inspected upstream snapshot; retained notices apply. |
| Muon | `vendor/muon.py` | Adopted through [tabular-dl-optimizers](https://github.com/yandex-research/tabular-dl-optimizers/blob/c34e33152db9eea8ed4b2d742b9bef5628d71fb5/vendor/muon.py); original [Keller Jordan source at f90a42b](https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py) | [MIT](vendor/licenses/muon-LICENSE) |
| RealMLP | `vendor/pytabkit/` | [RealMLP paper](https://arxiv.org/abs/2407.04491); [pytabkit](https://github.com/dholzmueller/pytabkit) | [Apache-2.0](vendor/pytabkit/LICENSE.txt) |
| Optimal transport | `vendor/pot/` | [Python Optimal Transport](https://github.com/PythonOT/POT), dependency of ROT-REG | [MIT](vendor/pot/LICENSE) |
| Ordered sets | `vendor/ordered_set.py` | [ordered-set](https://github.com/rspeer/ordered-set), version 4.1.0, dependency of the OptFormer serializers | [MIT](vendor/licenses/ordered-set-LICENSE) |

Source links identify the upstream projects; file headers record pinned original
revisions where available. The local files are retained research snapshots and
should be used to reproduce these configurations. We do not replace them with
the latest upstream versions.

## Decoder and OptFormer

The decoder experiments implement the autoregressive regression approach of
Song and Bahri (2025). The author's TensorFlow/Keras `AttentionDecoder` in
`optformer/decoding_regression/models.py` provided the design adapted in
`bin/decoder.py`: feature encoding, projection into a token sequence, learned
token and positional embeddings, causal attention blocks, position-wise
cross-entropy training, constrained autoregressive sampling, and float decoding.
Our published configurations use the shared PyTorch MLP backbone and benchmark
data processing and optimization. Decoder width, depth, head count, dropout,
sampling parameters, and the normalized decoder's vocabulary parameters are
tuned for this implementation under the benchmark's hyperparameter-selection protocol.

The following adaptations are retained:

- `vendor/decoder_vocabs.py` derives from the authors'
  `optformer/decoding_regression/vocabs.py`. The unnormalized vocabulary exposes
  base, exponent length, and mantissa length as explicit configuration parameters.
  The normalized vocabulary exposes its base and token length. NumPy masks
  replace the upstream array typing annotations.
- The numeric serializer utilities use the local `vendor.optformer` namespace,
  a vendored ordered set, and omit optional Gin configuration decorators.
  Their numeric encodings and token masks are preserved.
- `BatchNormalizedVocab` and `BatchUnnormalizedVocab` in `bin/decoder.py`
  reconstruct a batch of decoded numbers with Torch tensor operations on the
  current device. Cached base powers replace repeated Python/NumPy scalar
  reconstruction.
- PyTorch `MultiheadAttention`, `LayerNorm`, tensor masks, and
  `torch.multinomial` implement the attention and sampling operations. The
  local attention width is the total embedding width, divided among heads;
  the Keras reference passes `units` as the per-head key/value dimension.
  The local LayerNorm uses PyTorch's default epsilon, while the reference uses
  `1e-6` (the retained PyTorch version defaults to `1e-5`). Initializers and
  random-number streams follow the respective frameworks.
- Evaluation uses the sampling configuration saved for each experiment,
  including temperature and the number of samples used for point prediction.

The autoregressive objective and float representations follow the cited method.
The framework port and the layer conventions above define the concrete model
evaluated here; the saved configurations reproduce this local implementation.
The cited method is evaluated through this tuned PyTorch adaptation; the
published search spaces and selected configurations specify its model sizes
and inference settings. Framework conventions are documented above for
reproducibility. Original Google copyright and Apache notices are retained
on the derived vocabulary and decoder code.

## HL-Gaussian

The algorithm is the Gaussian Histogram Loss introduced by Imani and White
(2018). The retained PyTorch implementation derives from Phil Wang's
`lucidrains/hl-gauss-pytorch`, rather than the TensorFlow code in
`marthawhite/Histogram_loss`.

Local changes use native Torch broadcasting in place of Einx operations,
explicit device selection, stable probability normalization, and an optional
reverse-KL switch. The Gaussian target distribution is obtained from CDF
differences at the bin boundaries. The retained configurations specify the
actual bin count, support and smoothing; preparation preserves these operations.
The inherited Appendix A reference to
[Stop Regressing (2024)](https://arxiv.org/abs/2403.03950) identifies the later
implementation example followed by the upstream PyTorch package. The method
remains Gaussian Histogram Loss from Imani and White (2018), with normalized
Gaussian probability mass per bin and categorical cross-entropy. The MLP
experiments explicitly tune `sigma`; they do not rely on the package's default
sigma-to-bin-width ratio mentioned in its RL-related comments.

## TabM, TabPack, and RealMLP

`tabm_reference.py` retains the reference model and initialization used by the
experiments. Its numerical implementation matches the upstream reference;
the local snapshot omits the later deprecation warning and uses local documentation.

TabPack retains the authors' packing, feature embeddings, optimizer machinery,
and online-ensemble workflow. Local changes add the regression objective heads,
histogram/quantile/AdaQR losses, scale regularization, and uncertainty reporting
required by this benchmark. `run_tabpack_experiment.py` and `report_adapter.py`
are local integration utilities for running the TOML experiments and exposing
the selected online ensemble in the common report format. The main comparisons
record hyperparameter search in `main/config.toml` under `sampler.space`; the
scale-regularization ablation mostly retains explicit sampled members under `configs`.

The modified pytabkit snapshot supplies RealMLP. Its local changes integrate
Muon, additional regression heads and losses, AdaQR scale initialization and
scale regularization, plus scalar prediction conversion for distributional
heads. The relevant code is in `models/nn_models/`,
`models/training/`, `models/sklearn/sklearn_interfaces.py`, and
`models/optim/`. The data utility modules required by RealMLP are retained.
The dataloader factory matches the version used in the source RealMLP experiments.
Unrelated benchmarking, plotting and scheduling tools have been removed.
The global-sigma ablation uses `bin/realmlp_adaqr_global.py`, retained from the
matching research branch. It reuses the quantile-sized RealMLP head and replaces
its loss factory within the training call with the fixed-sigma smoothed objective.
`loss_sigma` is an HPO-selected constant, not a learned or input-dependent scale.

## Optimizer dependencies

The published configurations use AdamW and Muon, including their RealMLP and
TabPack integrations. The broader optimizer registry was inherited from the
research code for [tabular-dl-optimizers](https://github.com/yandex-research/tabular-dl-optimizers).
Its additional implementations are legacy dependencies retained to preserve the
research snapshot's imports and behavior. They are not used by the published
configurations and are not methods or contributions of this work.

Original source links, adaptation notes, and licenses remain in the corresponding
files. The following third-party implementations support this retained registry:

| File | Upstream | License |
| --- | --- | --- |
| `adan.py` | [Adan](https://github.com/sail-sg/Adan) | Apache-2.0, header retained; the complete license is also available in the vendored Apache components. |
| `ademamix.py` | [AdEMAMix](https://github.com/apple/ml-ademamix) | [Upstream license](vendor/licenses/ademamix-LICENSE) |
| `adopt.py` | [ADOPT](https://github.com/iShohei220/adopt) | [Apache-2.0](vendor/licenses/adopt-LICENSE) |
| `c_muon.py` | [C-Optim](https://github.com/kyleliang919/C-Optim) | [MIT](vendor/licenses/c-muon-LICENSE) |
| `lion_pytorch.py` | [Lion](https://github.com/google/automl/tree/master/lion) | Apache-2.0, header retained. |
| `sam.py` | [SAM](https://github.com/davda54/sam) | [MIT](vendor/licenses/sam-LICENSE) |
| `soap.py` | [SOAP](https://github.com/nikhilvyas/SOAP) | [MIT](vendor/licenses/soap-LICENSE) |
