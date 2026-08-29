# TrackPermanence.mojo

[![mojoshelf](https://mojoshelf.org/badge/trackpermanence.svg)](https://mojoshelf.org/tins/trackpermanence) [![mojo nightly](https://mojoshelf.org/badge/trackpermanence/nightly.svg)](https://mojoshelf.org/tins/trackpermanence)

Pure-[Mojo](https://www.modular.com/mojo) inference for **"Offline Tracking
with Object Permanence"** (Liu & Caesar,
[arXiv:2310.01288](https://arxiv.org/abs/2310.01288)) — the learned
occlusion-recovery plugin for offline auto-labeling: a **Re-ID** network that
links a terminated tracklet to the later tracklet of the same object, and a
**track completion** network that regresses the occluded poses in between.
Trained in [TrackPermanence.py](https://github.com/labelrefinery/TrackPermanence.py)
(PyTorch, ArgoVerse 2 pseudo-occlusions); this repo runs the exported weights
with no dependencies beyond the Mojo standard library.

The paper does not name its model; *TrackPermanence* is this project's name.

## Parity with PyTorch

`pixi run parity` runs the exported validation samples through the Mojo
forward passes and compares with the recorded PyTorch outputs, for both the
trained smoke weights and a random-weights export (which stresses every op —
the trained completion residual is small):

| weights | Re-ID logit max diff | completion pose max diff |
| --- | --- | --- |
| trained (smoke) | 1.2e-7 | 7.5e-9 |
| random | 7.5e-9 | 4.8e-7 |

Ops implemented step-for-step: 2-layer ReLU MLP, `GRUCell` recurrence
(forward / reverse, U-GRU), `nn.MultiheadAttention` (batch_first, no masks —
single un-padded samples), `LayerNorm`, cubic-Hermite prior.

## Install as a mojoshelf tin

Published on [mojoshelf](https://mojoshelf.org/tins/trackpermanence) as `trackpermanence`:

```sh
pixi shelf add trackpermanence     # pixi mode (git source dependency)
shelf add trackpermanence          # or as a git submodule
```

Maintainers release new versions with `shelf publish` from the repo root
(see [getting started](https://mojoshelf.org/getting-started)).

## Usage

```sh
pixi run test     # unit tests (NN primitives, frames, features)
pixi run parity   # PARITY: PASS x2
pixi run demo     # examples/occluded.csv -> examples/completed.csv
```

CLI — the same CSV contract as
[OfflinePoly.mojo](https://github.com/labelrefinery/OfflinePoly.mojo)
(`track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf`, global coordinates):

```sh
mojo run -I src src/main.mojo data/weights.lft IN.csv OUT.csv [--threshold 0.5]
```

Every tracklet that terminates ≥ 1.5 s before the scene end is scored
against every same-class tracklet starting 1.5–12.5 s after it; links are
chosen greedily above the threshold (chains allowed), each gap is filled at
the scene frame rate, and merged tracks are written out. On the demo scene
(a 41-track AV2 val log with one track cut by a 4 s occlusion) the Mojo CLI
reproduces the Python plugin to 2e-6 m and recovers the cut track with
8 mm mean error against ground truth.

Where it sits: OfflinePoly's STWO re-links fragments within its 1 s
motion-model horizon; TrackPermanence is the learned re-ID/completion stage
for longer occlusions (the paper's Table VIII-style "learned re-ID" slot),
and its output feeds
[LabelFormer.mojo](https://github.com/labelrefinery/LabelFormer.mojo)'s
trajectory refiner.

## Weights

`data/weights.lft` (1.2 MB) are the smoke weights from `TrackPermanence.py`
(`configs/smoke.yaml`, 12 AV2 logs, ~1 min on a laptop CPU) plus parity
samples, in the LFT1 container shared with LabelFormer.mojo. They are trained
on AV2 (CC BY-NC-SA 4.0) and inherit that non-commercial term; the code is
MIT. Re-export after paper-scale training with
`uv run python scripts/export_mojo.py` in TrackPermanence.py.

## Deviations from the paper

Inherited from the training side: map-free (motion branch only), velocity
features and a cubic-Hermite prior for completion (on 10 Hz AV2 data the
prior alone cuts the occluded-pose error from 0.61 m linear to 0.24 m), and
fixed input scaling. See TrackPermanence.py's README for the measurements.

## Citation

```bibtex
@article{liu2023offline,
  title={Offline Tracking with Object Permanence},
  author={Liu, Xianzhong and Caesar, Holger},
  journal={arXiv preprint arXiv:2310.01288},
  year={2023}
}
```
