# Third-party components

CoFiDrive is implemented within the NAVSIM training and evaluation framework. The
release preserves the upstream NAVSIM package layout so its Hydra entry points and
evaluation tools remain usable. The original NAVSIM project is available at
<https://github.com/autonomousvision/navsim>. CoFiDrive's register-based model also
builds on ideas and implementation structure from DrivoR, *Driving on Registers*.
The original DrivoR project is available at <https://github.com/valeoai/DrivoR>.

The included `nuplan-devkit/` is a vendored upstream dependency from Motional.
Its own license notice remains at `nuplan-devkit/LICENSE.txt`; the upstream project
is <https://github.com/motional/nuplan-devkit>. The repository's top-level
`LICENSE` retains the Apache License 2.0 text used by the NAVSIM/DrivoR-derived
code. The license text does not replace any third-party attribution obligations.

The paper figures in `site/assets/` are produced from the CoFiDrive manuscript
and NAVSIM visualizations. The NAVSIM/OpenScene datasets and pretrained DINOv2
weights are **not** included. Obtain them from their respective publishers under
their own terms. Do not redistribute benchmark data, map assets, or model weights
by adding them to this repository.
