# Licensing

This repository contains several kinds of material, and they are licensed differently.
This file explains which license covers what and why.

| Part of the repository | License | Full text |
|---|---|---|
| Source code (`src/`, `scripts/`, `tests/`), `Makefile`, `pyproject.toml` and documentation | **GNU AGPL v3.0 only** (`AGPL-3.0-only`), with the additional permission below | [`LICENSE`](LICENSE) |
| Trained classification-head weights (`checkpoint/checkpoint.pth`) | **CC BY-NC-SA 4.0** | [`checkpoint/LICENSE`](checkpoint/LICENSE) |
| Screenshots in `docs/images/` | Not relicensed. They show sample images from the MIDV-2020 dataset; see [Datasets](#datasets) | — |
| Third-party feature extractors, downloaded into `external/` by `make build` / `make build_app` | Their own licenses; see [Third-party components](#third-party-components) | Upstream repositories |

Copyright © 2026 Martin Zůbek.

## Source code: GNU AGPL v3.0 only

The original code of this project is free software. You can redistribute it and modify it under the
terms of the **GNU Affero General Public License, version 3 only**, as published by the Free
Software Foundation.

The AGPL is a strong copyleft license:

- Anyone who distributes this code or a modified version of it must release the complete source
  code under the same license and keep all copyright and license notices.
- The same applies if a modified version runs as a network service (for example a web API for
  identity verification). Users of that service must be offered its source code (AGPL §13).
- Only version 3 applies. Later versions of the AGPL do not apply automatically.

The program is distributed WITHOUT ANY WARRANTY, without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [`LICENSE`](LICENSE) for details.

### Additional permission under GNU AGPL version 3, section 7

This project imports code from TruFor and MMFusion-IML at runtime. Their licenses are not
compatible with the AGPL (see below). So that combining them with this project is legally
consistent, the copyright holder grants the following additional permission:

> If you modify this Program, or any covered work, by linking or combining it with
> **TruFor** (<https://github.com/grip-unina/TruFor>) or **MMFusion-IML**
> (<https://github.com/IDT-ITI/MMFusion-IML>), or with a modified version of either, including their
> pretrained weights, the licensors of this Program grant you additional permission to convey the
> resulting work. The Corresponding Source for such a combination must include the source code of
> the covered work. It does not need to include those third-party components, which remain subject
> to their own licenses.

This permission only allows the combination. It grants **no** rights to TruFor or MMFusion-IML
themselves. Their terms, including the ban on commercial use, still apply in full.

## Trained weights: CC BY-NC-SA 4.0

`checkpoint/checkpoint.pth` holds the weights of the MLP classification head that was trained on
MMFusion embeddings (Configuration B in the thesis). These weights are licensed under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/):

- **Attribution**: credit the author and the thesis (see [Citation](README.md#citation)).
- **NonCommercial**: no commercial use.
- **ShareAlike**: derived weights (for example fine-tuned ones) must use the same license.

The non-commercial restriction is required. The weights were derived from MMFusion features, and
MMFusion is licensed for academic, non-commercial use only. They were also trained on datasets
whose terms restrict use to research.

## Third-party components

`make build` and `make build_app` download these projects and weights into `external/`. They are
**not** part of this repository, and their licenses are not affected by the license of this project.

| Component | Used for | License (upstream) |
|---|---|---|
| [TruFor](https://github.com/grip-unina/TruFor) (GRIP-UNINA) | Feature extractor | Custom research license. **Non-commercial, non-profit use only**, notices must be kept, and the original authors must be credited when results are published |
| [MMFusion-IML](https://github.com/IDT-ITI/MMFusion-IML) (CERTH-ITI) | Feature extractor **used by the application** | **Academic, non-commercial use only** (BSD-style conditions) |
| [IML-ViT](https://github.com/SunnyHaze/IML-ViT) | Feature extractor | MIT |
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (Microsoft) | Feature extractor | MIT |
| MAE ViT-B pretrained weights (Meta AI) | IML-ViT initialisation | See the upstream [MAE repository](https://github.com/facebookresearch/mae) |

Python dependencies (PyTorch, torchvision, timm, NumPy, OpenCV, PyQt5, …) are installed from PyPI
and keep their own licenses. Note that **PyQt5 is GPL v3 / commercial dual-licensed**. GPL v3 is
compatible with the AGPL v3 used here.

> [!IMPORTANT]
> Because the application depends on MMFusion and the trained weights are licensed for
> non-commercial use only, **the system as a whole can only be used for non-commercial purposes**,
> even though the source code of this repository is under the AGPL.

## Datasets

The datasets are **not** distributed with this repository. The model was trained on data based on:

- **FantasyID**: Korshunov et al., IJCB 2025, Idiap Research Institute. Parts of it are available
  only under an EULA restricted to non-commercial research.
- **MIDV family** (MIDV-500, MIDV-2020, MIDV-LAIT, MIDV-HOLO, MIDV-UP): Smart Engines and partners.
  Access requires accepting the dataset license. The documents in these datasets use fictional
  personal data.
- Additional forged samples, created by the author with generative tools on top of MIDV templates.

The screenshots in `docs/images/` show MIDV-2020 sample documents (fictional identities). They are
included only to illustrate the application, they remain subject to the MIDV-2020 terms, and
this project does not relicense them.

## Copyright of a school work (Czech law)

This project was created as a bachelor's thesis at the Brno University of Technology. It is a
*school work* under §60 of the Czech Copyright Act (Act No. 121/2000 Coll.). The statutory rights
of the university under that act are not affected by the licenses above.
