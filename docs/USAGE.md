# Usage manual

This manual covers installing, running and developing the project. For an overview of the
project, see the [README](../README.md).

All commands are run from the **repository root**.

- [Requirements](#requirements)
- [Desktop application](#desktop-application)
- [Training](#training)
- [Dataset layout](#dataset-layout)
- [Training outputs](#training-outputs)
- [Tests and code quality](#tests-and-code-quality)
- [Makefile reference](#makefile-reference)
- [Troubleshooting](#troubleshooting)

## Requirements

| Requirement | Notes |
|---|---|
| Linux | Developed and tested on Linux. The GUI is a PyQt5 desktop app. |
| Python **3.12** | The Makefile calls `python3.12` directly. |
| GNU Make, `git` | Used to set up the environment and clone the extractors. |
| `wget` | Needed by `make build`, which downloads the IML-ViT and Swin weights. |
| DejaVu fonts | The header of the fraud-map preview uses `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` (Debian/Ubuntu package `fonts-dejavu-core`). |
| Internet connection | Needed for the dependencies, the upstream repositories and the pretrained weights (partly from Google Drive). |
| NVIDIA GPU with CUDA (optional) | Used automatically when available. The application also runs on the CPU, but more slowly. |

Several gigabytes of free disk space are recommended for the virtual environment, PyTorch and the
pretrained weights.

## Desktop application

### 1. Build

```bash
make build_app
```

This command:

1. creates a virtual environment in `.venv/` and installs the project (`pip install -e .`),
2. clones **MMFusion-IML** into `external/MMFusion` and downloads its checkpoint and its
   Noiseprint++ weights,
3. installs the MMFusion requirements.

### 2. Run

```bash
make run_app
```

### 3. Classify an image

1. Click the preview area and choose an image of an identity document
   (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` or `.webp`).
2. Click **Classify**. Inference runs in a background thread, so the window stays responsive.
3. The result appears below the image:
   - **"The ID card is genuine"**: the document is classified as genuine.
   - **"The ID card is fake"**: the document is classified as forged, and a **fraud map** is
     shown. Regions that the model considers manipulated are highlighted in red.

The application uses MMFusion as the feature extractor and the trained MLP head in
`checkpoint/checkpoint.pth`.

## Training

### 1. Build

```bash
make build
```

This creates `.venv/` with the development dependencies (`pytest`, `mypy`, `ruff`). It also
downloads **all four** extractors into `external/`, together with their weights:

| Extractor | Directory | Embedding size |
|---|---|---|
| TruFor | `external/TruFor` | 512 |
| MMFusion | `external/MMFusion` | 512 |
| Swin Transformer (Swin-T) | `external/SwinT` | 768 |
| IML-ViT | `external/IML_VIT` | 768 |

> `make build` skips the download if `external/` already exists. If you built only the
> application first, run `make clean` (or delete `external/`) before `make build`.

### 2. Run

```bash
.venv/bin/python3.12 scripts/train.py \
    --extractor EXTRACTOR_NAME \
    --data_path DATA_PATH \
    --batch_size BATCH_SIZE \
    --epochs EPOCHS \
    --lr LR \
    [--dropout DROPOUT] \
    --device {cpu,cuda}
```

| Argument | Required | Description |
|---|:---:|---|
| `--extractor` | Yes | Feature extractor: `trufor`, `mmfusion`, `swint` or `imlvit`. |
| `--data_path` | Yes | Path to the dataset root (see [Dataset layout](#dataset-layout)). |
| `--batch_size` | Yes | Training batch size. |
| `--epochs` | Yes | Maximum number of epochs. Early stopping may end training sooner. |
| `--lr` | Yes | Learning rate for the Adam optimiser. |
| `--dropout` | No | Dropout rate of the MLP head (default `0.2`). |
| `--device` | Yes | `cpu` or `cuda`. |

Example (configuration B from the thesis, which produced the shipped checkpoint):

```bash
.venv/bin/python3.12 scripts/train.py \
    --extractor mmfusion --data_path /data/dataset \
    --batch_size 32 --epochs 30 --lr 5e-4 --dropout 0.2 --device cuda
```

### What happens during training

1. The dataset is loaded and split with a fixed seed (`28`) into **75 % train, 10 % validation
   and 15 % test**. The split is stratified by label and attack type, so every extractor sees
   exactly the same split.
2. Embeddings are extracted **once** for the train and validation sets and kept in memory. The
   extractor is then released to free GPU memory.
3. The MLP head is trained with Adam and cross-entropy loss:
   - **ReduceLROnPlateau**: the learning rate drops 10× after 2 epochs without improvement,
     down to a minimum of `1e-6`,
   - **Early stopping**: training stops after 5 epochs without an improvement of at least
     `0.001` in validation loss.
4. The best checkpoint (lowest validation loss) is loaded and evaluated on the test set.

## Dataset layout

The dataset is not included in this repository (see [LICENSING.md](../LICENSING.md#datasets)).
The loader expects this structure:

```text
dataset/
├── positive/                 # genuine documents
│   ├── fantasy_id/           # any name: the source dataset
│   │   ├── img_0001.png
│   │   └── ...
│   └── midv2020/
│       └── ...
└── attack/                   # forged documents
    ├── fantasy_id/           # any name: the source dataset
    │   ├── text_only/        # only text fields changed
    │   ├── photo_only/       # only the holder's portrait changed
    │   └── both_detp/        # both: any folder starting with "both" (both, both_*)
    └── midv2020/
        ├── text_only/
        ├── photo_only/
        └── both/
```

- Images are found recursively inside each source directory.
- Attack-type folder names are case-insensitive.
- Supported extensions: `.jpg .jpeg .png .bmp .tif .tiff .webp`.

## Training outputs

Results are written to `experiments/<extractor>/`:

```text
experiments/mmfusion/
├── config.json                            # training configuration
├── metrics.json                           # per-epoch train/validation metrics
├── summary.json                           # best epoch and its metrics
├── graphs/                                # loss and metric curves
├── checkpoints/checkpoint.pth             # best MLP head weights
└── test/
    ├── test_metrics.json
    ├── test_confusion_matrix_values.json
    └── test_confusion_matrix.png
```

The forged class is the **positive** class for precision, recall and F1.

To use your own head in the application, copy `checkpoints/checkpoint.pth` from an **MMFusion**
experiment to `checkpoint/checkpoint.pth`.

## Tests and code quality

```bash
make test_data_processing   # quick: no extractors needed
make test_heads
make test_training
make test_extractors        # needs all extractors (runs make build)
make test_all               # everything

make ruff_check             # lint (rules E, F, I; line length 100)
make ruff_fix               # lint and fix automatically
make mypy_check             # strict static type checking
```

## Makefile reference

| Target | Description |
|---|---|
| `build` | Development environment plus all four extractors and their weights. |
| `build_app` | Application environment plus MMFusion only. |
| `run_app` | Starts the desktop application. |
| `test_all`, `test_data_processing`, `test_extractors`, `test_heads`, `test_training` | Test suites (pytest). |
| `ruff_check`, `ruff_fix`, `mypy_check` | Linting and type checking. |
| `zip` | Packs the sources into `submission.zip`. |
| `clean` | Removes `.venv/`, `external/`, `meta_data/` and all caches. |

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3.12: command not found` | Install Python 3.12 (for example `sudo apt install python3.12 python3.12-venv`). |
| Google Drive download fails (`gdown`) | Drive limits how often a file can be downloaded. Wait and retry, or download the files from the links in `scripts/download_extractors.py` by hand into `external/MMFusion/ckpt` and `external/MMFusion/pretrained`. |
| "Extractors already exist" but training cannot find an extractor | `external/` was created by `make build_app` and contains only MMFusion. Run `make clean && make build`. |
| CUDA out of memory | MMFusion processes images at 2048 px. Use `--device cpu`, a smaller batch size, or a GPU with more memory. |
| Font error when a fraud map is shown | Install `fonts-dejavu-core`. |
| The Qt window does not start (`xcb` plugin error) | Install the Qt xcb runtime libraries, for example `sudo apt install libxcb-xinerama0 libxkbcommon-x11-0`. |
