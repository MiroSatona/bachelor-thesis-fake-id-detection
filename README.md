# Usage Manual

For setting up and running the application or training, you must be located in the `source_code/` directory of the submitted solution.

## Requirements

- Linux-based operating system
- Python 3.12
- GNU Make
- Permissions to create and modify files in the `source_code` directory
- Internet connection for downloading the repository and required dependencies



## Training setup and execution

For setting up the environment of the training, execute this command:

```bash
make build
```

This Makefile target creates a Python virtual environment and installs all required dependencies. In the next step, this target installs or clones the feature extractors with their pretrained weights to the newly created directory `external`.

For running the training, a script `train.py` is used, which is located in the `scripts` directory. From the `source_code` directory, it can be executed with the following parameters:

>  For running the training, `make build` must be executed beforehand.


```bash
python3.12 scripts/train.py \
    --extractor EXTRACTOR_NAME \
    --data_path DATA_PATH \
    --batch_size BATCH_SIZE \
    --epochs EPOCHS \
    --lr LR \
    [--dropout DROPOUT] \
    --device {cpu,cuda}
```

where the table below specifies the parameters.

| Argument | Description |
|----------------|-------------------------------------------------------------------|
| `--extractor`  | Feature extractor name {`trufor`, `mmfusion`, `swint`, `imlvit`}. |
| `--data_path`  | Path to the training dataset.                                     |
| `--batch_size` | Training batch size.                                              |
| `--epochs`     | Number of training epochs.                                        |
| `--lr`         | Learning rate.                                                    |
| `--dropout`    | Dropout percentage for the MLP head.                              |
| `--device`     | Training device {`cpu`,`cuda`}.                                   |


## Application setup and execution

For setting up the environment of the application, execute this command:

```bash
make build_app
```

This Makefile target creates a Python virtual environment and installs all required dependencies. Next, this target installs or clones `MMFusion`, which is used as a feature extractor in this application, together with its pretrained weights to the newly created `external` directory.

For running the application, execute the following command:

>  For running the application, `make build_app` must be executed beforehand.


```bash
make run_app
```

After this command, the GUI will appear.