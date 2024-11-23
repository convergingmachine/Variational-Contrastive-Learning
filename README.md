# Variational Contrastive Learning

Welcome to the Variational Contrastive Learning repository! This project provides tools and scripts to run benchmarks on the ImageNet ILSVRC2012 dataset using various contrastive learning methods. Below, you'll find instructions for setting up the environment, running benchmarks, and configuring experiments.

## Benchmark Results

**Note**: Evaluation settings are based on these papers:

- Linear: [SimCLR](https://arxiv.org/abs/2002.05709)
- Finetune: [SimCLR](https://arxiv.org/abs/2002.05709)
- KNN: [InstDisc](https://arxiv.org/abs/1805.01978)

| Model           | Backbone | Batch Size | Epochs | Linear Top1 | Finetune Top1 | kNN Top1 | Checkpoint                                                                                                                                                              |
| --------------- | -------- | ---------- | ------ | ----------- | ------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VCL - beta div.  | Res50    | 256        | 100    | -        | -             | -     | [link](https://huggingface.co/ogrenenmakine/vcl/resolve/main/beta_vcl_e100.ckpt)      | 
| VCL      | Res50    | 256        | 100    | 61.1        | -             | 41.2     | [link](https://huggingface.co/ogrenenmakine/vcl/resolve/main/vcl_e100.ckpt)      | 
| _Other SSLs_ |
| BarlowTwins     | Res50    | 256        | 100    | 62.9        | 72.6          | 45.6     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| BYOL            | Res50    | 256        | 100    | 62.5        | 74.5          | 46.0     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)        |
| DINO            | Res50    | 128        | 100    | 68.2        | 72.5          | 49.9     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt)       |
| MAE             | ViT-B/16 | 256        | 100    | 46.0        | 81.3          | 11.2     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)           |
| MoCoV2          | Res50    | 256        | 100    | 61.5        | 74.3          | 41.8     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)      |
| SimCLR\*        | Res50    | 256        | 100    | 63.2        | 73.9          | 44.8     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)      |
| SimCLR\* + DCL  | Res50    | 256        | 100    | 65.1        | 73.5          | 49.6     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)         |
| SimCLR\* + DCLW | Res50    | 256        | 100    | 64.5        | 73.2          | 48.5     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)        |
| SwAV            | Res50    | 256        | 100    | 67.2        | 75.4          | 49.5     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)        |
| TiCo            | Res50    | 256        | 100    | 49.7        | 72.7          | 26.6     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/checkpoints/epoch%3D99-step%3D250200.ckpt)        |
| VICReg          | Res50    | 256        | 100    | 63.0        | 73.7          | 46.3     | [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)      |

_\*We use square root learning rate scaling instead of linear scaling as it yields


## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Running Benchmarks](#running-benchmarks)
  - [Local Execution](#local-execution)
  - [SLURM Execution](#slurm-execution)
- [Configuration Options](#configuration-options)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, you'll need to set up your environment. We recommend using Conda for managing dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/variational-contrastive-learning.git
   cd variational-contrastive-learning
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create --name lightly-env python=3.8
   conda activate lightly-env
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

To run the benchmark, you need to download the ImageNet ILSVRC2012 dataset. You can obtain it from the official website:

[Download ImageNet ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/)

Place the dataset in your desired directory structure:

```
/datasets/imagenet/
    ├── train/
    └── val/
```

## Running Benchmarks

You can run benchmarks either locally or using a SLURM cluster.

### Local Execution

To run the benchmark locally, use the following command:

```bash
python main.py --epochs 100 --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --num-workers 12 --devices 2 --batch-size-per-device 128 --skip-finetune-eval
```

### SLURM Execution

For execution on a SLURM cluster, create a script named `run_imagenet.sh` with the following content:

```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # Must match --devices argument
#SBATCH --ntasks-per-node=2     # Must match --devices argument
#SBATCH --cpus-per-task=16      # Must be >= --num-workers argument
#SBATCH --mem=0

eval "$(conda shell.bash hook)"

conda activate lightly-env
srun python main.py --epochs 100 --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --num-workers 12 --devices 2 --batch-size-per-device 128
conda deactivate
```

Submit the job with:

```bash
sbatch run_imagenet.sh
```

## Configuration Options

You can customize your experiments using various command-line arguments:

- **Specify Methods**: Use the `--methods` flag to choose specific contrastive learning methods.
  ```bash
  python main.py --epochs 100 --batch-size-per-device 128 --methods simclr byol
  ```

- **Skip Training/Evaluation Steps**: You can skip certain steps to save time.
  ```bash
  python main.py --batch-size-per-device 128 \
      --epochs 0              # no pretraining
      --skip-knn-eval         # no KNN evaluation
      --skip-linear-eval      # no linear evaluation
      --skip-finetune-eval    # no finetune evaluation
  ```

## Contributing

We welcome contributions to improve this project! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Thank you for using Variational Contrastive Learning! We hope this tool helps advance your research and projects in machine learning. If you have any questions or feedback, please don't hesitate to reach out.
