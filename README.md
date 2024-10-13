# Variational Contrastive Learning

Welcome to the Variational Contrastive Learning repository! This project provides tools and scripts to run benchmarks on the ImageNet ILSVRC2012 dataset using various contrastive learning methods. Below, you'll find instructions for setting up the environment, running benchmarks, and configuring experiments.

## Benchmark Results

.. csv-table:: Imagenet benchmark results.
  :header: "Model", "Backbone", "Batch Size", "Epochs", "Linear Top1", "Linear Top5", "Finetune Top1", "Finetune Top5", "kNN Top1", "kNN Top5", "Tensorboard", "Checkpoint"
  :widths: 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20

  "BarlowTwins", "Res50", "256", "100", "62.9", "84.3", "72.6", "90.9", "45.6", "73.9", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/events.out.tfevents.1692310273.Machine2.569794.0>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  
  "BYOL", "Res50", "256", "100", "62.5", "85.0", "74.5", "92.0", "46.0", "74.8", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/events.out.tfevents.1707923418.Machine2.3205.0>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  
  "DINO", "Res50", "128", "100", "68.2", "87.9", "72.5", "90.8", "49.9", "78.7", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/events.out.tfevents.1686052799.Machine2.482599.0>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt>`_"
  
  "MAE", "ViT-B/16", "256", "100", "46.0", "70.2", "81.3", "95.5", "11.2", "24.5", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/events.out.tfevents.1708887459.Machine2.1092409.0>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  
  "MoCoV2", "Res50", "256", "100", "61.5", "84.1", "74.3", "91.9", "41.8", "72.2", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/events.out.tfevents.1708248562.Machine2.439033.0>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  
  *"SimCLR*",* *"Res50"*, *"256"*, *"100"*, *"63`.2`, `"85`.2`, `"73`.9`, `"91`.9`, `"44`.8`, `"73`.9`, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_simclr _2023 -06 -22 _09 -11 -13 / pretrain / version _0 / events.out.tfevents .1687417883 .Machine2 .33270 .0 > ` _ ``, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_simclr _2023 -06 -22 _09 -11 -13 / pretrain / version _0 / checkpoints / epoch %99-step %500400 .ckpt > ` _ `
  
*SimCLR* + DCL*,* *"Res50"*, *"256"*, *"100"*, *"65`.1`, `"86`.2`, `"73`.5`, `"91`.7`, `"49`.6`, `"77`.5`, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_dcl _2023 -07 -04 _16 -51 -40 / pretrain / version _0 / events.out.tfevents .1688482310 .Machine2 .247807 .0 > ` _ ``, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_dcl _2023 -07 -04 _16 -51--40--pretrain--version--version--version--version--version--version--version--version--version--version--version--version--version--version--version--version--version--version--version
  
*SimCLR* + DCLW*,* *"Res50"*, *"256"*, *"100"*, *"64`.5`, `"86`.0`, `"73`.2`, `"91`.5`, `"48`.5`, `"76`.8`, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_dclw _2023 -07 -07 _14 -57 -13 / pretrain / version _0 / events.out.tfevents .1688734645 .Machine2 .3176 .0 > ` _ ``, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_dclw _2023 -07 -07 _14 -57--13--pretrain--version--version--version
  
*"SwAV*",* *"Res50"*, *"256"*, *"100"*, *"67`.2`, `"88`.1`, `"75`.4`, `"92`.7`, `"49`.5`, `"78`.6`, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_swav _2023 -05 -25 _08 -29 -14 / pretrain / version _0 / events.out.tfevents .1684996168 .Machine2 .1445108 .0 > ` _ ``, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_swav _2023 -05---25---08---29---14---pretrain---version
  
*"TiCo*",* *"Res50"*, *"256"*, *"100"*, *"49`.7`, `"74`.4`, `"72`.7`, `"90`.9`, `"26`.6`, `"53`.6`, `` `link` `<` https: // lightly - ssl - checkpoints.s3.amazonaws.com / imagenet_resnet50_tico _2024---01---07---18---40---57---pretrain---version
  
*"VICReg*",* *"Res50"* ,*"256"* ,*"100"* ,*"63 `.0`,`85 `.4`,`73 `.7`,`91 `.9`,`46 `.3`,`75 `.2`,` link `< https :// lightly --- ssl --- checkpoints .s3 .amazonaws .com --- imagenet --- resnet50 --- vicreg --- pretrain --- version --- events --- out --- tfevents --- Machine --- ckpt `_*

*\*We use square root learning rate scaling instead of linear scaling as it yields better results for smaller batch sizes.* See Appendix B.*1 in the SimCLR paper.*


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
