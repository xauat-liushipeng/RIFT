# RIFT: Rethinking Efficient Crack Segmentation with Task-Aligned Structural-Directional Modeling



## Environment

Recommended setup:

```bash
# CUDA 11.8

conda create -n rift python=3.10 -y
conda activate rift
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset Layout

The dataset loader expects the following directory structure:

```text
dataset_path/
├─ train_img/
├─ train_lab/
├─ val_img/
├─ val_lab/
├─ test_img/
└─ test_lab/
```

Each file in `*_img` is matched to its mask in `*_lab` by basename, using the `.png` label suffix.

## Model Variants

`RIFT-B` (base):

```text
dims=32,64,128,192
depths=2,2,3,2
decoder_dim=64
```

`RIFT-T` (tiny):

```text
dims=16,32,64,128
depths=2,2,2,1
decoder_dim=48
```

## Training

Train `RIFT-B`:

```bash
python train.py \
  --dataset_path ../datasets/CrackMap \
  --dims 32,64,128,192 \
  --depths 2,2,3,2 \
  --decoder_dim 64 \
  --kernel_size 13
```

Train `RIFT-T`:

```bash
python train.py \
  --dataset_path ../datasets/CrackMap \
  --dims 16,32,64,128 \
  --depths 2,2,2,1 \
  --decoder_dim 48 \
  --kernel_size 13
```

Outputs:

```text
./outputs/<timestamp>_Dataset-><dataset_name>/
├─ checkpoint.pth
├─ checkpoint_best.pth
└─ checkpoint<epoch>.pth
```

Logs are written to `./logs/`.

## Inference

Run inference with a trained checkpoint:

```bash
python test.py \
  --dataset_path ../datasets/CrackMap \
  --checkpoint ./outputs/<run>/checkpoint_best.pth \
  --save_dir ./results_test \
  --dims 32,64,128,192 \
  --depths 2,2,3,2 \
  --decoder_dim 64 \
  --kernel_size 13
```

The prediction directory will contain paired files:

```text
<image_name>_lab.png
<image_name>_pre.png
```

We provide the trained RIFT weights at [RIFT Weights](https://drive.google.com/file/d/1sY3VGl43GN4Dq4h9_8QBlKmE-FDAeiry/view?usp=drive_link), and the inference results for each model at [RIFT results](https://drive.google.com/file/d/1R5BqTf44ZilCwgJQ1SQee_ACMvjZnRp5/view?usp=drive_link).

## Evaluation

Evaluate a prediction directory from the command line:

```bash
python eval.py \
  --results_dir ./results_test \
  --thresh_step 0.01
```

Reported metrics include `mIoU`, `ODS`, `OIS`, `F1`, `Precision`, and `Recall`.

## Citation
If you are using our RIFT for your research, please cite the following paper:
```
@article{liu2026rethinking,
      title={Rethinking Efficient Crack Segmentation with Task-Aligned Structural-Directional Modeling}, 
      author={Shipeng Liu and Liang Zhao and Dengfeng Chen and Weihua Zhang},
      year={2026},
      eprint={2605.31048},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.31048}, 
}
```

## Acknowledgement
This project is based on [MixerCSeg](https://github.com/spiderforest/MixerCSeg), thanks for their excellent works.
