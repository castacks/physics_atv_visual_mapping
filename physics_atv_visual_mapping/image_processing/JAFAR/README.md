# JAFAR: Jack up Any Feature at Any Resolution
**NeurIPS 2025**

[![Website](https://img.shields.io/badge/JAFAR-%F0%9F%8C%90Website-purple?style=flat)](https://jafar-upsampler.github.io)
[![arXiv](https://img.shields.io/badge/-arXiv-%23B31B1B.svg?logo=arxiv&logoColor=white&labelColor=333)](https://jafar-upsampler.github.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PaulCouairon/JAFAR/blob/main/colab_demo.ipynb)


> [Paul Couairon*](https://pcouairon.github.io),
> [Loick Chambon*](https://loickch.github.io/),
> [Louis Serrano](https://scholar.google.com/citations?user=fKlo-lUAAAAJ&hl=fr),
> Jean-Emmanuel Haugeard,
> [Matthieu Cord](https://cord.isir.upmc.fr/),
> [Nicolas Thome](https://thome.isir.upmc.fr)
> 
> *Equal Contribution.


<p align="center">
  <img src="asset/teaser.png" alt="JAFAR Overview" width="80%">
</p>
<p align="center"><i>JAFAR improves metrics on many downstream tasks: semantic segmentation, depth estimation, feature activation, zero-shot open vocabulary, bird's eye view segmentation by upsampling features from any backbone.</i></p>

# Abstract

<em> Foundation Vision Encoders have become essential for a wide range of dense vision tasks. However, their low-resolution spatial feature outputs necessitate feature upsampling to produce the high-resolution modalities required for downstream tasks. In this work, we introduce JAFAR—a lightweight and flexible feature upsampler that enhances the spatial resolution of visual features from any Foundation Vision Encoder to an arbitrary target resolution. JAFAR employs an attention- based module designed to promote semantic alignment between high-resolution queries—derived from low-level image features—and semantically enriched low- resolution keys, using Spatial Feature Transform (SFT) modulation. Notably, despite the absence of high-resolution supervision, we demonstrate that learning at low upsampling ratios and resolutions generalizes remarkably well to significantly higher output scales. Extensive xperiments show that JAFAR effectively recovers fine-grained spatial details and consistently outperforms existing feature upsampling methods across a diverse set of downstream tasks. </em>

<p align="center">
  <img src="asset/architecture.png" alt="JAFAR Architecture" width="80%">
</p>
<p align="center"><i>JAFAR is an efficient attention-based feature upsampler that allows upsampling to any resolution.</i></p>

## Updates:
* 【24/09/2025】 **JAFAR** checkpoints for **Radio v2.5** (B/L/H) released.
* 【23/09/2025】 **JAFAR** checkpoints for **DINOv3** (S/S+/B/L) released.
* 【18/09/2025】 **JAFAR** is accepted at NeurIPS 2025!
* 【10/06/2025】 Code released.
* 【16/06/2025】 [JAFAR](https://arxiv.org/abs/2506.11136) is now on arxiv.

# 🚀 Main results

<p align="center">
  <img src="asset/pca.png" alt="PCA Visualization" width="100%">
</p>
<p align="center"><i>PCA visualization of features from various upsamplers.</i></p>

### 🔥 Semantic Segmentation

<p align="center">
  <img src="asset/segmentation.png" alt="Segmentation" width="100%">
</p>
<p align="center"><i>Linear probing results for semantic segmentation across various upsamplers.</i></p>

<details>
<summary><strong>📊 Linear Probing Results</strong></summary>

<br>

<div align="center">

| Method               | COCO mIoU (↑) | VOC mIoU (↑) | ADE20k mIoU (↑) | Cityscapes mIoU (↑) |
|----------------------|------------:|-----------:|------------:|-------------:|
| **Training-Free**    |             |            |             |              |
| Bilinear             | 59.03       | 80.70      | 39.23       | 59.37        |
| **Task-Agnostic**    |             |            |             |              |
| [FeatUp](https://github.com/mhamilton723/FeatUp)               | 60.10       | 81.08      | 38.82       | 56.06        |
| [LiFT](https://github.com/saksham-s/lift/tree/main)                 | 58.18       | 78.06      | 38.73       | 58.75        |
| **JAFAR (ours)** 🥇     | **60.78**   | **84.44**  | **40.49**   | **61.47**    |

</div>

</details>



### 🔥 Depth estimation

<p align="center">
  <img src="asset/depth.png" alt="Depth" width="100%">
</p>
<p align="center"><i>Linear probing results for depth estimation across various upsamplers.</i></p>

<details>
<summary><strong>📊 Linear Probing Results </strong></summary>

<br>

<div align="center">

| Method           | δ₁ (↑)  | RMSE (↓) |
|------------------|--------:|---------:|
| **Training-Free**|         |          |
| Bilinear         | 59.92   | 0.66     |
| **Task-Agnostic**|         |          |
| [FeatUp](https://github.com/mhamilton723/FeatUp)           | 61.69   | 0.64     |
| [LiFT](https://github.com/saksham-s/lift/tree/main)             | 57.04   | 0.70     |
| **JAFAR (ours)** 🥇 | **62.18** | **0.62** |

</div>

</details>

### 🔥 Class Activation Maps

<p align="center">
  <img src="asset/gradcam.png" alt="Gradcam" width="100%">
</p>
<p align="center"><i>Class Activation Map visualizations across various upsamplers.</i></p>

<details>
<summary><strong>📊 Evaluation </strong></summary>

<br>

<div align="center">

| Method   | A.D (↓) | A.I (↑) | A.G (↑) | ADCC (↑) |
|----------|-------|-------|-------|--------|
| **Training-free** |       |       |       |        |
| Bilinear | 19.0  | 18.5  | 3.4   | 61.7   |
| **Task-Agnostic** |       |       |       |        |
| [FeatUp](https://github.com/mhamilton723/FeatUp) | **15.3**  | 24.0  | 4.3   | 64.3   |
| [LiFT](https://github.com/saksham-s/lift/tree/main)   | 66.9  | 8.7   | 2.3   | 53.0   |
| **JAFAR (ours)** 🥇 | 17.4 | **30.9** | **6.5** | **73.3** |
</div>

</details>

### 🔥 Vehicle segmentation

<p align="center">
  <img src="asset/bev.gif" alt="BeV" width="100%">
</p>
<p align="center"><i>Vehicle segmentation in Bird's Eye View using DINOv2 + JAFAR.</i></p>

<details>
<summary><strong>📊 Evaluation</strong></summary>

<br>

<div align="center">

| Upsampling  mIoU (↑) | [SimpleBeV](https://github.com/aharley/simple_bev) | [PointBeV](https://github.com/valeoai/PointBeV) | [BeVFormer](https://github.com/fundamentalvision/BEVFormer) |
|----------------------|-----------|----------|-----------|
| **Training-free** |       |       |       |        |
| Low-Res              | 31.75     | 34.89    | 33.72     |
| Bilinear             | 33.67     | 36.01    | 34.18     |
| **Task-Agnostic** |       |       |       |        |
| [FeatUp](https://github.com/mhamilton723/FeatUp)               | 33.95     | 35.38    | 34.01     |
| **JAFAR (ours)** 🥇                | **36.59**     | **37.20**    | **36.54**     |

</div>
</details>

## 🔨 Setup <a name="setup"></a>

➡️ Install.

Launch the following commands to install the dependencies and create a mamba (/ conda) environment.

<details>
  <summary> Details</summary>

``` bash
git clone https://github.com/...
cd JAFAR

micromamba create -n jafar python==3.10.14  -y -c conda-forge
micromamba activate jafar
micromamba install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge -y

pip install uv
uv pip install einops==0.8.0 matplotlib==3.7.0 numpy==1.24.4 timm==1.0.11 plotly tensorboard hydra-core ipykernel rich pytest scikit-learn torchmetrics==1.6.2 transformers
```
</details>


➡️ Datasets.

See [Preparing Datasets for JAFAR](docs/datasets.md) for details on how to download the datasets.

## 🔄 Training <a name="training"></a>

To train JAFAR with the dinov2 backbone, execute the following command:

```python
python train.py backbone.name=vit_small_patch14_dinov2.lvd142m hydra.run.dir=output/jafar/dinov2
```

You can change the backbone to any other available backbone in the timm library by just changing the `backbone.name` argument.

To fast prototyping we add a sanity argument, it will execute the code for only a few steps and helps you to see if everything is working properly. You can use it as follows:

```python
python train.py sanity=True
```


## 🔄 Evaluation <a name="evaluating"></a>
To evaluate the model on segmentation on the VOC dataset with the dinov2 backbone, execute:
```python
python evaluation/train_probes.py eval.task=seg dataset_evaluation=voc \
  backbone.name=vit_small_patch14_dinov2.lvd142m \
  eval.model_ckpt=model.pth \
  hydra.run.dir=evaluation/unsupervised/voc/vit_small_patch14_dinov2.lvd142m
```
You can change the dataset and the backbone to any other available dataset and backbone in the timm library by just changing the `dataset_evaluation` and `backbone.name` arguments. It will save logs, tensorboard and checkpoits in the hydra directory.

We add a file to benchmark the evaluation time and memory usage of the model. You can run it as follows:
```python
pytest test/test_time_and_memory.py -s -v
```

## 🔄 Notebooks 

We provide notebooks to perform training, inference and visualisation.

## 🏆 Available Models

We provide pre-trained JAFAR models for various backbones. You can find the model weights from the following links:

<div align="center">

| Backbone Name     | Download Link                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| ViT-B-16            | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_224.pth)                      |
| ViT-B-16-DINO             | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_224.dino.pth)                           |
| ViT-S-14-DINOv2          | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_dinov2.lvd142m.pth)               |
| ViT-B-14-DINOv2          | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch14_dinov2.lvd142m.pth)               |
| ViT-S-Reg4-14-DINOv2      | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_reg4_dinov2.pth)                  |
| ViT-S-16-DINOv3      | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch16_dinov3.lvd1689m.pth)                  |
| ViT-S+-16-DINOv3      | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_plus_patch16_dinov3.lvd1689m.pth)                  |
| ViT-B-16-DINOv3      | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_dinov3.lvd1689m.pth)                  |
| ViT-L-16-DINOv3      | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_large_patch16_dinov3.lvd1689m.pth)                  |
| ViT-B-16-CLIP            | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_clip_384.pth)                      |
| ViT-B-16-SigLIP2          | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_base_patch16_siglip_512.v2_webli.pth)           |
| Radio_v2.5-B          | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/radio_v2.5-b.pth)           |
| Radio_v2.5-L         | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/radio_v2.5-l.pth)           |
| Radio_v2.5-H          | [Download](https://github.com/PaulCouairon/JAFAR/releases/download/Weights/radio_v2.5-h.pth)           |

</div>

Do not hesitate to open an issue if you need a specific backbone or if you have any questions to train it by yourself.


## 👍 Acknowledgements

Many thanks to these excellent open source projects:
* https://github.com/mhamilton723/FeatUp
* https://github.com/saksham-s/lift/tree/main
* https://github.com/Jiawei-Yang/Denoising-ViT
* https://github.com/chongzhou96/MaskCLIP
* https://github.com/valeoai/PointBeV

To structure our code we used:
* https://github.com/facebookresearch/hydra


## ✏️ Bibtex

If this work is helpful for your research, please consider citing the following BibTeX entry and putting a star on this repository.

```
@misc{couairon2025jafar,
      title={JAFAR: Jack up Any Feature at Any Resolution}, 
      author={Paul Couairon and Loick Chambon and Louis Serrano and Jean-Emmanuel Haugeard and Matthieu Cord and Nicolas Thome},
      year={2025},
      eprint={2506.11136},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.11136}, 
}
```

---
