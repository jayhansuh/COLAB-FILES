## Introduction

This project focuses on evaluating the robustness of an image foundation model using the GRIT benchmark, specifically on a localization task (object detection).
The objective is to run inference on a chosen model, [adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP) which is a variation of OpenAI's CLIP model.
Then, I calculate its evaluation metric (mAP), and determine its zero-shot metric on the GRIT detection dataset.

## Environment & Dataset Setup

* Google Colab equipped with an A100 GPU
* Google Drive 2TB plan
* GRIT benchmark, including ImageNet, COCO, ADE20K, and Visual Genome (VG) dataset


## Variants for object detection(localization) tasks

* [adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP) model by Jiahao Li et al. ([paper](https://arxiv.org/pdf/2204.03647.pdf))
> which uses Flickr and Visual Genome (VG) images along with Zero-Shot Generalization (ZSG) annotations. This model focuses on adapting the original CLIP model without any further training.
* [OWL-ViT](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) by GOOGLE ([paper](https://arxiv.org/pdf/2101.01169.pdf))
> Vision Transformer for Open-World Localization(OWL-ViT) is actually a variation of the Vision Transformer(ViT) model that focuse computational efficiency of original ViT model. Thus, this is not a CLIP variant. 
* [RegionCLIP](https://github.com/microsoft/RegionCLIP) by Microsoft ([paper](https://arxiv.org/abs/2112.09106))
> RegionCLIP is an extension of the CLIP model that learns region-level visual representations, enabling fine-grained alignment between image regions and textual concepts. This supports tasks like zero-shot and open-vocabulary object detection. The model leverages pretraining, zero-shot inference, and transfer learning to achieve state-of-the-art results in its target tasks.

# Adapting-CLIP

## Configuration
The below configuration is the setup used for the adapting-CLIP model.
```python
model = SLICViT
args = {
    'model': 'vit14',
    'alpha': 0.75,
    'aggregation': 'mean',
    'n_segments': list(range(100, 601, 50)),
    'temperature': 0.02,
    'upsample': 2,
    'start_block': 0,
    'compactness': 50,
    'sigma': 0,
}
dataset_full = FlickrDataset(data_type='flickr30k_c1/val')
iou_thr = 0.5
model = model(**args).cuda()
```

## Inference
The adapting-CLIP model's performance on the localization task is illustrated in the example images below. Bounding boxes are color-coded as follows:
* Red: predicted bounding box
* Blue: ground truth bounding box

![adCLIP_4by4](https://user-images.githubusercontent.com/84216960/232256051-528543a1-4035-4754-a209-c2273e0ba586.png)
## GRIT Benchmark
ABCDE

# RegionCLIP

## Configuration

## Inference

## GRIT Benchmark
ABCDE
## Troubleshooting

1. The **cudatoolkit** is not supported on **Apple silicon chips** (current local machine uses an ARM M1 chip).
2. While there are some methods to use **conda or venv on Colab**, they are not very convenient. Many Colab Jupyter example notebooks rely on a fresh Python runtime (a session or kernel) for individual files and install dependencies every time the notebook starts to run.
3. The **Flickr** dataset requires UIUC registration, which may introduce potential delays.
4. The official website for the **Visual Genome (VG)** dataset is unavailable. Instead, the data images were found here: [VG1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [VG2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip).
(ref:https://github.com/YiwuZhong/SGG_from_NLS/blob/main/DATASET.md)
5. Unzipping tar files and uploading images to Google Drive is inefficient (too slow). It is better to upload a **tar.gz file** and unzip it within a Colab session.
6. The GRIT dataset has considerable storage requirements: **COCO (25GB), ImageNet (155GB?), ADE20K (?), and VG (15GB)**, which requires a Google Drive storage upgrade.
7. Downloading the GRIT dataset takes several hours. Also, unzipping and processing the dataset on Google Drive consumes a significant amount of time too.

## Summary


## References

The following repositories were used as references in this project:

1. OpenAI CLIP: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
2. GRIT official: [https://github.com/allenai/grit_official](https://github.com/allenai/grit_official)
3. adapting-CLIP: [https://github.com/pals-ttic/adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP)
