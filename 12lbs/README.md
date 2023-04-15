# Introduction

This project focuses on evaluating the robustness of an image foundation model using the GRIT benchmark, specifically on a localization task (object detection).
The objective is to run inference on a chosen model, [adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP) which is a variation of OpenAI's CLIP model.
Then, I calculate its evaluation metric (mAP), and determine its zero-shot metric on the GRIT detection dataset.

# Environment & Dataset Setup

* Google Colab equipped with an A100 GPU
* Google Drive 2TB plan
* GRIT benchmark, including ImageNet, COCO, ADE20K, and Visual Genome (VG) dataset
* adapting-CLIP model, which uses Flickr and Visual Genome (VG) images along with Zero-Shot Generalization (ZSG) annotations.

# Model Inference

[Figures]

# GRIT Benchmark(mAP and zero-shot metric)


[Figures]

# Troubleshooting

1. The **cudatoolkit** is not supported on **Apple silicon chips** (current local machine uses an ARM M1 chip).
2. While there are some methods to use **conda or venv on Colab**, they are not very convenient. Many Colab Jupyter example notebooks rely on a fresh Python runtime (a session or kernel) for individual files and install dependencies every time the notebook starts to run.
3. The **Flickr** dataset requires UIUC registration, which may introduce potential delays.
4. The official website for the **Visual Genome (VG)** dataset is unavailable. Instead, the data images were found here: [VG1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [VG2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip).
(ref:https://github.com/YiwuZhong/SGG_from_NLS/blob/main/DATASET.md)
5. Unzipping tar files and uploading images to Google Drive is inefficient (too slow). It is better to upload a **tar.gz file** and unzip it within a Colab session.
6. The GRIT dataset has considerable storage requirements: **COCO (25GB), ImageNet (155GB?), ADE20K (?), and VG (15GB)**, which requires a Google Drive storage upgrade.
7. Downloading the GRIT dataset takes several hours. Also, unzipping and processing the dataset on Google Drive consumes a significant amount of time too.

# Summary


# References

