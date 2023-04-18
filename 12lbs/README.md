# Introduction

This project focuses on evaluating the robustness of an image foundation model using the GRIT benchmark, specifically on a localization task (object detection).
The objective is to run inference on a chosen model, [adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP) which is a variation of OpenAI's CLIP model.
Then, I calculate its evaluation metric (mAP), and determine its zero-shot metric on the GRIT detection dataset.

## Environment & Dataset Setup

* Google Colab equipped with an A100 GPU or two V100 GPUs
* Google Drive 2TB plan(200GB usage so far)
* GRIT benchmark, including ImageNet, COCO, ADE20K, and Visual Genome (VG) dataset


## Variants for object detection(localization) tasks

* [adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP) model by Jiahao Li et al. ([paper](https://arxiv.org/pdf/2204.03647.pdf))
> which uses Flickr and Visual Genome (VG) images along with Zero-Shot Generalization (ZSG) annotations. This model focuses on adapting the original CLIP model without any further training.
* [OWL-ViT](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) by GOOGLE ([paper](https://arxiv.org/pdf/2101.01169.pdf))
> Vision Transformer for Open-World Localization(OWL-ViT) is actually a variation of the Vision Transformer(ViT) model that focuse computational efficiency of original ViT model. Thus, this is not a CLIP variant. 
* [RegionCLIP](https://github.com/microsoft/RegionCLIP) by Microsoft ([paper](https://arxiv.org/abs/2112.09106))
> RegionCLIP is an extension of the CLIP model that learns region-level visual representations, enabling fine-grained alignment between image regions and textual concepts. This supports tasks like zero-shot and open-vocabulary object detection. The model leverages pretraining, zero-shot inference, and transfer learning to achieve state-of-the-art results in its target tasks.

---
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

|![adCLIP_4by4](https://user-images.githubusercontent.com/84216960/232256051-528543a1-4035-4754-a209-c2273e0ba586.png)|
|:--:|
|adapting-CLIP model's performance visualized by [`adCLIP-visualizer.ipynb`](adCLIP-visualizer.ipynb)|

## GRIT Benchmark

Tasks: [**'localization'**, 'categorization', 'vqa', 'refexp', 'segmentation', 'keypoint', 'normal']

Subsets: [**'ablation'**, 'test']

```JSON
// Example of the 'GRIT/samples/ablation/localization.json' element
[
    {
        "example_id": "coco_loc_test-reserve_spoon_527067",
        "image_id": "coco/test2015/COCO_test2015_000000527067.jpg",
        "output_options": null,
        "task_bbox": null,
        "task_name": "localization",
        "task_query": "spoon"
    }
    // , ...
]
```
```JSON
// Example of the localization.json submission(computed output) file
[
    {
        "example_id": "coco_loc_test-reserve_spoon_527067",
        "confidence": 0.5,
        "bboxes": [
            [
                234.2857208251953, // x1
                73.00892639160156, // y1
                571.4285888671875, // x2
                478.8526916503906  // y2
            ]
        ]
    }
    // , ...
]
```

There are total 21078 images in the GRIT benchmark for the ablation localization task. The minimal datasets are required for me are as following:

* coco/test2015
* distorted/localization/coco/test2015
* nyuv2
* open_images/test

The downloading of the GRIT benchmark datasets is done in the [`GRIT-setup.ipynb`](GRIT-setup.ipynb) notebook manually for those minimal datasets. Although GRIT itself supports downloading the datasets automatically, it throws multiple errors especially the Google Drive does not work well with folder containing many files. Thus, it requires different approaches to manage the datasets in the Google Colab enviroment.

The model inference is done in the [`adCLIP-gritbench.ipynb`](adCLIP-gritbench.ipynb) notebook. Using A100 GPU, it takes about 4sec/image that is about 23 hours to complete. A100 is overspeced for this model which requires ~5GB GPU RAM, and Colab doesn't allow to use multiple A100 GPUs. Thus, it is better to use two V100 GPUs for this model. Using two V100 GPUs, it takes about 5sec/image that is about 15 hours to complete.

This adaptation model doesn't work in the `model.eval()` mode. Also as I mention about the memeory usage, that is clearly not optimized, thus things can be optimized much more by proper setting like `torch.no_grad()` mode or `torch.nn.DataParallel`, `torch.nn.parallel.DistributedDataParallel` options that I noticed for the parallelization, but I haven't figured it out yet.

For parallelization, what I have for now is just a very simple way to do that - I open the notebook in two sperate sessions and process the JSON file from both ends, one starting at the beginning and the other at the end. Since two V100 GPUs are restriction for my Colab subscription anyways, more than two GPUs are not possible at this point.

The inference results are saved in the [`ablation/localization.json`](ablation/localization.json) file.

The adapting-CLIP model's performance on the GRIT benchmark is illustrated in the table below. The model's mAP score is (???). The model's zero-shot score is (???).

---
# RegionCLIP

## Configuration

## Inference

## GRIT Benchmark
ABCDE

---
# Troubleshooting

1. The **cudatoolkit** is not supported on **Apple silicon chips** (current local machine uses an ARM M1 chip).
2. While there are some methods to use **conda or venv on Colab**, they are not very convenient. Many Colab Jupyter example notebooks rely on a fresh Python runtime (a session or kernel) for individual files and install dependencies every time the notebook starts to run.
3. The **Flickr** dataset requires UIUC registration, which may introduce potential delays.
4. The official website for the **Visual Genome (VG)** dataset is unavailable. Instead, the data images were found here: [VG1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [VG2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip).
(ref:https://github.com/YiwuZhong/SGG_from_NLS/blob/main/DATASET.md)
5. Unzipping tar files and uploading images to Google Drive is inefficient (too slow). It is better to upload a **tar.gz file** and unzip it within a Colab session.
6. The GRIT dataset has considerable storage requirements: **COCO (25GB), ImageNet (155GB?), ADE20K (?), and VG (15GB)**, which requires a Google Drive storage upgrade.
7. Downloading the GRIT dataset takes several hours. Also, unzipping and processing the dataset on Google Drive consumes a significant amount of time too.
8. Accessing a Google Drive folder with a larger number of files is terribly slow. Even download.sh for grit_official was broken to unpack the zip file successfully so many of the images are missing while they are supposed to be in the directory - The solution is copy the zip file from the mounted virtual drive folder in Colab and unpack the files in the actual directory on the machine with cloud processors running.
9. GRIT need to submit the zip file to be graded and it requires approval by a human which may makes waiting time.
10. fiftyone and MongoDB dependencies issue with M1 chip. The related Github issue: https://github.com/voxel51/fiftyone/issues/1165
11. adapting CLIP model is not optimized for inference. It is not working in the `model.eval()` mode. It throws an integer error but I couldn't solve it yet.
12. ~5GB GPU memory usage is too small for using A100 GPU. It is better to use V100 GPU for this model. Or, it might be a way to load multiple models working parallely in the same GPU memory, but I haven't figured it out yet.
13. Overall ~20 hours of inference time was not expected initially.

---
# References

The following repositories were used as references in this project:

1. OpenAI CLIP: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
2. GRIT official: [https://github.com/allenai/grit_official](https://github.com/allenai/grit_official)
3. adapting-CLIP: [https://github.com/pals-ttic/adapting-CLIP](https://github.com/pals-ttic/adapting-CLIP)
4. RegionCLIP: [https://github.com/microsoft/RegionCLIP](https://github.com/microsoft/RegionCLIP)