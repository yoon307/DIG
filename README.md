# DiG
### Official repository for ECCV 2024 paper: [Diffusion-Guided Weakly Supervised Semantic Segmentation](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06482.pdf) by Sung-Hoon Yoon, Hoyong Kwon*, Jaeseok Jeong*, Daehee Park, and Kuk-Jin Yoon. 
---

# 1.Prerequisite
## 1.1 Environment
* Tested on Ubuntu 20.04, with Python 3.9, PyTorch 1.8.2, CUDA 11.7, multi gpus(two for pascal, four for coco) - Nvidia RTX 3090.

* You can create conda environment with the provided yaml file.
```
conda env create -f environment.yaml
```

## 1.2 Dataset Preparation
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
- Download MS COCO images from the official COCO website [here](https://cocodataset.org/#download).
- Download semantic segmentation annotations for the MS COCO dataset [here](https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view?usp=sharing). (Refer [RIB](https://github.com/jbeomlee93/RIB))

- Directory hierarchy 
```
    ./data
    ├── VOC2012       
    └── COCO2014            
            ├── SegmentationClass     # GT dir             
            ├── train2014  # train images downloaded from the official COCO website 
            └── val2014    # val images downloaded from the official COCO website
```


* ImageNet-pretrained weights for ViT are from [deit_small_imagenet.pth](https://drive.google.com/drive/folders/1bjcjMXS8_Q0SvRkChp9gGeygpJm7wGr0?usp=sharing).  
**You need to place the weights as "./pretrained/deit_small_imagenet.pth. "**

# 2. Usage

## 2.1 Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]

For PASCAL:
```
python train_trm.py --name [exp_name] --exp dig_eccv24
```
For COCO:
```
python train_trm_coco.py --name [exp_name] --exp dig_coco_eccv24
```

**Note that the mIoU in COCO training set is evaluated on the subset (5.2k images, not the full set of 80k images) for fast evaluation**

## 2.2 Inference (CAM)
* Pretrained weight (PASCAL, seed: 69.3% mIoU) can be downloaded [here](https://drive.google.com/drive/folders/1bjcjMXS8_Q0SvRkChp9gGeygpJm7wGr0?usp=sharing) (69.3_pascal.pth).

For pretrained model (69.3%):
```
python infer_trm.py --name [exp_name] --load_pretrained [DIR_of_69.3%_ckpt] --load_epo 100 --dict
```

For model you trained:

```
python infer_trm.py --name [exp_name] --load_epo [EPOCH] --dict
```

## 2.3 Evaluation (CAM)
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```


# 3. Additional Information
## 3.1 Paper citation
If our code be useful for you, please consider citing our ECCV 2024 paper using the following BibTeX entry.
```
@inproceedings{yoon2024diffusion,
  title={Diffusion-Guided Weakly Supervised Semantic Segmentation},
  author={Yoon, Sung-Hoon and Kwon, Hoyong and Jeong, Jaeseok and Park, Daehee and Yoon, Kuk-Jin},
  booktitle={European Conference on Computer Vision, ECCV 2024},
  year={2024},
  organization={European Conference On Computer Vision}
}
```
You can also check my earlier works published on ICCV 2021 ([OC-CSE](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf)) , ECCV 2022 ([AEFT](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890323.pdf)), CVPR 2023 ([ACR](https://openaccess.thecvf.com/content/CVPR2023/papers/Kweon_Weakly_Supervised_Semantic_Segmentation_via_Adversarial_Learning_of_Classifier_and_CVPR_2023_paper.pdf)), CVPR 2024 ([CTI](https://openaccess.thecvf.com/content/CVPR2024/papers/Yoon_Class_Tokens_Infusion_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2024_paper.pdf))



## 3.2 References
We heavily borrow the work from [MCTformer](https://github.com/xulianuwa/MCTformer)  and [DDPM-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repository. Thanks for the excellent codes!
```
[1] Xu, Lian, et al. "Multi-class token transformer for weakly supervised semantic segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[2] Lee, Jungbeom, et al. "Reducing information bottleneck for weakly supervised semantic segmentation." Advances in neural information processing systems 34 (2021): 27408-27421.
