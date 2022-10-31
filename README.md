# GMP-VIS

This repo will release an official implementation for paper: A Graph Matching Perspective with Transformers on Video Instance Segmentation.

# Abstract
In this work, we study the challenge of video Instance Segmentation (VIS), which needs to track and segment multiple objects in videos automatically. We introduce a novel network from a graph matching perspective to formulate VIS, called GMP-VIS. Unlike traditional tracking-by-detection paradigm or bottom-up generative solutions, GMP-VIS uses a novel, learnable graph matching Transformer to predict the instances by heuristically learning the spatial-temporal relationships. Specifically, we take advantage of the powerful Transformer and exploit temporal feature aggregation to capture long-term temporal information across frames implicitly. After generating instance proposals for each frame, the difﬁcult instance association problem is cast as a more leisurely, differentiable graph matching task. The graph matching mechanism performs the data association between current and historical frames based on the proposed instance feature, which can better infer the deformations and obscured foreground instances. Building graph-level annotation during network training allows our GMP-VIS to mine more structural supervision signiﬁcantly distinguished from current VIS solutions. Our extensive experiments over three representative benchmarks, including YouTube-VIS19, YouTube-VIS21, and OVIS, demonstrate that GMP-VIS outperforms the current alternatives by a large margin.

<p align="center">
<img src="https://github.com/zyqin19/GMP-VIS/blob/main/arch-.png" width="1000">
</p>

### Installation
We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/zyqin19/GMP-VIS.git
```
Then, install PyTorch 1.6 and torchvision 0.7:
```
conda install pytorch==1.6.0 torchvision==0.7.0
```
Install pycocotools
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```
Compile DCN module(requires GCC>=5.3, cuda>=10.0)
```
cd models/dcn
python setup.py build_ext --inplace
```

### Preparation

Download and extract 2019 version of YoutubeVIS  train and val images with annotations from
[CodeLab](https://competitions.codalab.org/competitions/20128#participate-get_data) or [YoutubeVIS](https://youtube-vos.org/dataset/vis/).
We expect the directory structure to be the following:
```
VisTR
├── data
│   ├── train
│   ├── val
│   ├── annotations
│   │   ├── instances_train_sub.json
│   │   ├── instances_val_sub.json
├── models
...
```

### Training
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --backbone resnet101/50 --ytvos_path /path/to/ytvos --masks --pretrained_weights /path/to/pretrained_path
```

### Inference

```
python inference.py --masks --model_path /path/to/model_weights --save_path /path/to/results.json
```

### Acknowledgement
We would like to thank the [DETR](https://github.com/facebookresearch/detr), [IFC](https://github.com/sukjunhwang/IFC), [GMTracker](https://github.com/jiaweihe1996/GMTracker) open-source project for its awesome work, part of the code are modified from its project.
