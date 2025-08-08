<div align="center">
<h1>Official Pytorch Implementation of KProBA: 

Kernel-Prototype Guided Background Adaptation for Class-Incremental Semantic Segmentation </h1>

<img src = "pipeline.jpg" width="100%" height="100%">
</div>

## Preparation

### Requirements

- CUDA>=11.8
- torch>=2.0.0
- torchvision>=0.15.0
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib
- visdom
- tensorboardX

```bash
pip install -r requirements.txt
```


### Datasets

We use the Pascal VOC 2012 and ADE20K datasets for evaluation following the previous methods. You can download the datasets from the following links:

Download Pascal VOC 2012 dataset:
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
Download Additional Segmentation Class Annotations:
```bash
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
```

Download ADE20K dataset:
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```


```
datasets/
   ├── VOC2012/
   │   ├── Annotations/
   │   ├── ImageSet/
   │   ├── JPEGImages/
   │   ├── SegmentationClassAug/
   │   └── saliency_map/
   └── ADEChallengeData2016
       ├── annotations
       │   ├── training
       │   └── validation
       └── images
           ├── training
           └── validation
```

## Getting Started

### Class-Incremental Segmentation Segmentation on ADE20K

Run our scripts `run_init.sh` and `run.sh` for class-incremental segmentation on ADE20K dataset, or follow the instructions below.

Initial step: 
```bash
MODEL=deeplabv3bga_resnet101
DATA_ROOT= # Your dataset root path
DATASET=ade
TASK=100-5
EPOCH=60
BATCH=8
LOSS=bce_loss
LR=0.001
THRESH=0.7
SUBPATH=KProBA
CURR=0

CUDA_VISIBLE_DEVICES=0 \
python train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp \
    --curr_step ${CURR} --subpath ${SUBPATH} --initial --overlap 
```

Incremental steps:
```bash
MODEL=deeplabv3bga_resnet101
DATA_ROOT= # Your dataset root path
DATASET=ade
TASK=100-10
EPOCH=100
BATCH=4
LOSS=bce_loss
LR=0.01
THRESH=0.7
SUBPATH=KProBA
CURR=1

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=19198 \
train.py --data_root ${DATA_ROOT} --model ${MODEL} --crop_val --lr ${LR} \
    --batch_size ${BATCH} --train_epoch ${EPOCH}  --loss_type ${LOSS} \
    --dataset ${DATASET} --task ${TASK} --lr_policy poly \
    --pseudo --pseudo_thresh ${THRESH}  --bn_freeze  --amp \
    --curr_step ${CURR} --subpath ${SUBPATH} --overlap
```

## Acknowledgement

Our implementation is based on these repositories: [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch), [SSUL](https://github.com/clovaai/SSUL), 
[BARM](https://github.com/ANDYZAQ/BARM), [KLDA](https://github.com/salehmomeni/klda).
Thanks for their great work!


