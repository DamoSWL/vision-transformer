# Dual-windowed Vision Transformer with Angular Self-Attention

This is the source code for the paper "Dual-windowed Vision Transformer with Angular Self-Attention".

### 1. Prerequisite

CUDA == 11.4.0

timm == 0.4.5

Pytorch >= 1.10

Numpy >=1.10

MMCV == 1.5.0

MMDetection == 2.25.3

MMSegmentation == 0.23.0



#### Install MMCV
The MMCV full version can be installed with the command:

```bash
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
```

#### Install MMDetection
```bash
git clone -b 2.25.3 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

#### Install MMSegmentation
```bash
git clone -b 0.23.0 https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

### 2. Image Classification
Pepare the ImageNet-1K dataset.

In `vision-transformer/`, for tiny-sized model with quadratic self-attention, run the following code to train the model.
```bash
torchrun --nproc_per_node=1 --master_port 29441 main.py --data $root \
        --model angular_tiny_quad_224 \
        -b 420 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results 
```

`--model` is the model name, `-b` is the batch size, `--data` is the path of the dataset, `--output` is the folder for output, `--nproc_per_node` is the number of GPUs used.

For tiny-sized model with cosine self-attention.
```bash
torchrun --nproc_per_node=1 --master_port 21418 main.py --data $root \
        --model angular_tiny_cos_224 \
        -b 460 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results 
```

For small-sized model with quadratic self-attention.
```bash
torchrun --nproc_per_node=1 --master_port 25187 main.py --data $root \
        --model angular_small_quad_224 \
        -b 200 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results
```

For small-sized model with cosine self-attention.
```bash
torchrun --nproc_per_node=1 --master_port 25187 main.py --data $root \
        --model angular_small_cos_224 \
        -b 200 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results
```

For base-sized model with quadratic self-attention.
```bash
torchrun --nproc_per_node=1 --master_port 21521 main.py --data $root \
        --model angular_base_quad_224 \
        -b 100 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results
```

For base-sized model with cosine self-attention.
```bash
torchrun --nproc_per_node=1 --master_port 25711 main.py --data $root \
        --model angular_base_cos_224 \
        -b 100 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results
```

### 3. Object Detection
Prepare the COCO2017 dataset and modify the path of the dataset in the file `mmdetection/configs/_base_/datasets/coco_detection.py`.

```python
data_root = 'data/coco/'
```

Copy the file `vision-transformer/object detecton/tool/train.sh` to `mmdetection/tools`.

Create the folder `mmdetection/configs/angularvit` and copy the files under the folder `vision-transformer/object detecton/configs` to folder `mmdetection/configs/angularvit`.

Copy the file `vision-transformer/object detecton/backbone/angularvit.py` to the folder `mmdetection/mmdet/models/backbones`

If you have pre-trained model, add the path in the configuration files. For instance, in `mmdetection/configs/angularvit/mask_rcnn_angularvit_quad_t.py`, modify the value of `pretrained=None`.

The model is trained with 3x training schedule, if you want to switch to 1x training schedule, comment the following codes in configuration files.
```pyton
lr_config = dict(warmup_iters=500,step=[27, 33])
runner = dict(max_epochs=36)
```

In `mmdetection/`, train the tiny-sized model with quadratic self-attention,
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_quad_t.py 1  --work-dir ./quad_t  #Mask-RCNN

bash tools/train.sh configs/angularvit/cascade_mask_rcnn_angularvit_quad_t.py 1 --work-dir ./quad_t_cas  # Cascade Mask-RCNN
```

The number `1` denotes the number of GPUs used.

Train the tiny-sized model with cosine self-attention,
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_cos_t.py 1  --work-dir ./cos_t  #Mask-RCNN

bash tools/train.sh configs/angularvit/cascade_mask_rcnn_angularvit_cos_t.py 1 --work-dir ./cos_t_cas  # Cascade Mask-RCNN
```

Train the small-sized model with quadratic self-attention,
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_quad_s.py 1  --work-dir ./quad_s  #Mask-RCNN

bash tools/train.sh configs/angularvit/cascade_mask_rcnn_angularvit_quad_s.py 1 --work-dir ./quad_s_cas  # Cascade Mask-RCNN
```

Train the small-sized model with cosine self-attention,
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_cos_s.py 1  --work-dir ./cos_s  #Mask-RCNN

bash tools/train.sh configs/angularvit/cascade_mask_rcnn_angularvit_cos_s.py 1 --work-dir ./cos_s_cas  # Cascade Mask-RCNN
```

Train the base-sized model with quadratic self-attention,
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_quad_b.py 1  --work-dir ./quad_b  #Mask-RCNN

bash tools/train.sh configs/angularvit/cascade_mask_rcnn_angularvit_quad_b.py 1 --work-dir ./quad_b_cas  # Cascade Mask-RCNN
```

Train the base-sized model with cosine self-attention,
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_cos_b.py 1  --work-dir ./cos_b  #Mask-RCNN

bash tools/train.sh configs/angularvit/cascade_mask_rcnn_angularvit_cos_b.py 1 --work-dir ./cos_b_cas  # Cascade Mask-RCNN
```


### 3. Semantic Segmentation
Prepare the ADE20K dataset and and modify the path of the dataset in the file `mmsegmentation/configs/_base_/datasets/ade20k.py`.

```python
data_root = 'data/ade20k'
```


Copy the file `vision-transformer/semantic segmentation/tool/train.sh` to `mmsegmentation/tools`.

Create the folder `mmsegmentation/configs/angularvit` and copy the files under the folder `semantic segmentation/configs` to folder `mmsegmentation/configs/angularvit`.

Copy the file `vision-transformer/semantic segmentation/backbone/angularvit.py` to the folder `mmsegmentation/mmdet/models/backbones`

If you have pre-trained model, add the path in the configuration files. For instance, in `mmsegmentation/configs/angularvit/angularvit_quad_t.py`, modify the value of `checkpoint_file=None`.


In `mmsegmentation/`, train the tiny-sized model with quadratic self-attention,

```bash
bash tools/train.sh configs/angularvit/angularvit_quad_tiny.py 1 --work-dir ./quad_t
```

Train the tiny-sized model with cosine self-attention.
```bash
bash tools/train.sh configs/angularvit/angularvit_cos_tiny.py 1 --work-dir ./cos_t
```

Train the small-sized model with quadratic self-attention.
```bash
bash tools/train.sh configs/angularvit/angularvit_quad_small.py 1 --work-dir ./quad_s
```

Train the small-sized model with cosine self-attention.
```bash
bash tools/train.sh configs/angularvit/angularvit_cos_small.py 1 --work-dir ./cos_s
```

Train the base-sized model with quadratic self-attention.
```bash
bash tools/train.sh configs/angularvit/angularvit_quad_base.py 1 --work-dir ./quad_b
```

Train the base-sized model with cosine self-attention.
```bash
bash tools/train.sh configs/angularvit/angularvit_cos_base.py 1 --work-dir ./cos_b
```


### 4. Ablation Study

In `vision-transformer/`, train our DWAViT-T model with scaled dot-product self-attention for image classification.
```bash
torchrun --nproc_per_node=1 --master_port 21453 main.py --data $root \
        --model angular_tiny_exp_224 \
        -b 480 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results
```

In `mmdetection/`, train our DWAViT-T model with scaled dot-product self-attention for object detection.
```bash
bash tools/train.sh configs/angularvit/mask_rcnn_angularvit_exp_t.py 1  --work-dir ./exp_t
```

In `mmsegmentation/`, train our DWAViT-T model with scaled dot-product self-attention for semantic segmentation.
```bash
bash tools/train.sh configs/angularvit/angularvit_exp_tiny.py 1 --work-dir ./exp_t
```

For CSWin and Deit with our angular self-attention for image classification task, just replace the code in `models.py` in their original codebase and run the revelant commands.
```python
q = q * self.scale
attn = (q @ k.transpose(-2, -1))
```
with
```python
q = F.normalize(q,dim=-1)
k = F.normalize(k,dim=-1)

attn = (q @ k.transpose(-2, -1)) / 0.1  #cosine
attn = (1-4*torch.square(theta/math.pi)) / 0.1  #quadratic
```

