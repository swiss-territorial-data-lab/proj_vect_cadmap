# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

This README is modified based on the implementation of the project `Vectorization of Historical Cadastral Maps`. For the original instruction on configuring environment, please click this [link](https://github.com/OpenGVLab/InternImage/tree/master/segmentation).

## Usage

### Install

- Activate conda virtual environment:

```bash
cd internimage
conda activate cadmap
```

- Install `CUDA==11.6` with following [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.13.1` and `torchvision==0.14.1`:

For examples, to install with pip:
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

- Install `timm==0.6.11` and `mmcv-full==1.6.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.6.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
- You can also install the operator using .whl files
[DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

### Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

**geneva_line** dataset: binary semantic segmentation for borderline extraction  
**geneva_semantic** dataset: multi-class semantic segmentation for polygon attributes classification 

### Training
The detailed training parameters are set in the `config/dataset_name/model_name.py` file. You can change the loaded pretrained weights, model architecture, optimizor and check points configuration in the python file. 

**Attention:** default setting for batch size is `data=dict(samples_per_gpu=2)`. If the GPU memory is limited, we suggest to set the batch size to 1 image per GPU.

To train an `InternImage` on geneva_line dataset, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-B` with a single gpu:

```bash
export CKPT_DIR=$(pwd)/ckpt-weights
python train.py configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py --work-dir ${CKPT_DIR} --gpu-id 0 
```

If with 8 GPU on 1 node, run:

```bash
export CKPT_DIR=$(pwd)/ckpt-weights
sh dist_train.sh configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py 8 --work-dir ${CKPT_DIR}
```

### Evaluation

To evaluate our `InternImage` on geneva_line, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single GPU:

```bash
# specify the weight path to selected model, change this accordingly before running each line
export WEIGHT_PATH=PATH_TO_PTH_FILE
python test.py configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py ${WEIGHT_PATH} --eval mIoU
```

With 8 GPUs:

```bash
# specify the weight path to selected model, change this accordingly before running each line
export WEIGHT_PATH=PATH_TO_PTH_FILE
sh dist_test.sh configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py ${WEIGHT_PATH} 8 --eval mIoU
```

### Inference on cadastral maps

To inference a series of cadastral map from a community for binary segmentation:

```bash
export IMG_PATH="/path/to/images"
export OUTPUT_PATH="../data/line_prediction_mask"
export CFG_PTH="configs/geneva_line/config_file.py"
export WEIGHT_PATH="/path/to/model/ckpt.pth"

# to observe the input image and the output mask at the same time, users can set the mask opacity to 0.5
python inference.py ${IMG_PATH} ${OUTPUT_PATH} ${CFG_PTH} ${WEIGHT_PATH} --opacity 1 --palette line 
```

For multi-class semantic segmentation:
```bash
export IMG_PATH="/path/to/images"
export OUTPUT_PATH="../data/semantic_prediction_mask"
export CFG_PTH="configs/geneva_semantic/config_file.py"
export WEIGHT_PATH="/path/to/model/ckpt.pth"

# to observe the input image and the output mask at the same time, users can set the mask opacity to 0.5
python inference.py ${IMG_PATH} ${OUTPUT_PATH} ${CFG_PTH} ${WEIGHT_PATH} --opacity 1 --palette semantic  
```