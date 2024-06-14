# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage algorithm for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

This README is modified based on the implementation of the project *Vectorization of Historical Cadastral Plans*. For the original instruction on configuring environment, please click this [link](https://github.com/OpenGVLab/InternImage/tree/master/segmentation).


## Usage

### Install

- Activate conda virtual environment:

```bash
cd internimage
conda activate cadmap
```

- Install `CUDA==11.6` with following [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.13.1` and `torchvision==0.14.1`:

For example, to install with pip:
```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

- Install `mmcv-full` and `timm`:

```bash
pip install mmcv-full==1.6.0 --no-cache
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

Prepare datasets according to the [guidelines](../README.md/#generate-datasets).

**geneva_line** dataset: binary semantic segmentation for borderline extraction  
**geneva_semantic** dataset: multi-class semantic segmentation for polygon attributes classification 

### Training
The detailed training parameters are set in the `configs/<DATASET_NAME>/<MODEL_NAME>.py` file. You can change the loaded pretrained weights, model architecture, optimizor and check points configuration in the python file. 

**Attention:** 
1. The default and minimum setting for batch size is `data=dict(samples_per_gpu=2)`. If the GPU memory is limited, we suggest choosing a model with less weights.
2. The working directory for the **Training and Testing** section is `~/internimage/`.
3. Before start training, the source code of `mmseg` need to be modified because of torch had changed output data format in version updates. Please go to the file `/path_to_conda/envs/cadmap/lib/python3.10/site-packages/mmseg/models/segmentors/encoder_decoder.py` and modify line 168 to `preds = img.new_zeros((batch_size, num_classes, h_img, w_img)).to(dtype=torch.float64)`.

To train a model on the `geneva_line` dataset:

With a single GPU, to train `InternImage-B`, run:

```bash
cd internimage
export CKPT_DIR=$(pwd)/ckpt-weights/geneva_line
python train.py configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py --work-dir ${CKPT_DIR}
```

With multiple GPU (_e.g._ 8 GPUs) on 1 node, run:

```bash
cd internimage
export CKPT_DIR=$(pwd)/ckpt-weights/geneva_line
export GPU_NUM=8 
sh dist_train.sh configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py ${GPU_NUM} --work-dir ${CKPT_DIR}
```

To train a model on the `geneva_semantic` dataset:

With a single GPU, to train `InternImage-B`, run:

```bash
export CKPT_DIR=$(pwd)/ckpt-weights/geneva_semantic
python train.py configs/geneva_semantic/upernet_internimage_b_512x1024_160k_geneva_semantic.py --work-dir ${CKPT_DIR}
```

With multiple GPU (_e.g._ 8 GPUs) on 1 node, run:

```bash
export CKPT_DIR=$(pwd)/ckpt-weights/geneva_semantic
export GPU_NUM=8 
sh dist_train.sh configs/geneva_semantic/upernet_internimage_b_512x1024_160k_geneva_semantic.py ${GPU_NUM} --work-dir ${CKPT_DIR}
```

### Evaluation

To evaluate a model on geneva_line dataset:

With a single GPU, to evaluate `InternImage-B`, run:

```bash
# specify the weight path to selected model, change this accordingly before running each line
export WEIGHT_PATH=<PATH_TO_PTH_FILE>
python test.py configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py ${WEIGHT_PATH} --eval mIoU
```

With multiple GPU (_e.g._ 8 GPUs) on 1 node, run:

```bash
# specify the weight path to selected model, change this accordingly before running each line
export WEIGHT_PATH=<PATH_TO_PTH_FILE>
sh dist_test.sh configs/geneva_line/upernet_internimage_b_512x1024_160k_geneva_line.py ${WEIGHT_PATH} 8 --eval mIoU
```

The semantic segmentation model can be evaluated following the same method.


### Inference on cadastral maps

Before running the script, the source code of `mmseg` need to be modified in `${CONDA_PATH}/envs/cadmap/lib/python3.10/site-packages/mmseg/datasets/pipelines/formatting.py`. Replace line 280~281 with following code:

```
        for key in self.meta_keys:
            if key in results.keys():
                img_meta[key] = results[key]
            img_meta['pad_shape'] = results['img_shape']
```

To infere the binary segmentation of a series of cadastral plans for a community, run :

```bash
export IMG_FOLDER_PATH="</path/to/image_folder>"
export OUTPUT_PATH="../data/line_prediction_mask"
export CFG_PTH="configs/geneva_line/<config_file.py>"
export WEIGHT_PATH="</path/to/model/ckpt.pth>"

# to observe the input image and the output mask at the same time, users can set the mask opacity to 0.5
python inference.py --input_path ${IMG_FOLDER_PATH} --save_path ${OUTPUT_PATH} --cfg ${CFG_PTH} --ckpt ${WEIGHT_PATH} --opacity 1 --palette line
```

For multi-class semantic segmentation:
```bash
export IMG_FOLDER_PATH="</path/to/image_folder>"
export OUTPUT_PATH="../data/semantic_prediction_mask"
export CFG_PTH="configs/geneva_semantic/config_file.py"
export WEIGHT_PATH="</path/to/model/ckpt.pth>"

# to observe the input image and the output mask at the same time, users can set the mask opacity to 0.5
python inference.py --input_path ${IMG_FOLDER_PATH} --save_path ${OUTPUT_PATH} --cfg ${CFG_PTH} --ckpt ${WEIGHT_PATH} --opacity 1 --palette semantic  
```