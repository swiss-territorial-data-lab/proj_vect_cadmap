# Vectorization of Historical Cadastral Plans

This repository is the implementation of the project *Vectorization of Historical Cadastral Plans* conducted by the [STDL](https://www.stdl.ch/). It uses the capacity of computer vision algorithms and deep learning techniques to achieve semi-automatic vectorization of historical cadastral plans in Geneva (1850s). Details of the project can be found in the published [documentation](https://tech.stdl.ch/PROJ-/) and in this [report](./assets/Vectorization_Cadmap_report.pdf). 

An annotation pipeline was developed to get the ground truth label for semantic segmentation neural networks. An historical cadastral map dataset with the label for binary and the multi-class segmentation as also created. 

The deep learning models are built based on the official implementation of [InternImage](https://github.com/opengvlab/internimage) framework. In this project, a CNN architecture **UperNet** and a ViT (Vision Transformer) **Segformer** are tested. 

## Hardware requirements

OS: Linux 

In the experimental process, we used 4 * NVIDIA V100 GPU (32 GB). We strongly recommend using GPU with a memory capacity of 16 GB or more for training the deep learning models in this project.  

## Software Requirements

Model training and vectorization:
* NVIDIA CUDA 11.6
* Python 3.10

    ```bash
	#  The dependencies is suggested to be installed with `conda`, as GDAL is much complicated to config with `pip`
    $ conda create -n cadmap -c conda-forge python=3.10
    $ conda activate cadmap
    $ pip install -r setup/requirements.txt
    ```

The following software tools were used to create the annotated dataset:
* Photoshop 
* ArcGIS Pro

## Folder structure

```
├── assets										# images for README.md
├── data
│   ├── Dufour_Origine_Plan						# Cadastral map GeoTiff file folder  
│   ├── annotations								# Annotated cadastral map and raster labels for semantic segmentation 
│   │   ├── images								# Cadastral map (png) file folder
│   │   ├── labels_line							# RGB image of line mask for visualization 
│   │   ├── labels_line_gray					# Grayscale image of line mask for model training  
│   │   ├── labels_semantic						# RGB image of multi-class semantic mask for visualization 
│   │   └── labels_semantic_gray				# Grayscale image of multi-class semantic mask for model training  
│   ├── arcgis									# ArcGIS project file for each cadastral map
│   ├── delineation								# Folder of delineation temp file 
│   ├── line_prediction_mask					# Folder of line prediction mask
│   ├── semantic_prediction_mask				# Folder of multi-class semantic prediction mask
│   ├── raster_tif								# tif file in image frame exported from arcgis project
│   └── tiles									# Generated map tiles and all the corresponding labels 
│       ├── images								# Map tiles
│       ├── labels_line							# RGB image of line mask
│       ├── labels_line_gray					# Grayscale image line mask tiles
│       ├── labels_semantic						# RGB multi-class semantic mask tiles
│       └── labels_semantic_gray				# Grayscale multi-class semantic mask tiles
├── internimage									# InternImage Framework for segmentation 
|   ├── ckpt-weights                            # Checkpoints weights folder
│   ├── configs									# Config files
│   │   ├── _base_								# Base configuration about model architecture and dataset
│   │   ├── geneva_line							# Training config files for line segmentation in Geneva dataset 
│   │   ├── geneva_semantic						# Training config files for semantic segmentation in Geneva dataset 
│   │   ├── pretrain_line						# Training config files for line segmentation in Lausanne and Neuchatel
│   │   └── trans_geneva_line					# Training config files for transfer learning 
│   ├── data									# Dataset folder
│   │   ├── geneva_line							
│   │   └── geneva_semantic
│   ├── deploy									# Config file for deployment 
│   ├── mmcv_custom								# Custom modification for mmcv 
│   ├── mmseg_custom							# Custom modification for mmseg 
│   └── ops_dcnv3								# Deformable convolution V3 package folder
├── output										# Vectorized results							
├── scripts										# Scripts for data generation and post-processing 
│   ├── dataset_generation.py					
│   ├── post_processing.py
│   └── raster_delineation.py
└── setup										# Environment configuration file
    └── requirements.txt
```

## Scripts and Procedure

<figure align="center">
<image src="assets/workflow.svg" >
</figure>

The project workflow is illustrated in the image above. It uses georeferenced (GeoTiff) file as a starting point. The annotation pipeline creates two datasets, `Geneva_line` and `Geneva_semantic`, with thousands of image tiles and their corresponding semantic masks for binary and multi-class semantic segmentation. Binary segmentation prediction (borderline) is performed to extract the plan topology, which is then vectorized to obtain the vector polygon of the parcels. The multi-class semantic segmentation module identifies the classification of the extracted polygons. The optical recognition module aims to detect and recognize the parcel/building indices appearing on the plan. Finally, all the information is aggregated in a shape file and projected onto the spatially referenced coordinate system.

For this project, the original georeferenced cadastral plan (GeoTiff) file is supplied by the domain experts from the canton of Geneva. The scanned cadastral plans are manually georeferenced in the local Geneva coordinate system. 

If you do not want to reproduce the annotation pipeline, you can start from [dataset generation](#generate-datasets) with the published data. Otherwise, please follow the procedure below.


### Data preparation

#### Data

Input data (Dufour plans, Annotations and Raster GeoTiff files) and model weights are available on request.

### Annotate the cadastral plans with ArcGIS Pro & Photoshop 

The detailed annotation procedures are documented in the [report](./assets/Vectorization_Cadmap_report.pdf) (page 14). Follow the method and store the project folder of each annotated plan in `./data/arcgis/`. Export the raster plan (png) file to `./data/annotations/images/`, the binary and multi-class semantic labels to `./data/annotations/labels_line/` and `./data/annotations/labels_line/labels_semantic/` respectively.	

***Attention: the project name should be consist with the original .tif file name.***

### Generate datasets

This script samples the high-resolution cadastral plan image (approx. 12,500 * 8,000) into small image tiles with corresponding labels. The size and quantity of the randomly sampled tiles can be configured with arguments.

```
python scripts/dataset_generation.py --input_path <path to annotation folder> --save_path <path to dataset folder> 
```

The split of `train`, `val` and `test` sets is implemented at the cadastral plan level. The user must to split the annotated plans and labels and then run the scripts on each set. 

Next, copy or link the generated `images` and `labels_xxx_gray` tiles into the `./internimage/data` folder, which is organized as follows:

```
├── internimage				    # InternImage Framework for segmentation
    └── data
        ├── geneva_line         # Binary segmentation dataset 
        │   ├── images
        │   │   ├── train
        │   │   └── val
        │   └── labels
        │       ├── train
        │       └── val
        └── geneva_semantic     # Multi-class semantic segmentation dataset
            ├── images
            │   ├── train
            │   └── val
            └── labels
                ├── train
                └── val
```

### Semantic segmentation 

Use the generated datasets and follow the instruction of InternImage [here](/internimage/README.md) to train the neural networks and produce the binary and multi-class semantic segmentation prediction masks on the test plans. Results should be saved in `./data/line_prediction_mask` and `./data/semantic_prediction_mask`.


### Raster delineation

Delineate the line prediction mask using the path to folder containing the test GeoTiff file and masks.  

```
python scripts/raster_delineation.py --tif_path <path to the initial tiff file folder> --line_mask <path to the line prediction mask folder> --save_path <path to save delineation result folder> 
```

### ArcGIS operation 

In this part, each plan resulting from the raster delineation is vectorized according to the following steps (elementary method as an example):

1. Create a new ArcGIS project **named** with the name of the plan file in the `./data/arcgis/` folder.
2. Import the `raster_tif` (.tif) and `elementary_result` (.png) files.
2. Search for `Raster to Polyline` in ArcGIS Conversion tools: select `elementary_result.png` as input raster. Name the output polyline `elementary_polyline`.
3. Open the `Edit` panel, use tools such as `Select`, `Create` and `Delete` to remove the polyline from the single plan area. Manually remove false positives and false negatives wherever possible. Connect missing lines to form polygons. Here, false positives that cannot form a polygon can be ignored.
4. In the `Catalog` panel, create a new polygon feature class named `polygon_elementary` in the `Databases>project_name.gdb` file.
5. Select all features in `polyline_elementary` and open the `Construct Polygons` tool in the `Modify Features` panel. Choose `polygon_elementary` as template and run the tool.
6. Search for `Simplify shared edges (Mapping tools)`. Choose `polygon_elementary` as input feature. Set the simplification algorithm to `Retain effective areas` and the simplification tolerance to 5 meters. Run the algorithm.

### Post-processing 

After vectorization, the workflow integrates text recognition from [EasyOCR](https://github.com/JaidedAI/EasyOCR) and multi-class semantic information into the vector polygon as attributes.

Due to time constraints, the pre-trained model of EasyOCR for French is used in this case. The accuracy of the text recognition depends strongly on the writing style of the historical plan. The user can perform custom training on his own dataset using the EasyOCR API or disable this feature by setting the argument `--ocr False`.
```
python scripts/post_processing.py --arcgis <path to arcgis output folder> --tif <path to the initial tiff file folder> --raster_tif <path to raster tif folder> --semantic <path to multi-class semantic prediction folder> --ocr <True/False> --method <elementary/sophisticated>
```

After post-processing, the projected vectorized results can be found in `output` folder. 