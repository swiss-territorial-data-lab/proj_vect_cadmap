# Vectorization of Historical Cadastral Maps
This repository is the implementation of project [*Vectorization of Historical Cadastral Maps*](https://tech.stdl.ch/PROJ-ROADSURF/) by [STDL](https://www.stdl.ch/). This project utilizes the capacity of computer vision algorithm and deep learning techniques to achieve the semi-automatic vectorization of histogram cadastral maps in Geneva (1850s). Details can be explored in this [report](./assets/Vectorization_Cadmap_report.pdf). 

A annotation pipeline was developed to get the ground truth label for semantic segmentation neural networks. We also created a historical cadastral map dataset with the label for binary and the multi-class segmentation. 

The deep learning models are build based on the official implementation of [InternImage](https://github.com/opengvlab/internimage) framework. In this project, a CNN architecture **UperNet** and a ViT (Vision Transformer) **Segformer** are tested. 

## Hardware requirements

OS: Linux 

In the experimental process, we used 4 * NVIDIA V100 GPU (32GB). We strongly recommend using GPU with a memory capacity of 16GB or more for training the deep learning models in this project.  

## Software Requirements

Model training and vectorization:
* NVIDIA CUDA 11.6
* Python 3.8

    ```bash
	#  The dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file.
    $ conda create -n cadmap -c conda-forge python=3.8 gdal
    $ conda activate cadmap
    $ pip install -r setup/requirements.txt
    ```

Annotation pipeline to create your own dataset:
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

The project uses Geo-referenced Tiff file as the start point. With the annotation pipeline, it created 2 datasets: `Geneva_line` and `Geneva_semantic` with thousands of image tiles and their corresponding semantic masks for binary and multi-class semantic segmentation. The prediction of binary segmentation (borderline) is conducted to extract the topology of the map, which is later vectorized to obtain the vector polygon for parcels. Multi-class semantic segmentation module aims to identity the classification of the extracted polygons. The optical recognition module targets on detect and recognize the parcel/building indexes shown on the map. Finally, all the information are aggregated to a .shp file and projected to the real spatial referenced coordinate system.

The workflow of the project is illustrated in the image above. Original geo-referenced cadastral map (GeoTiff) file is provided by the domain expertise from Conton of Geneva. The scanned cadastral maps are manually calibrated within Geneva_local coordinate system. If you do not want to reproduce the annotation pipeline, you can start from [dataset generation](#generate-datasets) with published data. Otherwise, please following the procedure below.


### Data preparation
#### Download link 

Download these data to its folder shown in the folder structure.

[Dufour_Origine_plans]()

[Annotations]() 

[Raster GeoTiff]()


### Annotate the cadastral map with ArcGIS Pro & Photoshop 

The detailed procedures are recorded in the [report](./assets/Vectorization_Cadmap_report.pdf) (page 14). Follow the method documented and store the project folder of each annotated map in the `./data/arcgis/`. Export the raster map file (.png) to `./data/annotations/images/`, the binary and multi-class semantic labels to `./data/annotations/labels_line/` and `./data/annotations/labels_line/labels_semantic/`.	

***Attention: the project name should be consist with the original .tif file name.***

### Generate datasets
This script samples the cadastral map image with high resolution (around 12,500 * 8,000) to small image tiles with corresponding labels. The size and amount of the random sampled tiles can be configured with arguments.

```
python scripts/dataset_generation.py --input_path <path to annotation folder> --save_path <path to dataset folder> 
```

The split of `train`, `val` and `test` set is implemented on the cadastral map level. The user need to split the annotated maps and labels and then run the script on each set. 

After that,  copy or link the generated tiles to the `./internimage/data` folder, which will be organized as following:

```
├── internimage									# InternImage Framework for segmentation
    └── data
        ├── geneva_line
        │   ├── images
        │   │   ├── train
        │   │   └── val
        │   └── labels
        │       ├── train
        │       └── val
        └── geneva_semantic
            ├── images
            │   ├── train
            │   └── val
            └── labels
                ├── train
                └── val
```

### Semantic segmentation 
Use the generated datasets and follow the instruction of InternImage [here](/internimage/README.md) to train the neural networks and produce the binary and multi-class semantic segmentation prediction mask on the test maps respectively. Results should be stored in `./data/line_prediction_mask` and `./data/semantic_prediction_mask`.

### Raster delineation
Delineate line prediction mask given the path to folder of test GeoTiff file and the masks.  

```
python scripts/raster_delineation.py --tif_path <original tif> --line_mask <line prediction mask> --save_path <delineation folder> 
```

### ArcGIS operation  
In this part, each resulted plan from raster delineation are vectorized with the following steps (elementary method as example):

1. Create a new ArcGIS map project **named** with the map filename under `./data/arcgis/` folder.
2. Import raster_tif file (.tif), elementary_result (.png).
2. Search `Raster to Polyline (Convention Tools)`: select `elementary_result.png` as input raster. Name the output ployline features as `polyline_elementary`.
3. Open the `Edit` panel, utilize tools like `Select`, `Create` and `Delete` to delete the polyline outside the unique area of the map. Manually remove the false positive and false negatives as much as possible. Connect missed lines to form polygons. Here, false positives that can not form a polygon can be ignored.
4. In `Catalog` panel, create a new polygon feature class named `polygon_elementary` in `Databases>project_name.gdb` file.
5. Select all features in `polyline_elementary` and open `Construct Polygons` tool in `Modify Features` panel. Choose `polygon_elementary` as template, run the tool.
6. Search `Simplify Shared Edges (Cartography Tools)`. Choose `polygon_elementary` as input feature. Set Simplification Algorithm to `Retain effective areas`, Simplification Tolerance to 5 meters. Run the algorithm.

### Post-processing 
After vectorization, this script will integrate the text recognition from [EasyOCR](https://github.com/JaidedAI/EasyOCR) and multi-class semantic information into the vector polygon as attributes.

Limited by time, the pretrained model from EasyOCR for French is used in this case. The accuracy of text recognition is strongly related to the handwriting style of the historical map. The user can do customized training on their own dataset with the API from EasyOCR or deactivate this function by setting the argument `--ocr False`.
```
python scripts/post_processing.py --arcgis <arcgis folder> --tif <original tif> --raster_tif <raster tif> --semantic <multi-class semantic prediction> --ocr <True/False> --method <elementary/sophisticated>
```

After post-processing, the projected vectorized results can be found in `output` folder. 


### Documentation
The full documentation of the project is available on the STDL's technical website: **give link here...**



