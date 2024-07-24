from doctest import OutputChecker
import os
import cv2
import mmcv
import argparse
import os.path as osp
import numpy as np
import pdb

from PIL import Image
from mmseg.apis import init_segmentor, inference_segmentor


import shutil
import time
import tqdm
import warnings

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate, scatter
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model, load_state_dict)
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose



import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403


np.random.seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description='Split plans')
    parser.add_argument('--input_path',
                        required=True,
                        help='the dir to load initial plans')
    parser.add_argument('--save_path',
                        required=True,
                        help='the dir to save splited patches')
    parser.add_argument('--cfg',
                        required=True,
                        help='the dir to load initial plans')
    parser.add_argument('--ckpt',
                        required=True,
                        help='the dir to save splited patches')
    parser.add_argument('--slide', default=(256, 512),
                        help='slide size when predicting')
    parser.add_argument('--opacity', type=float, default=0.5,
                        required=True,
                        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--palette', choices=['line', 'semantic'],
                        required=True,
                        help='palette for semantic or borderline prediction.')

    args = parser.parse_args()

    return args


def main():

    # init parameters
    args = parse_args()
    input_path = args.input_path
    save_path = args.save_path
    config_file = args.cfg
    checkpoint_file = args.ckpt
    slide = args.slide
    opacity = args.opacity
    size = (512, 1024)

    if args.palette == 'line':
        palette = [[0, 0, 0], [255, 255, 255]]
    
    elif args.palette == 'semantic':
        palette = [[0, 0, 0], [255, 255, 255], [255, 115, 223], [211, 255, 190], [78, 78, 78], [255, 255, 0], [190, 232, 255]]
    # model = init_config(args)

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    if hasattr(model, 'module'):
        model = model.module
    # model = MMDataParallel(model, device_ids=[0])

    assert osp.exists(input_path)

    if not os.path.exists(save_path): 
        os.mkdir(save_path)


    # iterate through the .tff file in the map folder
    origin_plan_ls = os.listdir(input_path)
    origin_plan_ls = [file for file in origin_plan_ls if file.endswith('.tif') and not file.endswith('Assemblage.tif')]
    
    for plan in origin_plan_ls:
        print('predicting: {}'.format(plan))
        plan_path = os.path.join(input_path, plan)
        plan_img = cv2.imread(plan_path)
        # get shape
        H, W, _ = plan_img.shape
        h, w = size
        
        # crop with grid
        row = int(np.ceil(H / h))
        col = int(np.ceil(W / w))
        padded_img = np.pad(plan_img, ((0, row*h-H), (0, col*w-W), (0, 0)), mode='constant', constant_values=np.transpose([[0,0,0], [0,0,0]]))
   
        seg_pred = inference_segmentor(model, padded_img)
        pred_plan = model.show_result(padded_img, seg_pred, palette=palette, opacity=opacity)

        pred_path = osp.join(save_path, plan.split('.')[0] + '.png')
        cv2.imwrite(pred_path, pred_plan[:H, :W, :])
        print('Prediction saved at {}'.format(pred_path))


if __name__ == '__main__':
    main()
                