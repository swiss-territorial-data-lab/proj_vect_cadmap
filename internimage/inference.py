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
                        default="/scratch/izar/shanli/Cadmap/dufour_plan/Carouge/",
                        help='the dir to load initial plans')
    parser.add_argument('--save_path',
                        default="/scratch/izar/shanli/Cadmap/dufour_plan/Carouge-prediction/",
                        help='the dir to save splited patches')
    parser.add_argument('--cfg',
                        default="/home/shanli/Cadmap/InternImage/segmentation/configs/geneva_line/upernet_internimage_xl_512x1024_160k_geneva_line.py",
                        help='the dir to load initial plans')
    parser.add_argument('--ckpt',
                        default="/scratch/izar/shanli/Cadmap/internimage/upernet_internimage_xl_512x1024_160k_geneva_line-2e-5-0.7153/best_mIoU_iter_98000.pth",
                        help='the dir to save splited patches')
    parser.add_argument('--slide', default=(256, 512),
                        help='slide size when predicting')
    parser.add_argument('--opacity', type=float, default=0.5,
                        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--palette', default='line',
                        help='palette for semantic or borderline prediction.')

    args = parser.parse_args()

    return args


# class LoadImage:
#     """A simple pipeline to load image."""

#     def __call__(self, results):
#         """Call function to load images into results.

#         Args:
#             results (dict): A result dict contains the file name
#                 of the image to be read.

#         Returns:
#             dict: ``results`` will be returned containing loaded image.
#         """

#         if isinstance(results['img'], str):
#             results['filename'] = results['img']
#             results['ori_filename'] = results['img']
#         else:
#             results['filename'] = None
#             results['ori_filename'] = None
#             results['scale_factor'] = 1.0
#             results['img_norm_cfg'] = {'mean': [0., 0., 0.], 'std': [1., 1., 1.], 'to_rgb': False}
#         img = mmcv.imread(results['img'])
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         results['pad_shape'] = img.shape
#         return results
    

# def inference_segmentor(model, img):
#     """Inference image(s) with the segmentor.

#     Args:
#         model (nn.Module): The loaded segmentor.
#         imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
#             images.

#     Returns:
#         (list[Tensor]): The segmentation result.
#     """
#     cfg = model.cfg
#     device = next(model.parameters()).device  # model device
#     # build the data pipeline
#     test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
#     test_pipeline = Compose(test_pipeline)
#     # prepare data
#     data = dict(img=img)
#     data = test_pipeline(data)
#     data = collate([data], samples_per_gpu=1)
#     if next(model.parameters()).is_cuda:
#         # scatter to specified GPU
#         data = scatter(data, [device])[0]
#     else:
#         data['img_metas'] = [i.data[0] for i in data['img_metas']]

#     # forward the model
#     pdb.set_trace()
#     with torch.no_grad():
#         seg_logit = model.inference(rescale=True, **data)
#     return seg_logit


def main():
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

    origin_plan_ls = os.listdir(input_path)

    origin_plan_ls = [file for file in origin_plan_ls if file.endswith('.tif') and not file.endswith('Assemblage.tif')]

    # for plan in origin_plan_ls:
    #     print('predicting: {}'.format(plan))
    #     plan_path = os.path.join(input_path, plan)
    #     plan_img = cv2.imread(plan_path)
    #     # get shape
    #     H, W, _ = plan_img.shape
    #     h, w = size
        
    #     # crop with grid
    #     row = int(np.ceil(H / h))
    #     col = int(np.ceil(W / w))
    #     padded_img = np.pad(plan_img, ((0, row*h-H), (0, col*w-W), (0, 0)), mode='constant', constant_values=np.transpose([[0,0,0], [0,0,0]]))
    #     logit_plan = np.zeros([*padded_img.shape[:2], len(palette)])
    #     mask_plan = np.zeros_like(padded_img)

    #     y1 = 0
    #     y2 = y1 + h

    #     while y2 < H:
    #         # for c in range(col):
    #         x1 = 0
    #         x2 = x1 + w
    #         while x2 < W:
    #             # crop bounding box
    #             img_crop = padded_img[y1:y2,x1:x2]
    #             seg_logit = inference_segmentor(model, img_crop)
    #             logit_plan[y1:y2, x1:x2, :] = torch.maximum(logit_plan[y1:y2, x1:x2, :], seg_logit)
    #             seg_pred = seg_logit.argmax(dim=1)
    #             seg_pred = seg_pred.cpu().numpy()
    #             seg_pred = list(seg_pred)
    #             patch_mask = model.show_result(img_crop, seg_pred, palette=palette, opacity=opacity)
    #             mask_plan[y1:y2, x1:x2] = patch_mask
                

    #             # update bounding box
    #             x1 = x1 + slide[1] 
    #             x2 = x1 + w

    #         y1 = y1 + slide[0] 
    #         y2 = y1 + h


    
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
                