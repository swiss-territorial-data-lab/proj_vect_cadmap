import cv2
import numpy as np
import os
import os.path as osp
from PIL import Image
import argparse
from tqdm import tqdm

np.random.seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset given the scanned cadastral plans and corresponding labels.')
    parser.add_argument('--input_path',
                        default="./data/annotations",
                        help='the dir to load initial plans and labels')
    parser.add_argument('--save_path',
                        default="./data/dataset",
                        help='the dir to save segmented tiles')
    parser.add_argument('--size', default=1024,
                        help='Generated tile size')
    parser.add_argument('--nbr_tiles', default=500,
                        help='Generated tile numbers.')
    parser.add_argument('--grid', default=True,
                        help='Whether initiate the tile generation with grid split.')

    args = parser.parse_args()

    return args

def crop_bbox(img_name, input_path, save_path, Crop_N=300, L=2048, grid=True):

    # config the folder names 
    img_folder = "images"
    rgb_line_folder = "labels_line"
    grayscale_line_folder = "labels_line_gray"
    rgb_semantic_folder = "labels_semantic"
    grayscale_semantic_folder = "labels_semantic_gray"

    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    for folder in [img_folder, rgb_line_folder, grayscale_line_folder, rgb_semantic_folder, grayscale_semantic_folder]:
        if not os.path.exists(osp.join(save_path, folder)):
            os.mkdir(osp.join(save_path, folder))       

    img_path = os.path.join(input_path, img_folder, img_name)
    rgb_line_path = os.path.join(input_path, rgb_line_folder, img_name)
    grayscale_line_path = os.path.join(input_path, grayscale_line_folder, img_name)
    rgb_semantic_path = os.path.join(input_path, rgb_semantic_folder, img_name)
    grayscale_semantic_path = os.path.join(input_path, grayscale_semantic_folder, img_name)

    # loading data
    img = cv2.imread(img_path)
    rgb_line = cv2.imread(rgb_line_path)
    grayscale_line = cv2.imread(grayscale_line_path, cv2.IMREAD_GRAYSCALE)
    rgb_semantic = cv2.imread(rgb_semantic_path)
    grayscale_semantic = cv2.imread(grayscale_semantic_path, cv2.IMREAD_GRAYSCALE)
    # get shape
    H, W, C = img.shape
    count = 0
    
    # grid segmentation of the plan 
    if grid:
        # crop with grid
        row = int(np.ceil(H / L))
        col = int(np.ceil(W / L))
        padded_img = np.pad(img, ((0, row*L-H), (0, col*L-W), (0, 0)), mode='constant', constant_values=np.transpose([[0,0,0], [0,0,0]]))
        padded_rgb_line = np.pad(rgb_line, ((0, row*L-H), (0, col*L-W), (0, 0)), mode='constant', constant_values=np.transpose([[0,0,0], [0,0,0]]))
        padded_grayscale_line = np.pad(grayscale_line, ((0, row*L-H), (0, col*L-W)), mode='constant', constant_values=0)
        padded_rgb_semantic = np.pad(rgb_semantic, ((0, row*L-H), (0, col*L-W), (0, 0)), mode='constant', constant_values=np.transpose([[0,0,0], [0,0,0]]))
        padded_grayscale_semantic = np.pad(grayscale_semantic, ((0, row*L-H), (0, col*L-W)), mode='constant', constant_values=0)

        for r in range(row):
            for c in range(col):
                x1 = c * L 
                x2 = x1 + L
                y1 = r * L 
                y2 = y1 + L

                # crop bounding box
                img_crop = padded_img[y1:y2,x1:x2]
                nbr_bg_pixel = L**2 - np.count_nonzero(img_crop) / 3
                if nbr_bg_pixel / L**2 == 1.0:
                    continue
                rgb_line_crop = padded_rgb_line[y1:y2,x1:x2]
                grayscale_line_crop = padded_grayscale_line[y1:y2,x1:x2]
                rgb_semantic_crop = padded_rgb_semantic[y1:y2,x1:x2]
                grayscale_semantic_crop = padded_grayscale_semantic[y1:y2,x1:x2]

                crop_file_name = img_name[:-4] + "_" + str(count) + ".png"
                cv2.imwrite(osp.join(save_path, img_folder, crop_file_name), img_crop)
                cv2.imwrite(osp.join(save_path, rgb_line_folder, crop_file_name), rgb_line_crop)
                cv2.imwrite(osp.join(save_path, grayscale_line_folder, crop_file_name), grayscale_line_crop)
                cv2.imwrite(osp.join(save_path, rgb_semantic_folder, crop_file_name), rgb_semantic_crop)
                cv2.imwrite(osp.join(save_path, grayscale_semantic_folder, crop_file_name), grayscale_semantic_crop)
                count += 1
        print('grid split: {} images'.format(count))
        
    # each crop
    while count < Crop_N:
        x1 = np.random.randint(W - L)
        # get left top y of crop bounding box
        y1 = np.random.randint(H - L)
        # get right bottom x of crop bounding box
        x2 = x1 + L
        # get right bottom y of crop bounding box
        y2 = y1 + L

        # crop bounding box
        img_crop = padded_img[y1:y2,x1:x2]
        nbr_bg_pixel = L**2 - np.count_nonzero(img_crop) / 3
        if nbr_bg_pixel / L**2 > 0.5:
            continue
        rgb_line_crop = padded_rgb_line[y1:y2,x1:x2]
        grayscale_line_crop = padded_grayscale_line[y1:y2,x1:x2]
        rgb_semantic_crop = padded_rgb_semantic[y1:y2,x1:x2]
        grayscale_semantic_crop = padded_grayscale_semantic[y1:y2,x1:x2]

        crop_file_name = img_name[:-4] + "_" + str(count) + ".png"
        cv2.imwrite(osp.join(save_path, img_folder, crop_file_name), img_crop)
        cv2.imwrite(osp.join(save_path, rgb_line_folder, crop_file_name), rgb_line_crop)
        cv2.imwrite(osp.join(save_path, grayscale_line_folder, crop_file_name), grayscale_line_crop)
        cv2.imwrite(osp.join(save_path, rgb_semantic_folder, crop_file_name), rgb_semantic_crop)
        cv2.imwrite(osp.join(save_path, grayscale_semantic_folder, crop_file_name), grayscale_semantic_crop)
        count += 1


def grayscle_convert_from_color(arr_3d, palette):
    """RGB-color encoding to grayscale labels."""

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8) * 255
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


def main():

    # init 
    args = parse_args()
    input_path = args.input_path
    save_path = args.save_path
    nbr_tiles = args.nbr_tiles
    size = args.size
    img_folder = "images"
    rgb_line_folder = "labels_line"
    rgb_semantic_folder = "labels_semantic"

    if not osp.exists(input_path): 
        raise Exception('Input folder does not exists!')

    # transform rgb label into grayscale label 

    line_palette = \
    {
        0: (0, 0, 0),
        1: (255, 255, 255)
    }

    semantic_palette = \
    {
        0: (0, 0, 0),           # background
        1: (255, 255, 255),     # borderline
        2: (255, 115,223),      # building
        3: (211,255,190),       # unbuilt   
        4: (78, 78, 78),        # wall
        5: (255, 255, 0),       # road  
        6: (190, 232, 255)      # river      
    }

    line_invert_palette = {v: k for k, v in line_palette.items()}
    semantic_invert_palette = {v: k for k, v in semantic_palette.items()}    

    img_lists = os.listdir(osp.join(input_path, img_folder))

    # convert line and semantic mask respectively
    for label_folder, palette in [(rgb_line_folder, line_invert_palette), (rgb_semantic_folder, semantic_invert_palette)]:
        print("Generating grayscale label for folder: " + label_folder)
        # iterate each plan 
        gray_folder = osp.join(input_path, label_folder + '_gray')
        if not osp.exists(gray_folder): 
            os.mkdir(gray_folder)
        for img_name in tqdm(img_lists):
            label_bgr = cv2.imread(osp.join(input_path, label_folder, img_name))
            label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
            label_gray = grayscle_convert_from_color(label_rgb, palette)    
            label = Image.fromarray(label_gray.astype(np.uint8), mode='P')
            label.save(osp.join(gray_folder, img_name))
    print("Generating grayscale label: Done!")

    
    for img_name in img_lists:
        print("Generating tiles: " + img_name)
        crop_bbox(img_name, input_path, save_path, L=size, Crop_N=nbr_tiles, grid=True)
    print("Generating tiles: Done!")

if __name__ == '__main__':
    main()
















