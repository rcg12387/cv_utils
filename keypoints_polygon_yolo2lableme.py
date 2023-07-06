import argparse
import json
import os
import os.path as osp
import copy
import glob
import itertools

import imagesize
import numpy as np
from tqdm import tqdm

DEFAULT_LABELME = {
    "version": "5.0.2",
    "flags": {},
}

DEFAULT_SHAPE = {"label": None}


def getFilenames(dir_path, exts):
    fnames = [glob.glob(osp.join(dir_path, ext)) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    fnames.sort()
    return fnames


def make_point(x, y, imW, imH, label):
    x *= imW
    y *= imH
    shape = {
        'label': label,
        'points': [(x, y)],
        'group_id': None,
        'shape_type': 'point',
        'flags': {}
    }
    return shape


def keypoints_polygon_yolo2lableme(img_path, yolo_path, labelme_path, cls_file):
    os.makedirs(osp.join(labelme_path), exist_ok=True)
    if cls_file is None:
        dir_path = os.path.dirname(os.path.realpath(yolo_path))
        class_list_path = os.path.join(dir_path, "classes.txt")
    else:
        class_list_path = cls_file

    classes_file = open(class_list_path, 'r')
    categories = classes_file.read().strip('\n').split('\n')

    img_list = getFilenames(img_path, ['*.jpg', '*.png', '*.bmp'])
    for img in tqdm(img_list):
        labelme_json = copy.deepcopy(DEFAULT_LABELME)

        img = osp.split(img)[-1]
        iw, ih = imagesize.get(osp.join(img_path, img))
        label_name = osp.join(yolo_path,
                              ''.join(*(osp.splitext(img)[:-1])) + '.txt')
        shapes = []
        if osp.exists(label_name):
            with open(label_name) as f:
                labels = f.read().strip().splitlines()
                for label in labels:
                    if len(label) == 0:
                        continue

                    ann = label.split()
                    shape = copy.deepcopy(DEFAULT_SHAPE)

                    # Category
                    category = categories[int(ann[0])]

                    # Points
                    points = np.array([float(x) for x in ann[5:]])
                    points[::2] *= iw
                    points[1::2] *= ih

                    shape['label'] = category
                    shape['points'] = list(zip(points[::2], points[1::2]))
                    shape['group_id'] = None
                    shape['shape_type'] = 'polygon'
                    shape['flags'] = {}
                    shapes.append(shape)

            labelme_json['shapes'] = shapes
            labelme_json['imagePath'] = osp.relpath(osp.join(img_path, img), labelme_path)
            labelme_json['imageData'] = None
            labelme_json['imageHeight'] = ih
            labelme_json['imageWidth'] = iw
            with open(osp.join(labelme_path, osp.splitext(osp.basename(img))[0] + '.json'), 'w') as f:
                f.write(json.dumps(labelme_json, indent=4, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='images', type=str)
    parser.add_argument('--labelme', default='labelme', type=str)
    parser.add_argument('--yolo', default='yolo', type=str)
    parser.add_argument('--cls-file', default=None, type=str)
    opt = parser.parse_args()
    keypoints_polygon_yolo2lableme(opt.images, opt.yolo, opt.labelme, opt.cls_file)
