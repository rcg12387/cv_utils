#!/usr/bin/python3
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


def yolo2labelme(img_path, yolo_path, labelme_path, cls_file):
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
            labels = np.loadtxt(label_name, delimiter=' ', ndmin=2)
            for (cid, cx, cy, w, h, lt_x, lt_y, rt_x, rt_y,
                 rb_x, rb_y, lb_x, lb_y) in labels:

                # bbox
                label = categories[int(cid)]
                shape = copy.deepcopy(DEFAULT_SHAPE)
                cx *= iw
                cy *= ih
                w *= iw
                h *= ih
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                shape['label'] = label
                shape['points'] = list(zip([x1, x2], [y1, y2]))
                shape['group_id'] = None
                shape['shape_type'] = 'rectangle'
                shape['flags'] = {}
                shapes.append(shape)

                # points
                shapes.append(make_point(lt_x, lt_y, iw, ih, 'lt'))
                shapes.append(make_point(rt_x, rt_y, iw, ih, 'rt'))
                shapes.append(make_point(rb_x, rb_y, iw, ih, 'rb'))
                shapes.append(make_point(lb_x, lb_y, iw, ih, 'lb'))

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
    parser.add_argument('--cls-file', default='classes.txt', type=str)
    opt = parser.parse_args()
    yolo2labelme(opt.images, opt.yolo, opt.labelme, opt.cls_file)
