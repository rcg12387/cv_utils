#!/usr/bin/python3
# Convert labelme rect $ keypoints to yolo txt to train with yolov5-face
import argparse
import json
import os
import os.path as osp
import logging

import numpy as np
from tqdm import tqdm


def seg_labelme2yolo(labelme_path, yolo_path, class_file):
    os.makedirs(osp.join(yolo_path), exist_ok=True)

    if class_file is None:
        categories = []
    else:
        classes_file = open(class_file, 'r')
        categories = classes_file.read().strip('\n').split('\n')
        if categories[0] == '__ignore__':
            categories.pop(0)

    if osp.isfile(labelme_path):
        files = [labelme_path]
        labelme_path = osp.dirname(labelme_path)
    else:
        files = [
            osp.join(labelme_path, f) for f in os.listdir(labelme_path)
            if osp.splitext(f)[-1] == '.json'
        ]
        files.sort()
    for labelme_file in tqdm(files):
        with open(labelme_file, 'r') as f:
            data = json.loads(f.read())
        img_name = osp.split(data['imagePath'])[-1]
        img_width = data['imageWidth']
        img_height = data['imageHeight']

        seg_ann_list = []
        seg_ann = None
        for shapes in data['shapes']:
            category = shapes['label']
            # rectangle
            if shapes['shape_type'] == 'polygon':
                if category not in categories:
                    categories.append(category)
                cid = categories.index(category)

                points = np.array(np.float32(shapes['points']).reshape(-1).tolist())
                points[::2] /= img_width
                points[1::2] /= img_height
                seg_ann = ({
                    'category_id': cid,
                    'points': points,
                })
            if seg_ann is not None:
                seg_ann_list.append(seg_ann)

        yolo_list = []
        for seg_ann in seg_ann_list:
            cid = seg_ann['category_id']
            points = seg_ann['points']
            txt_format = '%d' + ' %g' * points.__len__()
            yolo_list.append(txt_format % (cid, *points))

        lable_name = osp.splitext(img_name)[0] + '.txt'
        # if len(yolo_list):
        with open(osp.join(yolo_path, lable_name), 'w') as f:
            f.write('\n'.join(yolo_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme', default='labelme', type=str, help='Path to save labels for labelme style')
    parser.add_argument('--yolo', default='labelme', type=str, help='Path to save labels for yolo style')
    parser.add_argument('--cls-file', default=None, type=str, help='Text file that includes category names')
    opt = parser.parse_args()
    seg_labelme2yolo(opt.labelme, opt.yolo, opt.cls_file)
