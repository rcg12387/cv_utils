#!/usr/bin/python3
# Convert labelme rect $ keypoints to yolo txt to train with yolov5-face
import argparse
import json
import os
import os.path as osp
import logging

import numpy as np
from tqdm import tqdm

DEF_CATEGORIES = ['leye', 'reye', 'nose', 'lmouth', 'rmouth']


def labelme2yolo(labelme_path, yolo_path, class_file):
    os.makedirs(osp.join(yolo_path), exist_ok=True)

    if class_file is None:
        categories = []
    else:
        classes_file = open(class_file, 'r')
        categories = classes_file.read().strip('\n').split('\n')

    if osp.isfile(labelme_path):
        files = [labelme_path]
        labelme_path = osp.dirname(labelme_path)
    else:
        files = [
            osp.join(labelme_path, f) for f in os.listdir(labelme_path)
            if osp.splitext(f)[-1] == '.json'
        ]
    for labelme_file in tqdm(files):
        with open(labelme_file, 'r') as f:
            data = json.loads(f.read())
        img_name = osp.split(data['imagePath'])[-1]
        img_width = data['imageWidth']
        img_height = data['imageHeight']

        inst_ann_list = []
        inst_ann = None
        for shapes in data['shapes']:
            label = shapes['label']
            # rectangle
            if shapes['shape_type'] == 'rectangle':
                if label not in categories:
                    categories.append(label)
                cid = categories.index(label)

                if inst_ann is not None:
                    inst_ann_list.append(inst_ann)

                points = np.float32(shapes['points']).reshape(-1).tolist()
                x1 = points[0]
                y1 = points[1]
                x2 = points[2]
                y2 = points[3]
                cx = (x1 + x2) / 2 / img_width
                cy = (y1 + y2) / 2 / img_height
                w = (x2 - x1) / img_width
                h = (y2 - y1) / img_height
                inst_ann = ({
                    'category_id': cid,
                    'bbox': [cx, cy, w, h],
                })
                for label in DEF_CATEGORIES:
                    inst_ann.update({label: [0, 0]})
            elif shapes['shape_type'] == 'point':
                point = np.float32(shapes['points']).reshape(-1).tolist()
                x = point[0] / img_width
                y = point[1] / img_height
                if not (label in DEF_CATEGORIES):
                    logging.getLogger('').warn(
                        "{} has the wrong point label '{}'. "
                        "Correct point labels: {}".format(
                            labelme_file, label, DEF_CATEGORIES
                        )
                    )
                inst_ann.update({
                    label: [x, y]
                })
        if inst_ann is not None:
            inst_ann_list.append(inst_ann)

        yolo_list = []
        for inst_ann in inst_ann_list:
            cid = inst_ann['category_id']
            bbox = inst_ann['bbox']
            leye = inst_ann['leye']
            reye = inst_ann['reye']
            nose = inst_ann['nose']
            lmouth = inst_ann['lmouth']
            rmouth = inst_ann['rmouth']
            yolo_list.append('%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf' %
                             (cid, bbox[0], bbox[1], bbox[2], bbox[3],
                              leye[0], leye[1], reye[0], reye[1],
                              nose[0], nose[1],
                              lmouth[0], lmouth[1], rmouth[0], rmouth[1]))

        lable_name = osp.splitext(img_name)[0] + '.txt'
        # if len(yolo_list):
        with open(osp.join(yolo_path, lable_name), 'w') as f:
            f.write('\n'.join(yolo_list))

        with open(osp.join(yolo_path, 'classes.txt'), 'w') as f:
            f.write('\n'.join(categories))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme', default='labelme', type=str, help='Path to save labels for labelme style')
    parser.add_argument('--yolo', default='labelme', type=str, help='Path to save labels for yolo style')
    parser.add_argument('--cls-file', default=None, type=str, help='Text file that includes category names')
    opt = parser.parse_args()
    labelme2yolo(opt.labelme, opt.yolo, opt.class_file)
