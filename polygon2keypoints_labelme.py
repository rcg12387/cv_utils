#!/usr/bin/python3
# Convert labelme rect $ keypoints to yolo txt to train with yolov5-face
import argparse
import json
import os
import os.path as osp
import logging
import copy

import numpy as np
from tqdm import tqdm

DEFAULT_LABELME = {
    "version": "5.0.2",
    "flags": {},
}


def polygon2keypoints(kp_path, poly_path):
    os.makedirs(osp.join(poly_path), exist_ok=True)
    categories = []

    if osp.isfile(kp_path):
        files = [kp_path]
        kp_path = osp.dirname(kp_path)
    else:
        files = [
            osp.join(kp_path, f) for f in os.listdir(kp_path)
            if osp.splitext(f)[-1] == '.json'
        ]
    for kp_file in tqdm(files):
        with open(kp_file, 'r') as f:
            kp_data = json.loads(f.read())

        poly_data = copy.deepcopy(DEFAULT_LABELME)

        poly_shapes = []
        polygon = None
        for shapes in kp_data['shapes']:
            label = shapes['label']

            # rectangle
            if shapes['shape_type'] == 'rectangle':
                if polygon is None:
                    polygon = ({
                        'label': shapes['label'],
                        'points': None,
                        'group_id': None,
                        'shape_type': 'polygon',
                        'flags': {}
                    })
                else:
                    polygon['points'] = list(zip([lt[0], rt[0], rb[0], lb[0]], [lt[1], rt[1], rb[1], lb[1]]))
                    poly_shapes.append(copy.deepcopy(polygon))
            # keypoints
            elif shapes['shape_type'] == 'point':
                if shapes['label'] == 'lt':
                    lt = np.float32(shapes['points']).reshape(-1).tolist()
                elif shapes['label'] == 'rt':
                    rt = np.float32(shapes['points']).reshape(-1).tolist()
                elif shapes['label'] == 'lb':
                    lb = np.float32(shapes['points']).reshape(-1).tolist()
                elif shapes['label'] == 'rb':
                    rb = np.float32(shapes['points']).reshape(-1).tolist()

        polygon['points'] = list(zip([lt[0], rt[0], rb[0], lb[0]], [lt[1], rt[1], rb[1], lb[1]]))
        poly_shapes.append(copy.deepcopy(polygon))

        poly_data['shapes'] = poly_shapes
        poly_data['imagePath'] = kp_data['imagePath']
        poly_data['imageData'] = kp_data['imageData']
        poly_data['imageHeight'] = kp_data['imageHeight']
        poly_data['imageWidth'] = kp_data['imageWidth']

        poly_file = osp.join(poly_path, osp.split(kp_file)[-1])
        # if len(yolo_list):
        with open(poly_file, 'w') as f:
            f.write(json.dumps(poly_data, indent=4, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--kp', default='json-keypoints', type=str, help='Path to save labels for keypoints style')
    parser.add_argument('--poly', default='json-polygon', type=str, help='Path to save labels for polygon style')
    opt = parser.parse_args()
    polygon2keypoints(opt.kp, opt.poly)
