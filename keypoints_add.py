# Add keypoints in the middle of two vertices
#   lt      t       rt
#   l               r
#   lb      b       rb
import argparse
import json
import os
import os.path as osp
import logging

import numpy as np
from tqdm import tqdm

# DEF_CATEGORIES = ['leye', 'reye', 'nose', 'lmouth', 'rmouth']
DEF_CATEGORIES = ['lt', 'rt', 'lb', 'rb', 'l', 'r', 't', 'b']


def make_mid_point(pt1, pt2, label):
    x = (pt1[0] + pt2[0]) / 2
    y = (pt1[1] + pt2[1]) / 2
    shape = {
        'label': label,
        'points': [(x, y)],
        'group_id': None,
        'shape_type': 'point',
        'flags': {}
    }
    return shape


def labelme2yolo(in_path, out_path, class_file):
    os.makedirs(osp.join(out_path), exist_ok=True)

    if class_file is None:
        categories = []
    else:
        classes_file = open(class_file, 'r')
        categories = classes_file.read().strip('\n').split('\n')

    if osp.isfile(in_path):
        files = [in_path]
        in_path = osp.dirname(in_path)
    else:
        files = [
            osp.join(in_path, f) for f in os.listdir(in_path)
            if osp.splitext(f)[-1] == '.json'
        ]
    for in_file in tqdm(files):
        with open(in_file, 'r') as f:
            data = json.loads(f.read())
        shapes_list = data['shapes']
        if not shapes_list:
            continue

        img_path = data['imagePath']
        img_path = osp.relpath(img_path, out_path)[3:]
        data['imagePath'] = img_path

        for shape in shapes_list:
            label = shape['label']
            if shape['shape_type'] == 'point':
                if label == 'leye':
                    shape['label'] = 'lt'
                    lt_pt = np.float32(shape['points']).reshape(-1).tolist()
                elif label == 'reye':
                    shape['label'] = 'rt'
                    rt_pt = np.float32(shape['points']).reshape(-1).tolist()
                elif label == 'lmouth':
                    shape['label'] = 'lb'
                    lb_pt = np.float32(shape['points']).reshape(-1).tolist()
                elif label == 'rmouth':
                    shape['label'] = 'rb'
                    rb_pt = np.float32(shape['points']).reshape(-1).tolist()
                elif label == 'nose':
                    nose_shape = shape

        # additional points
        shapes_list.remove(nose_shape)
        shapes_list.append(make_mid_point(lt_pt, lb_pt, 'l'))
        shapes_list.append(make_mid_point(rt_pt, rb_pt, 'r'))
        shapes_list.append(make_mid_point(lt_pt, rt_pt, 't'))
        shapes_list.append(make_mid_point(lb_pt, rb_pt, 'b'))

        with open(osp.join(out_path, osp.basename(in_file)), 'w') as f:
            f.write(json.dumps(data, indent=4, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', default='', type=str, help='Input path')
    parser.add_argument('--out-path', default='', type=str, help='Output path')
    parser.add_argument('--class-file', default=None, type=str, help='Text file that includes category names')
    opt = parser.parse_args()
    labelme2yolo(opt.in_path, opt.out_path, opt.class_file)
