#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import pprint

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from openimage import load_categories
from image import box_viz

GT = False    # set True to viz gt
THRESHOLD = 0.5

def convert_coco_anns(raw):
    result = dict()
    cats = dict()
    for cat in raw['categories']:
        cats[str(cat["id"])] = cat["name"]
    for img in raw['images']:
        result[img["file_name"]] = list()
    for box in raw['annotations']:
        img = box['image_id'] + '.jpg'
        result[img].append([box["bbox"][0], box["bbox"][1], round(box["bbox"][0]+box["bbox"][2], 2), round(box["bbox"][1]+box["bbox"][3], 2), 1.00 ,cats[str(box["category_id"])]])
    print("Converted {} image annotations.".format(len(result)))
    return result


def main():
    '''
    :params: /path/to/image/list /path/to/input/json /path/to/save/dir/ /path/to/catrgory/file /path/to/image/prefix [optinal]/path/to/alias/file
    '''
    cats = load_categories(sys.argv[4]) 
    class_names = cats
    if len(sys.argv) == 7:
        _alias = load_categories(sys.argv[6])
        class_names = _alias 
        assert len(_alias) == len(cats), 'Number of alias must match categories.'
        alias = dict()
        for idx,cat in enumerate(cats):
            alias[cat] = _alias[idx]
    else:
        alias = None
    with open(sys.argv[1], 'r') as f: 
        img_lst = [x.strip() for x in f.readlines()]
    with open(sys.argv[2], 'r') as f:
        result = json.load(f)
    if GT:
        result = convert_coco_anns(result)
    pixel_means = [0,0,0]
    for index,img in enumerate(img_lst[:]):
        if img not in result:
            print('Detections of {} not found in json file.'.format(img))
            continue
        else:
            dets = result[img]  
            if alias:
                for det in dets:
                    det[-1] = alias[det[-1]]
            save_path = os.path.join(sys.argv[3], 'DET_'+img)
            box_viz(os.path.join(sys.argv[5],img), dets, pixel_means, class_names, threshold=THRESHOLD, save_path=save_path, transform=True, dpi=80, coor_scale=1.0) 
            print('Image[{}] {} saved'.format(index+1, save_path))

if __name__ == '__main__':
    main()
    print('...Done')
