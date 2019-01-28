#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Convert dummy dataset of coco style
# to oiv4 style csv:
# coco style with absolute coordinates
# ->
# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside


from __future__ import print_function
import sys, os, json, pprint
import cv2
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path,'../lib'))
from coco import load_categories,load_annotations
from image import get_image_size_core,check_bounding_box

def main():
    '''
    params: /path/to/input/json /path/to/output/csv
    '''
    raw = load_annotations(sys.argv[1]) 
    with open(sys.argv[2], 'w') as f:
        f.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside')
        for box in raw:
            f.write('{},{},{},{},{},{},{},{},0,0,0,0,0\n'.format(box['ImageID'],box['Source'],box['LabelName'],box['Confidence'],box['XMin'],box['XMax'],box['YMin'],box['YMax']))


if __name__ == '__main__':
    print('=> Start converting...')
    main()
    print('=> ...Done')
