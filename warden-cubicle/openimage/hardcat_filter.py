#!/usr/bin/env python 
from __future__ import print_function
import os
import sys
import json


FILTER_BOX = False    # set False to filter 51 cls 

def cat_mask(hardcat_file):
    with open(hardcat_file,'r') as f:
        mask = [x.strip() for x in f.readlines()]
    print('mask list length:', len(mask))
    return mask


def img_mask(imagelist_file):
    with open(imagelist_file,'r') as f:
        mask = [x.strip() for x in f.readlines()[1:]]
    print('mask list length:', len(mask))
    return mask


def cat_map(all_cats, hardcat_mask, other_cat='aggregate_450'):
    cmap = dict()
    for cat in all_cats:
        if cat not in hardcat_mask:
            cmap[cat] = other_cat
        else:
            cmap[cat] = cat
    return cmap 


def filter_box(raw_anns, mask, result_file, filter_box=True, cat_map=None):
    raw, result = 0, 0
    with open(result_file,'w') as f:
        f.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n')
        if filter_box:  # filter by box mask
            for box in raw_anns:
                raw+=1
                if box.split(',')[2] in mask:
                    f.write('{}\n'.format(box)) 
                    result+=1
                if raw%1000 == 0:
                    print('=> {} boxes processed...'.format(raw))
        else:  # filter by image mask
            for box in raw_anns:
                raw+=1
                if box.split(',')[0] in mask:
                    tmp = box.split(',')
                    tmp[2] = cat_map[tmp[2]]
                    f.write('{}\n'.format(','.join(tmp)))
                    result+=1
                if raw%1000 == 0:
                    print('=> {} boxes processed...'.format(raw))
    print('Filtered {} boxes from {}'.format(result, raw))
        

def main():
    '''
    :params: /path/to/raw/ann/csv /path/to/hard/cat/list /path/to/result/csv | /path/to/all/cat/list /path/to/image/list
    '''
    mask = cat_mask(sys.argv[2]) 
    if not FILTER_BOX:
        all_cats = cat_mask(sys.argv[4])
        img_lst = img_mask(sys.argv[5])
        cmap = cat_map(all_cats, mask) 
    with open(sys.argv[1],'r') as f:
        raw_anns = [x.strip() for x in f.readlines()[1:]]
    if FILTER_BOX :
        filter_box(raw_anns, mask, sys.argv[3])
    else:
        filter_box(raw_anns, img_lst, sys.argv[3], filter_box=False, cat_map=cmap)


if __name__ == '__main__':
    main()

