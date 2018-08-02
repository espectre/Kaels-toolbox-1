#!/usr/bin/env python 
from __future__ import print_function
import os
import sys
import json

def cat_mask(hardcat_file):
    with open(hardcat_file,'r') as f:
        mask = [x.strip() for x in f.readlines()]
    print('mask list length:', len(mask))
    return mask

def filter(raw_anns, mask, result_file):
    raw, result = 0, 0
    with open(result_file,'w') as f:
        f.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n')
        for box in raw_anns:
            raw+=1
            if box.split(',')[2] in mask:
                f.write('{}\n'.format(box)) 
                result+=1
    print('Filtered {} boxes from {}'.format(result, raw))
        

def main():
    mask = cat_mask(sys.argv[2]) 
    with open(sys.argv[1],'r') as f:
        raw_anns = [x.strip() for x in f.readlines()[1:]]
    filter(raw_anns, mask, sys.argv[3])


if __name__ == '__main__':
    main()

