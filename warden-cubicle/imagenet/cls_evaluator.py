#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import json

def main():
    '''
    '''
    with open(sys.argv[1], 'r') as f:
        input_log = json.load(f)
    with open(sys.argv[2], 'r') as f:
        gt = dict()
        for line in f.readlines():
            tup = line.strip().split()
            gt[tup[0]] = int(tup[1])
    top_1, top_5 = 0, 0
    test_num = len(input_log) 
    print('Tested Images:', test_num)
    for key in input_log:
        tops = [int(x) for x in input_log[key]['Top-5 Index']]
        if key not in gt:
            print('GT not found:', key)
            continue
        if gt[key] in tops:
            top_5 +=1
            if gt[key] == tops[0]:
                top_1 +=1
    print('Top-1 acc:', float(top_1)/test_num)
    print('Top-5 acc:', float(top_5)/test_num)


if __name__ == '__main__':
    main()
