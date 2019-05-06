#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/12/14 @Northrend
# 
# Draw PR curve and F1 curve 
# Input CSV syntax:
# 
# ImageID,PredLabel,PredScore,GTLabel
# image_name_0.jpg,0,0.999988,1
# image_name_1.jpg,1,0.999977,1
# ...
#

from __future__ import print_function
import os,sys
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot


class SyntaxErr(RuntimeError):
    def __init__(self, msg):
        self.msg = msg
        print(self.msg)


def is_positive(pred_score, threshold):
    if pred_score >= threshold:
        return True
    else:
        return False


def load_input_file(input_path):
    output_dic = dict()
    gt_labels = list()
    with open(input_path, 'r') as f:
        for idx,line in enumerate(f.readlines()):
            _split = line.strip().split(',')
            if len(_split) != 4:
                raise SyntaxErr('[SYNTAX ERR] line {}: {}'.format(idx+1, line))
            elif line.strip() == "ImageID,PredLabel,PredScore,GTLabel":
                continue
            try:
                output_dic[_split[0]] = dict()
                output_dic[_split[0]]['pred_label'] = int(_split[1])
                output_dic[_split[0]]['pred_score'] = round(float(_split[2]), 6)
                output_dic[_split[0]]['gt_label'] = int(_split[3])
                gt_labels.append(output_dic[_split[0]]['gt_label'])
            except:
                raise SyntaxErr('[SYNTAX ERR] line {}: {}'.format(idx+1, line))
    print('==> {} logs loaded.'.format(len(output_dic)))
    return output_dic, list(set(gt_labels))


def draw_2d_curve(lst_x, lst_y, save_path='./tmp.png', style='--r', xlabel='x', ylabel='y'):
    '''
    draw curves
    '''
    # pyplot.axis([0, 1, 0, 1])
    pyplot.plot(lst_x, lst_y, style, lw=1.5)
    pyplot.grid(True)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title('{}-{} Curve'.format(ylabel, xlabel))
    pyplot.savefig(save_path)
    pyplot.close()      # release cache, or next picture will get fucked


def calculate_pr_curve(dict_log, positive_label):
    '''
    calculate list of precision and recall values, for one given class
    '''
    lst_precision, lst_recall, lst_f1, lst_threshold = list(), list(), list(), list()
    for i in xrange(1, 100):
        threshold = float(i) / 100    # not 0.01*i
        tp, fp, fn = 0, 0, 0
        lst_threshold.append(threshold)
        for image in dict_log:
            if dict_log[image]['pred_label'] == positive_label == dict_log[image]['gt_label'] and is_positive(dict_log[image]['pred_score'], threshold):
                tp += 1
            elif dict_log[image]['pred_label'] == positive_label != dict_log[image]['gt_label'] and is_positive(dict_log[image]['pred_score'], threshold): 
                fp += 1
            elif (dict_log[image]['gt_label'] == positive_label == dict_log[image]['pred_label'] and not is_positive(dict_log[image]['pred_score'], threshold)) or (dict_log[image]['gt_label'] == positive_label != dict_log[image]['pred_label']):
                fn += 1
        if (tp + fp) == 0 or (tp + fn) == 0:    # eps
            fp, fn = float(1e-8), float(1e-8)
        lst_precision.append(float(tp) / (tp + fp))
        lst_recall.append(float(tp) / (tp + fn))
        lst_f1.append(float(2 * tp) / (2 * tp + fp + fn))
        # print(tp,fp,fn)
    return lst_precision, lst_recall, lst_f1, lst_threshold


def main():
    '''
    :params: /path/to/input.csv /path/to/folder/to/save/png/
    '''
    dummy, positive_labels = load_input_file(sys.argv[1])
    result_root_path = sys.argv[2]
    for positive_label in positive_labels:
        print('==> Processing label {} ...'.format(positive_label))
        # calculating pr and f1 list
        lst_precision, lst_recall, lst_f1, lst_threshold = calculate_pr_curve(dummy, positive_label)
        print('==> Drawing...')
        # set saving path
        pr_curve, f1_curve = os.path.join(result_root_path, 'pr-{}.png'.format(positive_label)), os.path.join(result_root_path, 'f1-{}.png'.format(positive_label))
        # save pr curve
        draw_2d_curve(lst_recall, lst_precision, save_path=pr_curve, xlabel='Recall', ylabel='Precision')
        print('==> PR curve saved as:', pr_curve)
        # save f1 curve
        draw_2d_curve(lst_threshold, lst_f1, save_path=f1_curve, xlabel='Threshold', ylabel='F1score')
        print('==> F1 curve saved as:', f1_curve)

if __name__ == '__main__':
    main()
    print('==> done.')
