#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/03/13 @Northrend
#
# Generate AVA-standard jsonlist
#


from __future__ import print_function
import os
import sys
import json
import re
import docopt


def _init_():
    '''
    Script for generating AVA-standard jsonlist file
    Update: 2018/08/22
    Author: @Northrend
    Contributor: 

    Change log:
    2018/08/22   v1.3        some fit stuff 
    2018/08/13   v1.2        support detection pre-json 
    2018/05/03   v1.1        support detection 
    2018/03/13   v1.0        basic functions

    Usage:
        gen_ava_jsonlist.py          <in-file> <out-list> 
                                     [ -c | --classification]
                                     [ -d | --detection ]
                                     [ -l | --clustering ]
                                     [--prefix=str --sub-task=str --pre-json=str --pre-label=str --url-map=str]
        gen_ava_jsonlist.py          -v | --version
        gen_ava_jsonlist.py          -h | --help

    Arguments:
        <in-file>                       input list path
        <out-list>                      output jsonlit file path

    Options:
        -h --help                       show this help screen
        -v --version                    show current version
        -c --classification             classification task mode
        -d --detection                  detection task mode
        -l --clustering                 clustering task mode
        -------------------------------------------------------------------------------------------
        --prefix=str                    prefix of each url, such as bucket-domain.
        --sub-task=str                  sub-task type such as general, pulp, terror, places.
        --pre-json=str                  optional pre-annotation json, required under clusering task.
        --pre-label=str                 optional pre-annotation label, such as "cat".
        --url-map=str                   optional file2url json map.
    '''
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


class input_syntax_err(Exception):
    '''
    Catch input file-list syntax error
    '''
    pass


def read_labels(input_file):
    dct = dict()
    with open(input_file) as f:
        for buff in f.readlines():
            img = buff.strip().split()[0]
            dct[img] = int(buff.strip().split()[1])
    print('{} pre label loaded.'.format(len(dct)))
    return dct


def generate_dict(filename, prefix, classification=False, detection=False, clustering=False, sub_task=None, pre_ann=None, pre_label=None, url_map=None):
    temp = dict()
    if url_map:
        if filename not in url_map:
            print(filename)
            if filename.startswith("Image-tupu-"):
                temp['url'] = 'http://oi7xsjr83.bkt.clouddn.com/'+filename
            elif filename.startswith("pulp-"):
                temp['url'] = 'http://oquqvdmso.bkt.clouddn.com/atflow-log-proxy/images/'+filename
            elif filename.startswith("screenshot_mobile_"):
                temp['url'] = 'http://p28cyzi19.bkt.clouddn.com/screenshot_mobile/'+filename
            elif filename.startswith("screenshot_pc_"):
                temp['url'] = 'http://p28cyzi19.bkt.clouddn.com/screenshot_pc/'+filename
            elif filename.startswith("blademaster_1228"):
                temp['url'] = 'http://pbzod2s8y.bkt.clouddn.com/blademaster_1228/Image/'+filename
            else:
                assert 0
        else:
            temp['url'] = url_map[filename]
    else: 
        temp['url'] = os.path.join(prefix,filename) if prefix else filename
    # temp['ops'] = 'download()'
    # temp['source_url'] = temp['url']
    temp['type'] = 'image'
    temp['label'] = list() 
    pulp_label = ['pulp', 'sexy', 'normal']
    custom_cats = ['generic_porn', 'generic_obscene', 'int_sex_con', 'alm_naked', 'close_up', 'flirt_sex_con', 'baby_gen', 'sex_toy', 'generic_sexy', 'ani_sexy', 'sli_hot', 'generic_normal'] 
    if classification:
        tmp = dict()
        tmp['type'] = 'classification'
        tmp['version'] = '1'
        tmp['name'] = sub_task
        tmp['data'] = list()
        if pre_ann:
            # ---- modify pre-annotated label here ----
            # tmp['data'].append({'class': custom_cats[pre_ann[filename]]})
            pass
            # -----------------------------------------
        elif pre_label:
            # tmp['data'].append({'class': pulp_label[pre_label]})
            tmp['data'].append({'class': custom_cats[int(pre_label)]})
        temp['label'].append(tmp)
    if detection:
        tmp = dict()
        tmp['type'] = 'detection'
        tmp['version'] = '1'
        tmp['name'] = sub_task
        tmp['data'] = list()
        if pre_ann:
            # ---- modify pre-annotated label here ----
            for box in pre_ann[filename]:
                x1,y1,x2,y2,score,cls = box[:]
                tmp_box = dict()
                tmp_box['bbox'] = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                tmp_box['class'] = cls
                tmp['data'].append(tmp_box)
            # -----------------------------------------
        temp['label'].append(tmp)
    # if clustering:
    #     assert pre_ann, 'pre-annotation file should be provided under clustering task.'
    #     # ---- modify pre-annotated label here ----
    #     temp['label']['facecluster'] = pre_ann[filename]
    #     # -----------------------------------------
    
    return temp


def main():
    '''
    Generate labelX standard json.
    '''
    sub_task = args['--sub-task'] if args['--sub-task'] else 'general'
    pre_ann = json.load(open(args['--pre-json'], 'r')
                        ) if args['--pre-json'] else None
    pre_label = args['--pre-label'] if args['--pre-label'] else None
    url_map = json.load(open(args['--url-map'], 'r')) if args['--url-map'] else None
    with open(args['<in-file>'], 'r') as f:         # load input file list
        file_lst = list()
        for buff in f:
            if len(buff.strip().split()) == 1:      # input syntax error
                file_lst.append(buff.strip())
            elif len(buff.strip().split()) == 2:
                file_lst.append(buff.strip())
            else:
                raise input_syntax_err
            
    with open(args['<out-list>'], 'w') as f:
        # for image,label in file_lst:
        for image in file_lst:
            if len(image.strip().split()) == 2:
                pre_label = image.strip().split()[1]
            temp_dict = generate_dict(image.split()[0], args['--prefix'], args['--classification'], args['--detection'], args['--clustering'], sub_task, pre_ann, pre_label=pre_label, url_map=url_map)
            f.write('{}\n'.format(json.dumps(temp_dict)))


if __name__ == "__main__":
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='LabelX jsonlist generator {}'.format(version))
    _init_()
    print('Start generating jsonlist...')
    main()
    print('...done')
