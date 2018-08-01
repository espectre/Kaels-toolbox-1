#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/07/31 
# by Northrend#github.com
#
# Call atserving restful api  
# Delux tool
#

from __future__ import print_function
import os
import sys
import re
import json
import time
import requests
import docopt
import pprint
import ConfigParser
from qlib import qiniu_auth, qiniu_mac_auth, post_request


def _init_():
    """
    Call atserving restful api  
    Supported api: qpulp
    Update: 2018/08/01
    Contributor: 

    Change log:
    2018/08/01      v1.1                support minimum customized params of
                                        video api: pulp/terror/politician 
                                        minimum params
    2018/07/31      v1.0                basic functions

    Usage:
        atserving_delux.py              [-s|--single-mode]
                                        [--cfg=str --url=str] 
        atserving_delux.py              -v|--version
        atserving_delux.py              -h|--help

    Arguments:

    Options:
        -h --help                       show this screen
        -v --version                    show script version
        -s --single-mode                set to post just one single resource
        ------------------------------------------------------------------------------------
        --cfg=str                       configuration file [default: ./atserving_delux.conf]
        --url=str                       
    """
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print ('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)


def _init_auth(conf, auth_type='mac'):
    ak = conf.get('keys', 'ak')
    sk = conf.get('keys', 'sk')
    auth = qiniu_mac_auth(ak, sk) if auth_type == 'mac' else qiniu_auth(ak,sk)
    return auth 


def post(auth, conf, data_url):
    resp = post_request(auth, conf, data_url)
    # print(resp.status_code)
    # print(resp.content)
    try:
        resp_text = json.loads(resp.text)
    except:
        print('ERROR: response error\n  |-status code: {}\n  |-raw content: {}'.format(resp.status_code, resp.content))
        return None
    if resp.status_code == 200:
        return resp_text 
    else:
        print('ERROR: response error\n  |-status code: {}\n  |-error message: {}'.format(resp.status_code, resp_text['error']))
        return None


def main():
    conf = ConfigParser.ConfigParser()
    conf.read(args['--cfg'])
    print('=> ...Configuration file loaded')
    auth = _init_auth(conf) 
    print('=> Host: {}\n   Post: {}\n   Data uri: {}'.format(conf.get('params','host'), conf.get('params','query'), args['--url']))
    print('=> Posting...')
    tic = time.time()
    if args['--single-mode']:
        tmp = post(auth, conf, args['--url']) 
    print('=> Response time: {:.3f}s'.format(time.time()-tic))
    print('=> Result:')
    if tmp:
        pprint.pprint(tmp)


if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(
        _init_.__doc__, version='atserving delux tool {}'.format(version))
    _init_()
    print('=> Start processing...')
    main()
    print('=> ...Done')
