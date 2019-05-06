#!/usr/bin/env python
# -*- coding: utf-8 -*-
# created 2018/07/25 
# by Northrend#github.com
#
# Pop out bucket name of specified domain 
#

from __future__ import print_function
import os
import sys
import commands
import re
import cPickle
import docopt


def _init_():
    """
    Pop out bucket name of specified domain 
    Update: 2018/07/25
    Contributor: 

    Change log:
    2018/07/25      v1.0                basic functions

    Usage:
        domain_pop.py                   <domain> [-u|--update] [-c|--clear]
                                        [--login=str] [--qshell-path=str]         
        domain_pop.py                   -v|--version
        domain_pop.py                   -h|--help

    Arguments:
        <domain>                        target domain

    Options:
        -h --help                       show this screen
        -v --version                    show script version
        ------------------------------------------------------------------------------------------
        -c --clear                      set to clear up cache after querying  
        -u --update                     set to update cache 
        --qshell-path=str               path to qshell [default: ./qshell]
        --login=str                     qiniu account ak/sk, split with ","
    """
    print('=' * 80 + '\nArguments submitted:')
    for key in sorted(args.keys()):
        print ('{:<20}= {}'.format(key.replace('--', ''), args[key]))
    print('=' * 80)

    
CACHE = '.bd_map'   # path to save cache file


def check_qshell(qsh_path):
    '''
    Check qshell runing
    '''
    print('=> checking dependencies...')
    if not os.path.exists(qsh_path):
        print('ERROR: {} not found'.format(qsh_path))
        return 1
    qsh_ver = commands.getoutput('{} -v'.format(qsh_path))
    if not qsh_ver.startswith('QShell'):
        print('ERROR: qshell could not run correctly, platform may mismatch ,or just try \"chmod\"')
        return 1
    print('qshell version:', qsh_ver)
    return 0


def log_in(qsh_path, aksk):
    print('=> log in...')
    ak, sk = aksk.split(',')
    commands.getoutput('{} account {} {}'.format(qsh_path, ak, sk))
    return 0


def generate_bd_map(qsh_path):
    print('=> updating cache...')
    bd_map = dict()
    bkt_lst = sorted(commands.getoutput('{} buckets'.format(qsh_path)).strip().split('\n'))
    for bkt in bkt_lst:
        dmn = commands.getoutput('{} domains {}'.format(qsh_path, bkt))
        for line in dmn.strip().split('\n'):
            if '.com' in line:
                bd_map[line] = bkt
    with open(CACHE, 'wb') as f:
        cPickle.dump(bd_map,f)
    return 0 


def main():
    print('WARNING: Only bucket with one single domain could be catched for now')
    qsh_path = args['--qshell-path']
    if check_qshell(qsh_path):
        print('=> exit...')
        return 0
    if args['--login']:
        aksk = args['--login']
        log_in(qsh_path, aksk)
    if args['--login'] or args['--update']:
        generate_bd_map(qsh_path)
    if not os.path.exists(CACHE):
        print('ERROR: no cache found, run -u first')
        print('=> exit...')
        return 0
    with open(CACHE, 'rb') as f:
        bd_map = cPickle.load(f)
    print('=> searching for domain:\n{}'.format(args['<domain>']))
    print('=> bucket name found:\n{}'.format(bd_map[args['<domain>']]))
    if args['--clear']:
        print('=> clearing up cache file...')
        os.remove(CACHE)
    return 0
        
        
if __name__ == '__main__':
    version = re.compile('.*\d+/\d+\s+(v[\d.]+)').findall(_init_.__doc__)[0]
    args = docopt.docopt(_init_.__doc__, version='Domain-pop {}'.format(
        version), argv=None, help=True, options_first=False)
    _init_()
    print('=> start querying...')
    main()
    print('=> ...done')
