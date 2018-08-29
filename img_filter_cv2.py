from __future__ import print_function
import os
import sys
import multiprocessing 
import time
import cv2
import numpy as np


class ImgErr(RuntimeError):
    def __init__(self, msg):
        self.msg = msg

SILENT_MODE = True  # set to NOT print error image path
CLEAN_MODE = True  # DANGEROUS, set to delete error images simultaneously
invalid_list = multiprocessing.Manager().list()


def sub_proc(img_path):
    try:
        img_read = cv2.imread(img_path)
        if np.shape(img_read) == tuple():
            raise ImgErr('cv2 load error')
    except:
        if not SILENT_MODE:
            print('=> Image error: {}'.format(img_path))
        if CLEAN_MODE:
            os.remove(img_path)
            if not SILENT_MODE:
                print('=> Delete image: {}'.format(img_path))
        invalid_list.append(img_path)
    return 0


def main():
    '''
    :params: <num-proc> /path/to/image/list/file [/path/to/image/prefix, optinal]
    '''
    with open(sys.argv[2],'r') as f:
        if len(sys.argv) == 3:
            img_path_list = [x.strip() for x in f.readlines()]
        elif len(sys.argv) == 4:
            img_path_list = [os.path.join(sys.argv[3],x.strip()) for x in f.readlines()]
        else:
            print('Invalid args')
            return 0
    pool = multiprocessing.Pool(processes=int(sys.argv[1]))
    tic = time.time()
    ret = pool.map(sub_proc, img_path_list)
    with open('./_error_img.lst','w') as f:
        for img in invalid_list:
            f.write(img + '\n')
    print('=> Total image number:',len(img_path_list))
    print('=> Filtered image number:',len(invalid_list))
    print('=> Error image list temporarily saved in ./_error_img.lst')
    print('=> Processing time: {:.6f}s'.format(time.time() - tic))


if __name__ == '__main__':
    main()
