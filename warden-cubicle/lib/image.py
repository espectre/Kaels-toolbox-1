from __future__ import print_function
import os
import re
import struct
import magic
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UnknownImageFormat(Exception):
    pass


def get_image_size_magic(img_path):
    '''
    get image width and height without loading image file into memory
    '''
    temp = magic.from_file(img_path)
    try:
        width, height = re.findall('(\d+)x(\d+)', temp)[-1]
    except:
        print('get image size failed:',os.path.basename(img_path))
        return None, None
    return int(width), int(height)


def get_image_size_core(img_path):
    """
    Return (width, height) for a given img file content - no external dependencies except the os and struct modules from core
    """
    size = os.path.getsize(img_path)

    with open(img_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)

        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)

        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

def check_bounding_box(bbox,width,height,img_id,digits=2,rel_coor=False,restrict=False,log_error=False):
    '''
    check whether bounding-box coordinates are valid
    bounding box format: xywh
    default approximate digits: .2f
    '''
    x,y,w,h = bbox
    check = 0
    if not restrict:
        if rel_coor :
            if x<0 or y<0 or (x+w)>1 or (y+h)>1:
                check = 1
        else:
            if x<0 or y<0 or (x+w)>(width+pow(0.1,digits)) or (y+h)>(height+pow(0.1,digits)):
                check = 1
        if check==1 and log_error: 
            print('warning: encounterd invalid box {}, in image {}, w{}, h{}'.format(bbox, img_id, width, height))
        return check

    elif restrict:
        if x<0:
            x=0
            check = 1
        if y<0:
            y=0
            check = 1
        if rel_coor:
            if (x+w)>1:
                w=1-x
                check = 1
            if (y+h)>1:
                h=1-y
                check = 1
        else:
            if (x+w)>(width+pow(0.1,digits)):
                w=width-x
                check = 1
            if (y+h)>(height+pow(0.1,digits)):
                h=height-y
                check = 1
        if check==1 and log_error: 
            print('warning: encounterd invalid box {}, in image {}, w{}, h{}'.format(bbox, img_id, width, height))
        return x,y,w,h,check 

def box_viz(img_path, dets, pixel_means, class_names, threshold=0.5, save_path='./tmp.png', transform=True, dpi=80, coor_scale=1.0, line_width=3.0):
    img = cv2.imread(img_path)
    img = img[...,::-1]

    # if transform:
    #     im = transform_im(im, np.array(pixel_means)[[2, 1, 0]])

    # Create a canvas the same size of the image
    height, width, _ = img.shape
    out_size = width/float(dpi), height/float(dpi)
    fig = plt.figure(figsize=out_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Display the image

    ax.imshow(img, interpolation='nearest')

    # Display Detections
    color_map = [(random.random(), random.random(), random.random()) for x in range(len(class_names))] 
    for det in dets:
        bbox = [x * coor_scale for x in det[:4]]
        score = det[-2] 
        cls = det[-1]
        color = color_map[class_names.index(cls)]
        if score < threshold:
            continue
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=line_width)
        ax.add_patch(rect)
        ax.text(bbox[0]+4, bbox[1]-8 if bbox[1]>15 else bbox[1]+15, '{:s} {:.2f}'.format(cls, score), bbox=dict(facecolor=color, alpha=0.5), fontsize=10, color='white')

    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)
    fig.savefig(save_path, dpi=dpi, transparent=True)
    plt.cla()
    plt.clf()
    plt.close()

