from __future__ import print_function
import argparse
import os
import cv2
import json
import mxnet as mx
import numpy as np
from rcnn.config import config
import rcnn.symbol
from rcnn.io.image import resize, transform
from rcnn.core.tester import Predictor, im_detect, im_proposal, vis_all_detection, draw_all_detection
from rcnn.utils.load_model import load_param
from rcnn.processing.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

config.TEST.HAS_RPN = True
SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]
PIXEL_MEANS = config.PIXEL_MEANS
DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None
# visualization
CONF_THRESH = 0.7
NMS_THRESH = 0.3
nms = py_nms_wrapper(NMS_THRESH)


def get_net(symbol, prefix, epoch, ctx):
    arg_params, aux_params = load_param(
        prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(DATA_SHAPES)
    print('DATA_SHAPES:',DATA_SHAPES)
    arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    aux_shape_dict = dict(zip(aux_names, aux_shape))

    # check shapes
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + \
            str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + \
            str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    predictor = Predictor(symbol, DATA_NAMES, LABEL_NAMES, context=ctx,
                          provide_data=DATA_SHAPES, provide_label=LABEL_SHAPES,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor


def generate_batch(im):
    """
    preprocess image, return batch
    :param im: cv2.imread returns [height, width, channel] in BGR
    :return:
    data_batch: MXNet input batch
    data_names: names in data_batch
    im_scale: float number
    """
    im_array, im_scale = resize(im, SHORT_SIDE, LONG_SIDE)
    im_array = transform(im_array, PIXEL_MEANS)
    im_info = np.array(
        [[im_array.shape[2], im_array.shape[3], im_scale]], dtype=np.float32)
    data = [mx.nd.array(im_array), mx.nd.array(im_info)]
    data_shapes = [('data', im_array.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(
        data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch, DATA_NAMES, im_scale


def demo_net(predictor, image_name, vis=False, save_dir='./', save_name='tmp.jpg', threshold=0.7):
    """
    generate data_batch -> im_detect -> post process
    :param predictor: Predictor
    :param image_name: image name
    :param vis: will save as a new image if not visualized
    :return: None
    """
    assert os.path.exists(image_name), image_name + ' not found'
    result_lst = list()
    try:
        im = cv2.imread(image_name)
        data_batch, data_names, im_scale = generate_batch(im)
        scores, boxes, data_dict = im_detect(
            predictor, data_batch, data_names, im_scale)

        all_boxes = [[] for _ in CLASSES]
        for cls in CLASSES:
            cls_ind = CLASSES.index(cls)
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind, np.newaxis]
            keep = np.where(cls_scores >= threshold)[0]
            dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
            keep = nms(dets)
            all_boxes[cls_ind] = dets[keep, :]

        boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]

    except:
        print('detection error')
        return None

    # print results
    print('class ---- [[x1, y1, x2, y2, confidence]]')
    for ind, boxes in enumerate(boxes_this_image):
        if len(boxes) > 0:
            print('---------', CLASSES[ind], '---------')
            print(boxes)
            for box in boxes:
                tmp_box = [round(x, 6) for x in box.tolist()[:]]
                tmp_box.append(str(CLASSES[ind]))
                result_lst.append(tmp_box)
    if vis:
        vis_all_detection(data_dict['data'].asnumpy(),
                          boxes_this_image, CLASSES, im_scale)
    else:
        # result_dir = os.path.dirname(image_name)
        # result_file = save_dir + os.path.basename(image_name)
        result_file = save_dir + save_name
        print('results saved to %s' % result_file)
        im = draw_all_detection(
            data_dict['data'].asnumpy(), boxes_this_image, CLASSES, im_scale)
        if not os.path.exists(os.path.dirname(result_file)):
            os.system('mkdir -p '+os.path.dirname(result_file))
        cv2.imwrite(result_file, im)

    return result_lst


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demonstrate a Faster R-CNN network')
    parser.add_argument('--network', help='custom network architecture', default='resnet', type=str)
    parser.add_argument('--labellist', help='custom label list', type=str)
    parser.add_argument('--image', help='custom image', type=str)
    parser.add_argument('--imagelist', help='custom image list', type=str)
    parser.add_argument('--imageprefix', help='custom image prefix', type=str)
    parser.add_argument('--savedir', help='custom saving directory', type=str)
    parser.add_argument('--prefix', help='saved model prefix', type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model', type=int)
    parser.add_argument('--longside', help='max long side of resized image', default=1500, type=int)
    parser.add_argument('--shortside', help='short side of resized image', default=800, type=int)
    parser.add_argument('--threshold', help='score threshold', default=CONF_THRESH, type=float)
    parser.add_argument('--gpu', help='GPU device to use', default=0, type=int)
    parser.add_argument('--vis', help='display result', action='store_true')
    parser.add_argument('--test', help='single image test mode', action='store_true')
    parser.add_argument('--save_json', help='path to result json file', default='./tmp.json', type=str)
    args = parser.parse_args()
    return args


def main(args):
    global CLASSES, DATA_SHAPES, SHORT_SIDE, LONG_SIDE
    SHORT_SIDE, LONG_SIDE = args.shortside, args.longside
    DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
    # args = parse_args()
    ctx = mx.gpu(args.gpu)
    CLASSES = ['__background__']
    with open(args.labellist, 'r') as label_list:
        for label in label_list:
            CLASSES.append(label.strip())
    SHORT_SIDE, LONG_SIDE = args.shortside, args.longside
    DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
    # print(CLASSES)
    # symbol = eval('rcnn.symbol.get_' + args.network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    symbol = eval('rcnn.symbol.get_' + args.network + '_test')(num_classes=len(CLASSES), num_anchors=config.NUM_ANCHORS)
    # symbol.save('demo.py.json')
    # symbol = rcnn.symbol.get_resnet_test(
    #     num_classes=len(CLASSES), num_anchors=config.NUM_ANCHORS)
    predictor = get_net(symbol, args.prefix, args.epoch, ctx)
    lst_img = open(args.imagelist, 'r')
    result_dic = dict()
    for img in lst_img:
        print('processing: ' + img.strip())
        tmp = demo_net(predictor, args.imageprefix +
                 img.strip(), args.vis, save_dir=args.savedir, save_name=img.strip(), threshold=args.threshold)
        # if tmp:
        result_dic[img.strip()] = tmp
    json.dump(result_dic, open(args.save_json, 'w'),indent=2)
    print('result json file saved to: {}'.format(args.save_json))

def test(args):
    global CLASSES, DATA_SHAPES, SHORT_SIDE, LONG_SIDE
    # args = parse_args()
    ctx = mx.gpu(args.gpu)
    CLASSES = ['__background__']
    SHORT_SIDE, LONG_SIDE = args.shortside, args.longside
    DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
    with open(args.labellist, 'r') as label_list:
        for label in label_list:
            CLASSES.append(label.strip())
    # print(CLASSES)
    symbol = eval('rcnn.symbol.get_' + args.network + '_test')(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    # symbol = rcnn.symbol.get_resnet_test(
    #     num_classes=len(CLASSES), num_anchors=config.NUM_ANCHORS)
    predictor = get_net(symbol, args.prefix, args.epoch, ctx)
    demo_net(predictor, args.image, args.vis, save_dir=args.savedir, save_name='objdet_'+os.path.basename(args.image), threshold=args.threshold)

if __name__ == '__main__':
    args = parse_args()
    if args.test:
        print('testing single image...')
        test(args)
    else:
        print('start testing...')
        main(args)
