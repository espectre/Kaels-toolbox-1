# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import os
import time
import cv2
import mxnet as mx
import pprint
from collections import namedtuple
import numpy as np
import docopt


def net_init(model_prefix,model_epoch,gpu=0,feature_layers=['flatten0_output'],batch_size=1,image_width=224):
    '''
    initialize mxnet model
    '''
    # get compute graph
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)   # load original model
    output_layers = mx.sym.Group([sym.get_internals()[x] for x in feature_layers])
    # bind module with graph
    model = mx.mod.Module(symbol=output_layers, context=mx.gpu(gpu), label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, image_width, image_width))], label_shapes=model._label_shapes)

    # load model parameters
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    return model


def extra_feature(model, image_path):
    resize_width = 224
    mean_r, mean_g, mean_b = 123.68, 116.779, 103.939
    std_r, std_g, std_b = 58.395, 57.12, 57.375
    Batch = namedtuple('Batch', ['data'])
    img_read = cv2.imread(image_path)
    if np.shape(img_read) == tuple():
        return None
    img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = cv2.resize(img, (resize_width, resize_width))
    img -= [mean_r, mean_g, mean_b]
    img /= [std_r, std_g, std_b]
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img_batch = mx.nd.array(np.zeros((1,3,224,224)))
    img_batch[0] = mx.nd.array(img)
    
    model.forward(Batch([img_batch]))
    # output = model.get_outputs()[0].asnumpy()
    _output = [x.asnumpy() for x in model.get_outputs()]
    output = list()
    for op in _output:
        assert len(op.shape) > 1, 'output shape error!'
        # print(op.shape)
        if len(op.shape) > 2:
            output.append(np.mean(op,axis=(2,3)))
        else:
            output.append(op)
    # print('len(output):',len(output))
    # print('output.shape:',output[0].shape,output[1].shape)
    return output

def main():
    '''
    :inputs: /path/to/model/prefix epoch /path/to/images.lst /path/to/prefix/of/result.npy /path/to/image/prefix
    '''
    # root_path = './test-images/'
    root_path = sys.argv[5] 
    with open(sys.argv[3],'r') as f:
        images = [os.path.join(root_path,x.strip()) for x in f.readlines()]
    image_number = len(images)
    # feature_dim = 3
    # feature_dim_1 = [64,112,112]
    # feature_dim_2 = [256,56,56]
    # feature_dim_3 = [512,28,28]
    # feature_dim_4 = [1024]    # 1024,14,14 
    # feature_dim_5 = [2048]    # 2048,7,7
    feature_dim_6 = [2048]
    feature_dims = [[image_number]]
    feature_dims[0].extend(feature_dim_6)
    # feature_dims = [[image_number], [image_number]]
    # feature_dims[0].extend(feature_dim_4)
    # feature_dims[1].extend(feature_dim_5)
    features = [np.zeros([x for x in y]) for y in feature_dims]
    # feature_layers = ['_plus12_output','flatten0_output']
    feature_layers = ['flatten0_output']
    # model = net_init(sys.argv[1],int(sys.argv[2]),gpu=7)
    model = net_init(sys.argv[1],int(sys.argv[2]),gpu=7,feature_layers=feature_layers)
    # model = net_init(sys.argv[1],int(sys.argv[2]),gpu=7,feature_layers=['fc-3_output'])
    tic_0 = time.time()
    for i in xrange(image_number):
        tic = time.time()
        output = extra_feature(model, images[i])
        print('Batch [{}]: {:.4f}s'.format(i, time.time()-tic))
        for j in xrange(len(features)):
            if np.shape(output[j]) != tuple():
                features[j][i] = output[j]
    print('Total time: {:.4f}s'.format(time.time()-tic_0))
    for idx,feat in enumerate(features): 
        try:
            np.save(sys.argv[4]+'_{}.npy'.format(feature_layers[idx]), feat)
        except:
            np.save('./tmp_{}.npy'.format(feature_layers[idx]), feat)
            print('Saving failed, result file saved to ./tmp.npy')
    print('...done')
    # print('==> Original params:')
    # pprint.pprint(zip(sym.list_arguments(), sym.infer_shape(data=(1, 3, 224, 224))[0]))
    # sym_new, arg_new = modify_net(sym, arg_params)
    # mx.model.save_checkpoint(sys.argv[1] + '-modified', 0, sym_new, arg_new, aux_params)
    # print('New model saved at: {}'.format(sys.argv[1] + '-modified'))


if __name__ == '__main__':
    main()
