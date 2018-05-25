#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import logging
import mxnet as mx


def general_finetune_model(symbol, arg_params, num_classes, layer_name='flatten0', use_svm=None, reg_coeff=None, gluon_style=False, ctx=None):
    '''
    define the function which replaces the the last fully-connected layer for a given network
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    '''

    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(
        data=net, num_hidden=num_classes, name='fc-' + str(num_classes))
    if use_svm:
        assert reg_coeff is not None, "Regularization coefficient is needed for svm classifier"
        net = mx.symbol.SVMOutput(data=net, name='svm', regularization_coefficient=reg_coeff) if use_svm == 'l2' else mx.symbol.SVMOutput(data=net, name='svm', use_linear=1, regularization_coefficient=reg_coeff)
    else:
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc' not in k})
    return net, new_args


def gluon_finetune_model(): 
    '''
    '''
    def _init_gluon_style_params(raw_params, net_params, ctx):
        for param in raw_params:
            if param in net_params:
                net_params[param]._load_init(raw_params[param], ctx=ctx)
        return net_params

    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net_hybrid = mx.gluon.nn.SymbolBlock(outputs=net, inputs=mx.sym.var('data'))
    net_params = net_hybrid.collect_params()
    net_params = _init_gluon_style_params(arg_params,net_params,ctx)
    net_params = _init_gluon_style_params(aux_params,net_params,ctx)
    return net_hybrid


def gluon_init_classifer(num_class, ctx):
    net = mx.gluon.nn.Sequential()
    with net.name_scope():
        net.add(mx.gluon.nn.Dense(num_class))
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2), ctx=ctx)
    return net
    
