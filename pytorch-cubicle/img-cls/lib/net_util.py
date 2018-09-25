import torchvision
import torch.nn
import model as extra_models
from io_util import load_checkpoint 
from config import cfg


def _weight_xv_init(m):
    '''
    Usage:
        model = Model()
        model.apply(_weight_xv_init)
    '''
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)


def get_avail_models():
    '''
    '''
    available_models = torchvision.models

    for name in extra_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(extra_models.__dict__[name]):
            available_models.__dict__[name] = extra_models.__dict__[name]

    available_models_names = sorted(
            name for name in available_models.__dict__
            if name.islower() and not name.startswith("__")
            and callable(available_models.__dict__[name]))

    return available_models, available_models_names 


def init_model(model_lib):
    '''
    '''
    # create network
    model = eval("model_lib.{}()".format(cfg.TRAIN.NETWORK))

    # init weights
    if cfg.TRAIN.XAVIER_INIT:
        model.apply(_weight_xv_init)

    # load pretrained weight
    if cfg.TRAIN.FINETUNE:
        model_file = cfg.TRAIN.FT.PRETRAINED_MODEL_WEIGHTS
        model_weight = load_checkpoint(model_file, is_tar=model_file.endswith('.tar'))
        model.load_state_dict(model_weight)
    elif cfg.TRAIN.RESUME:
        pass    # TODO 
    elif cfg.TRAIN.SCRATCH:
        pass

    # change output dim
    num_filters = model.fc.in_features
    model.fc = torch.nn.Linear(num_filters, cfg.TRAIN.NUM_CLASSES)

    # parallelization
    if cfg.TRAIN.USE_GPU:
        model = torch.nn.DataParallel(model).cuda()
    return model
    

