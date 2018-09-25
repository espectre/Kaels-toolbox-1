import os
import shutil
import logging
import numpy as np
from torchvision import transforms
from torch.utils import data as pth_data
from torch import save as t_save
from torch import load as t_load
from PIL import Image
from config import cfg
from collections import OrderedDict

class DummyDataset(pth_data.dataset.Dataset):
    def __init__(self, lst_path, trans_list, img_path_prefix=str(), delimiter=' '):
        """
        Args:
            lst_path (string): path to the list file with images and labels 
            # transform: pytorch transforms for transforms and tensor conversion
        """
        def _read_lst(lst_path, delimiter):
            data_list = list()
            with open(lst_path,'r') as f:
                for buff in f.readlines():
                    temp_tuple = buff.strip().split(delimiter)
                    temp_tuple[0] = os.path.join(img_path_prefix, temp_tuple[0]) if img_path_prefix else temp_tuple[0]  # add prefix
                    temp_tuple[1] = int(temp_tuple[1])
                    if len(temp_tuple) == 2:    # syntax: image label
                        data_list.append(temp_tuple)
            return data_list

        def _read_csv(csv_path):
            pass
            
        # Transforms
        # self.to_tensor = transforms.ToTensor()
        self.transform = trans_list 
        # Read dataset lst
        # self.data_info = pd.read_csv(csv_path, header=None)
        self.data_info = _read_lst(lst_path, delimiter)
        # First column contains the image paths
        # self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.image_arr = np.asarray([x[0] for x in self.data_info])
        # Second column is the labels
        # self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.label_arr = np.asarray([x[1] for x in self.data_info])
        # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        self.operation_arr = np.asarray([None for x in self.data_info]) # mock
        # Calculate len
        self.data_len = len(self.data_info)
        
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_path)
        if img_as_img.mode != "RGB":
            # print(img_as_img.mode)
            img_as_img = img_as_img.convert("RGB")

        # Check if there is an operation
        some_operation = self.operation_arr[index]
        # If there is an operation
        if some_operation:
            # Do some operation on image
            pass
        # Transform image to tensor
        # img_as_tensor = self.to_tensor(img_as_img)
        img_as_tensor = self.transform(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        # print ('=> single_image_label',single_image_label)
        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len


def compose_transform_list(cfg_obj):
    result = list()
    if cfg_obj.RESIZE:
        _size = (cfg_obj.RESIZE, cfg_obj.RESIZE) if cfg_obj.FORCE_RESIZE else cfg_obj.RESIZE
        result.append(transforms.Resize(
                    size=_size, 
                    interpolation=cfg_obj.R_INTERPOLATION
                    ))
    if cfg_obj.RANDOM_RESIZED_CROP:
        result.append(transforms.RandomResizedCrop(
                    size=cfg_obj.INPUT_SHAPE[1],
                    scale=cfg_obj.RRC_SCALE,
                    ratio=cfg_obj.RRC_RATIO,
                    interpolation=cfg_obj.RRC_INTERPOLATION
                    ))
    if cfg_obj.COLOR_JITTER:
        result.append(transforms.ColorJitter(
                    brightness=cfg_obj.CJ_BRIGHTNESS,
                    contrast=cfg_obj.CJ_CONTRAST,
                    saturation=cfg_obj.CJ_SATURATION,
                    hue=cfg_obj.CJ_HUE
                    ))
    if cfg_obj.RANDOM_AFFINE:
        result.append(transforms.RandomAffine(
                    degrees=cfg_obj.RA_DEGREES,
                    translate=cfg_obj.RA_TRANSLATE,
                    scale=cfg_obj.RA_SCALE,
                    shear=cfg_obj.RA_SHEAR,
                    resample=cfg_obj.RA_RESAMPLE,
                    fillcolor=cfg_obj.RA_FILLCOLOR
                    ))
    if cfg_obj.RANDOM_CROP:
        result.append(transforms.RandomCrop(size=cfg_obj.INPUT_SHAPE[1]))
    if cfg_obj.CENTER_CROP:
        result.append(transforms.CenterCrop)
    if cfg_obj.RANDOM_HFLIP:
        result.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg_obj.RANDOM_VFLIP:
        result.append(transforms.RandomVerticalFlip(p=0.5))
    if cfg_obj.RANDOM_ROTATION:
        result.append(transforms.RandomRotation(
                    degrees=cfg_obj.RR_DEGREES,
                    resample=cfg_obj.RR_RESAMPLE,
                    expand=cfg_obj.RR_EXPAND,
                    center=cfg_obj.RR_CENTER
                    ))

    result.append(transforms.ToTensor())
    result.append(transforms.Normalize(
                mean=[x/256 for x in cfg_obj.MEAN_RGB],
                std=[x/256 for x in cfg_obj.STD_RGB]
                ))
    logging.info('Data transformations:')
    for trans in result:
        logging.info(trans)
    logging.info('---------------------')
    return result


def inst_data_loader(data_train, data_dev, batch_size):
    trans = {
    'train': transforms.Compose(compose_transform_list(cfg.TRAIN)),
    'dev': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    
    dataset_train = DummyDataset(data_train, trans['train'], img_path_prefix=cfg.TRAIN.TRAIN_IMG_PREFIX)  
    dataset_dev = DummyDataset(data_dev, trans['dev'], img_path_prefix=cfg.TRAIN.DEV_IMG_PREFIX)  

    data_loader = {
    'train': pth_data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=cfg.TRAIN.SHUFFLE, num_workers=cfg.TRAIN.PROCESS_THREAD),
    'dev': pth_data.DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, num_workers=cfg.TRAIN.PROCESS_THREAD)
    }

    data_size = {'train': len(dataset_train), 'dev': len(dataset_dev)}
    
    return data_loader, data_size


def load_checkpoint(model_file, is_tar=False):
    '''
    '''
    _state_dict = t_load(model_file)
    state_dict = _state_dict['state_dict'] if is_tar else _state_dict
    if cfg.TRAIN.FT.RENAME_STATE_DICT:
        renamed_state_dict = OrderedDict((k.replace('module.',''), v) for k, v in state_dict.viewitems())
        return renamed_state_dict
        # for key in state_dict:
        #     state_dict[key.replace('module','')] = state_dict[key]
        #     del state_dict[key]
    return state_dict 


def save_checkpoint(state, model_prefix, is_best=False):
    '''
        saved dict: {"epoch":, "state_dict":, "acc":, "optimizer":} 
    '''
    file_path = '{}-{:0>4}.pth.tar'.format(model_prefix, state["epoch"])
    best_path = '{}-accpeak.pth.tar'.format(model_prefix)
    t_save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, best_path) 

