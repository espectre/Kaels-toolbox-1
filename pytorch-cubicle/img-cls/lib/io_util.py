import os
from torchvision import transforms
import torch.utils.data as pth_data
from PIL import Image
import numpy as np
from config import cfg

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

def inst_data_loader(data_train, data_dev):
    trans_list = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'dev': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    
    dataset_train = DummyDataset(data_train, trans_list['train'], img_path_prefix=cfg.TRAIN.TRAIN_IMG_PREFIX)  
    dataset_dev = DummyDataset(data_dev, trans_list['dev'], img_path_prefix=cfg.TRAIN.DEV_IMG_PREFIX)  

    data_loader = {
    'train': pth_data.DataLoader(dataset=dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE, num_workers=cfg.TRAIN.PROCESS_THREAD),
    'dev': pth_data.DataLoader(dataset=dataset_dev, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.PROCESS_THREAD)
    }

    data_size = {'train': len(dataset_train), 'dev': len(dataset_dev)}
    
    return data_loader, data_size
