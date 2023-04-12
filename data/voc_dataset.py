import os
import os.path as osp
import numpy as np
import random
#import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
import scipy.io as scio

				
class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), label_subtract=1, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.label_subtract = label_subtract	# label classification subtracting 		
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "image_label/images/%s.jpg" % name)
            label_file = osp.join(self.root, "image_label/labels/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #scio.loadmat(datafiles["img"])  #
        #image =	 image0.get('image')
		
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE) #scio.loadmat(datafiles["label"])  
        #label =	 label0.get('label')
		
        size = image.shape
        name = datafiles["name"]
		
			
        image = np.asarray(image, np.float32)
        image -= self.mean
        label -= self.label_subtract	# label classification subtracting 	
        img_pad, label_pad = image, label	
		
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
		
        return image.copy(), label.copy(), np.array(size), name, index
        

		
class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        #self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
       
        for name in self.img_ids:
            img_file = osp.join(self.root, "test_images/%s.jpg" % name)
            self.files.append({
                "img": img_file
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #scio.loadmat(datafiles["img"]) 
        #image = image0.get('image')
		
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image.transpose((2, 0, 1))
        return image, size, name	

class VOCDataTestSet_all(data.Dataset):
    def __init__(self, root, list_image_path, mean=(128, 128, 128)):
        self.root = root
        self.mean = mean
        self.files = []     
        self.list_image_path = list_image_path       
        # self.img_ids = [i_id.strip() for i_id in open(list_image_path)] # 11
        self.img_ids = [i_id.split() for i_id in open(list_image_path)] # PA/Macapa/ 11
        
        for name in self.img_ids:
            img_file1 = osp.join(self.root, "%s" % name[0], "%s.jpg" % name[1])
            img_file2 = osp.join("%s" % name[0])            
            self.files.append({
                "img": img_file1,
                "province_city": img_file2
            })
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #scio.loadmat(datafiles["img"]) 
        #image = image0.get('image')
		
        size = image.shape
        img_name =   osp.splitext(osp.basename(datafiles["img"]))[0] 
        province_city_name = datafiles["province_city"]   # 
        
        image = np.asarray(image, np.float32)
        image -= self.mean
        image = image.transpose((2, 0, 1))
        
        return image, size, img_name, province_city_name	
      
        
class VOCDataSet_remain(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), label_subtract=1, scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.label_subtract = label_subtract	# label classification subtracting 		
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
	        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "image_remain/%s.jpg" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #scio.loadmat(datafiles["img"])  #
		
        size = image.shape
        name = datafiles["name"]
		
			
        image = np.asarray(image, np.float32)
        image -= self.mean
        # label classification subtracting 	
        img_pad = image
		
        img_h, img_w, _ = img_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
		
        return image.copy(), np.array(size), name, index
		
