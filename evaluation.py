import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from deeplabv3plus import DeeplabV3plus, Res50_DeeplabV3plus
from data.voc_dataset import VOCDataSet,VOCDataTestSet
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc


from metric import ConfusionMatrix

IMG_MEAN = np.array((102.2058,110.1798,120.2015), dtype=np.float32)



DATA_DIRECTORY = './Train_dataset/'
DATA_LIST_PATH = './Train_dataset/test_list.txt' 
NUM_CLASSES = 2 # 

RESTORE_FROM = './checkpoints/semi_res101deeplabv3plus/VOC_60000.pth'
SAVE_DIRECTORY = './results/'
os.environ['CUDA_VISIBLE_DEVICES']='0'





#interp = nn.Upsample(size=(1200, 1200), mode='bilinear', align_corners=True)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
						
                        
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")

    parser.add_argument("--with-mlmt", action="store_true",
                        help="combine with Multi-Label Mean Teacher branch")
    parser.add_argument("--save-output-images", action="store_false",
                        help="save output images")
    return parser.parse_args()

class Clou_Colorize(object):
    def __init__(self, n=4):
        self.cmap = color_map(4)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image
		
	

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):  # 0,1,2,...,N-1
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap



def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()
    #gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('save_dir'+ args.save_dir)		

    model = DeeplabV3plus(num_classes=args.num_classes) 
    # DeeplabV3plus(num_classes=args.num_classes) Res50_DeeplabV3plus(num_classes=args.num_classes)
    model.cuda()

		
    saved_state_dict = torch.load(args.restore_from)	# load model parameters path
    model.load_state_dict(saved_state_dict)  # load the model parameters to the network

    model.eval()  # test stage
    model.cuda() # using GPU 
	
    # load image for test
  
    testloader = data.DataLoader(VOCDataTestSet(args.data_dir, args.data_list, mean=IMG_MEAN), 
                                    batch_size=1, shuffle=False, pin_memory=True)	

                                    
    #interp = nn.Upsample(size=(1200, 1200), mode='bilinear', align_corners=True)

    interp = nn.Upsample(size=(980, 1880), mode='bilinear', align_corners=True)   
    #data_list = []
    colorize = Clou_Colorize()
   

 
    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, size, name = batch		
        size = size[0]
        with torch.no_grad():
             output,_ = model(Variable(image).cuda()) # model test 
			 
        output = interp(output).cpu().data[0].numpy() # resize the output result as the same size as the input image

        
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
       
        if args.save_output_images:
           filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
           color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
           color_file.save(filename)


if __name__ == '__main__':
    main()
