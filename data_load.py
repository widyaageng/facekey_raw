import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2

# Added libraries for rotation transform class
import math
import PIL
from torchvision import transforms, utils



class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        image = mpimg.imread(image_name)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        
        # adding np.asarray to image, giving error: src is not a numpy array, neither a scalar
        image_copy = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0


        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        key_pts = key_pts - torch.Tensor([left, top]).double() # fixing it

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

    
####### added class for rotation from CENTER #########
class RandomRotate(object):
    """Perform Randomrotation based on image center"""
    def __init__(self, degree):
        if 0 <= degree <= 180:
            self.degree = degree
        else:
            print('class RandomRotate:: Notebook1:: Degree argument is not within 0-180!')
            raise
    
    @staticmethod
    def _get_params(angle):
        # np use radian
        return np.random.uniform(-angle, angle, (1,)).item()*math.pi/180
            
            
    def __call__(self, sample):
        
        angle = self._get_params(self.degree)
       
        image, key_pts = sample['image'], sample['keypoints']
        image_center = [int(image.shape[0]/2), int(image.shape[1]/2)]
        
        # PIL rotation angle is in degree
        image = transforms.functional.rotate(PIL.Image.fromarray(np.array(image)), angle*180/math.pi,  center=image_center)
        
        # Getting rotation matrix from original cartesian coordinate
        if isinstance(sample['keypoints'], torch.Tensor):
            key_pts = sample['keypoints'].numpy() - image_center
        else:
            key_pts = sample['keypoints'] - image_center
        init_angle = np.arctan2(key_pts[:,0], key_pts[:,1])
        rot_mat = np.array([[np.sin(init_angle + angle), np.cos(init_angle + angle)]])
        rot_mat = rot_mat.squeeze(0).T
        
        # Getting new cartesian coordinate
        key_pts = np.sqrt(key_pts[:,0]**2 + key_pts[:,1]**2)
        key_pts = key_pts.reshape((len(key_pts), 1))
        key_pts = np.concatenate((key_pts, key_pts), axis=1)
        key_pts = np.multiply(key_pts, rot_mat)
        key_pts = key_pts + image_center
        
        assert key_pts.shape[0] == sample['keypoints'].shape[0]
        assert key_pts.shape[1] == sample['keypoints'].shape[1]
        
        return {'image': torch.from_numpy(np.array(image)),
                'keypoints': torch.from_numpy(key_pts)}