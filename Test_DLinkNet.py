#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D, Dropout,Flatten,UpSampling2D,concatenate,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import os 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import optimizers
from tensorflow import keras
import pandas as pd
from typing import Optional, Tuple, List
from glob import glob
from tensorflow.keras.utils import Sequence
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from PIL import Image 


# In[2]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# In[3]:


def get_project_dir():

    """
    Get the full path to the repository
    """

#     dir_as_list = os.path.dirname(__file__).split("/")
#     index = dir_as_list.index("test")
#     project_directory = f"/{os.path.join(*dir_as_list[:index + 1])}"
    
    project_directory = r"C:\Users\USER\Machine Learning course\ADIP"
    
    return Path(project_directory)


# In[4]:


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmenters import Sequential

class DataGenerator(Sequence):

    def __init__(self, dir: str, img_col: str, mask_col: str, img_augmentation: Optional[Sequential] = None, 
                 sample_size: Optional[int] = None, batch_size: int = 32, shuffle: bool = True):
        
        """
        Args:
            dir: directory in which images are stored
            sample_size: Optional; number of images will be sampled in each of sub_directory,
            from tying import Union
            sample_size: Union[int, None] -> Optional[int]
            if not provided all images in the dir are taken into account.
            batch_size: number of images in each of batch
            shuffle: if shuffle the order of the data
        """
        
        self.dir = dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_size = sample_size
        self.img_col = img_col
        self.mask_col = mask_col
        self.img_augmentation = img_augmentation

        self.on_epoch_end()

        self.max = self.__len__()
        self.n = 0

    def __transform_to_dataframe(self) -> pd.DataFrame:
        
        """
        transform the data into a pandas dataframe to track the image files and the corresponding masks
        """
        data = []

        data_files = glob(f"{get_project_dir()}\\{self.dir}/*.jpg")
        
        if self.dir == 'test':
            mask_files = [0 for f in data_files]
            df = pd.DataFrame(data=np.array([data_files,mask_files]).T, columns=['filepath','label'], dtype=object)
            return df
        
        else:
            if self.sample_size:
                sampled_files = random.sample(data_files, min(self.sample_size, len(data_files)))
            else:
                sampled_files = data_files
        
            mask_files = [f.replace('sat', 'mask').replace('jpg', 'png') for f in sampled_files]
    
            df = pd.DataFrame(data=np.array([sampled_files, mask_files]).T, columns=[self.img_col, self.mask_col], dtype=object)
        
        return df

    def on_epoch_end(self):
        
        self.df = self.__transform_to_dataframe()
        self.indices = self.df.index.tolist()

        self.index = np.arange(len(self.indices))
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        #  Denotes the number of batches per epoch
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        # Generate one batch of data
        # Generate indices of the batch
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        batch = [self.indices[k] for k in index]
        # Generate data
        X, y = self.__get_data(batch)

        return X, y

    def __get_data(self, batch: List) -> Tuple[np.ndarray, np.ndarray]:
        
        df_batch = self.df.loc[batch]

        sat_dataset = []
        mask_dataset = []
        
        if self.dir == 'test':
            for _, row in df_batch.iterrows():
                # input image
                f = row['filepath']
                sat_image = Image.open(f)
                sat_dataset.append(np.array(sat_image)/255.0)
                
                f = row['label']
                mask_image = np.zeros((1024, 1024))
                mask_dataset.append(np.where((mask_image > 128), 1, 0))

            return np.array(sat_dataset), np.array(mask_dataset)
        
        else:
            for _, row in df_batch.iterrows():
                if not self.img_augmentation: 
                    f = row[self.img_col]
                    sat_image = Image.open(f)
                    sat_dataset.append(np.array(sat_image)/255.0)
                
                    f = row[self.mask_col]
                    mask_image = np.array(Image.open(f).convert('L'))
                    mask_dataset.append(np.where((mask_image > 128), 1, 0))
                else:
                    # lock the image augmentation
                    seq_det = self.img_augmentation.to_deterministic()

                    # input image
                    f = row[self.img_col]
                    sat_image = seq_det.augment_image(np.array(Image.open(f)))
                    sat_dataset.append(sat_image/255.0)


                    # mask image
                    f = row[self.mask_col]
                    mask_image = seq_det.augment_image(np.array(Image.open(f).convert('L')), hooks=ia.HooksImages(activator=self.activator))
                    mask_dataset.append(np.where((mask_image > 128), 1, 0))

                    # for multiclasses: one-hot encoding
                    # new_mask = np.zeros(mask.shape + (num_classes, ))
                    # for i in range(num_classes):
                    #   new_mask[mask==i, i] = 1
                
            return np.array(sat_dataset, dtype='float32'), np.array(mask_dataset, dtype='float32')

    def __next__(self):
        
        """
        generate data of size batch_size
        """
        
        if self.n >= self.max:
            self.n = 0

        result = self.__getitem__(self.n)
        self.n += 1
        return result
    
    def activator(self, images, augmenter, parents, default):
        return False if augmenter.name in ["GaussianBlur"] else default


# In[5]:


from segmentation_models.metrics import iou_score
from segmentation_models.losses import  bce_dice_loss
net_final = tf.keras.models.load_model('weights-improvement-53-0.51.h5',custom_objects={'binary_crossentropy_plus_dice_loss': bce_dice_loss,'iou_score': iou_score})


# In[6]:


test_generator = DataGenerator(dir='test', img_col='sat', mask_col='mask', shuffle=False, batch_size=1)
test_generator.df


# In[7]:


import os  
os.environ["CUDA_VISIBLE_DEVICES"] = ""  
predict_test=net_final.predict_generator(test_generator, verbose=1)
predict_test.size


# In[8]:


def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv


# In[9]:


predict_test.shape


# In[10]:


import cv2
import statistics
name = [i.split('\\')[-1].split('_')[0] for i in test_generator.df['filepath']]
for i in range(21):
    predict = (np.reshape(predict_test[i,:,:], (1024, 1024)))
    predict = convert_binary(predict,statistics.mean(list(predict_test[i].flatten())))
    filename = "C:\\Users\\USER\\Machine Learning course\\ADIP\\test\\" + name[i] + "_mask.png"
    cv2.imwrite(filename, predict) 


# In[ ]:




