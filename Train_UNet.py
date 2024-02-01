#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras 
from tensorflow.keras.utils import Sequence
from skimage.io import imread
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
 
import albumentations as albu
from albumentations import Resize
from glob import glob
from typing import Optional, Tuple, List
from pathlib import Path
from PIL import Image 
from tensorflow.keras import optimizers


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


# root_dir = r'C:\Users\USER\Machine Learning course\ADIP\Unet\road_segmentation_ideal\training'
# image_folder = r'input'
# glob(f"{root_dir}\\{image_folder}/*.png")


# In[4]:


# class DataGeneratorFolder(Sequence):
#     def __init__(self, root_dir=str, image_folder=str, mask_folder=str, 
#                  batch_size=1, image_size=768, nb_y_features=1, 
#                  augmentation=None,
#                  suffle=True):
#         self.image_filenames = glob(f"{root_dir}\\{image_folder}/*.png")
#         self.mask_names = glob(f"{root_dir}\\{mask_folder}/*.png")
#         self.batch_size = batch_size
#         self.currentIndex = 0
#         self.augmentation = augmentation
#         self.image_size = image_size
#         self.nb_y_features = nb_y_features
#         self.indexes = None
#         self.suffle = suffle
        
#     def __len__(self):
#         """
#         Calculates size of batch
#         """
#         return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

#     def on_epoch_end(self):
#         """Updates indexes after each epoch"""
#         if self.suffle==True:
#             self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)
        
#     def read_image_mask(self, image_name, mask_name):
#         return imread(image_name)/255, (imread(mask_name, as_gray=True) > 0).astype(np.int8)

#     def __getitem__(self, index):
#         """
#         Generate one batch of data
        
#         """
#         # Generate indexes of the batch
#         data_index_min = int(index*self.batch_size)
#         data_index_max = int(min((index+1)*self.batch_size, len(self.image_filenames)))
        
#         indexes = self.image_filenames[data_index_min:data_index_max]
#         print(indexes)

#         this_batch_size = len(indexes) # The last batch can be smaller than the others
        
#         # Defining dataset
#         X = np.empty((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
#         y = np.empty((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.uint8)

#         for i, sample_index in enumerate(indexes):

#             X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i], 
#                                                     self.mask_names[index * self.batch_size + i])
                 
#             # if augmentation is defined, we assume its a train set
#             if self.augmentation is not None:
                  
#                 # Augmentation code
#                 augmented = self.augmentation(self.image_size)(image=X_sample, mask=y_sample)
#                 image_augm = augmented['image']
#                 mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)
#                 X[i, ...] = np.clip(image_augm, a_min = 0, a_max=1)
#                 y[i, ...] = mask_augm
            
#             # if augmentation isnt defined, we assume its a test set. 
#             # Because test images can have different sizes we resize it to be divisable by 32
#             elif self.augmentation is None and self.batch_size ==1:
#                 X_sample, y_sample = self.read_image_mask(self.image_filenames[index * 1 + i], 
#                                                       self.mask_names[index * 1 + i])
#                 augmented = Resize(height=(X_sample.shape[0]//32)*32, width=(X_sample.shape[1]//32)*32)(image = X_sample, mask = y_sample)
#                 X_sample, y_sample = augmented['image'], augmented['mask']

#                 return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32),\
#                        y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.uint8)

#         return X, y


# In[5]:


# def aug_with_crop(image_size = 256, crop_prob = 1):
#     return Compose([
#         RandomCrop(width = image_size, height = image_size, p=crop_prob),
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         RandomRotate90(p=0.5),
#         Transpose(p=0.5),
#         ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
#         RandomBrightnessContrast(p=0.5),
#         RandomGamma(p=0.25),
#         IAAEmboss(p=0.25),
#         Blur(p=0.01, blur_limit = 3),
#         OneOf([
#             ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#             GridDistortion(p=0.5),
#             OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
#         ], p=0.8)
#     ], p = 1)

# test_generator = DataGeneratorFolder(root_dir = r'C:\Users\USER\Machine Learning course\ADIP\Unet\road_segmentation_ideal\training',
#                                      image_folder = r'input', 
#                                      mask_folder = r'output',
#                                      batch_size = 1,
#                                      nb_y_features = 1, augmentation = aug_with_crop)
# Xtest, ytest = test_generator.__getitem__(0)
# plt.imshow(Xtest[0])     
# plt.show()
# plt.imshow(ytest[0, :,:,0])
# plt.show() 


# In[6]:


# test_generator = DataGeneratorFolder(root_dir = r'C:\Users\USER\Machine Learning course\ADIP\Unet\road_segmentation_ideal\training',
#                                      image_folder = r'input', 
#                                      mask_folder = r'output',
#                                      batch_size=1,
#                                      augmentation = aug_with_crop)

# train_generator = DataGeneratorFolder(root_dir = r'C:\Users\USER\Machine Learning course\ADIP\Unet\road_segmentation_ideal\training',
#                                      image_folder = r'input', 
#                                      mask_folder = r'output',
#                                       augmentation = aug_with_crop,
#                                       batch_size=4,
#                                       image_size=512)


# In[7]:


def get_project_dir():

    """
    Get the full path to the repository
    """

#     dir_as_list = os.path.dirname(__file__).split("/")
#     index = dir_as_list.index("test")
#     project_directory = f"/{os.path.join(*dir_as_list[:index + 1])}"
    
    project_directory = r"C:\Users\TESWISE\Desktop\test"
    
    return Path(project_directory)


# In[8]:


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
        return False if augmenter.name in ["UnnamedAddToHueAndSaturation"] else default


# In[9]:


sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # apply operations on 50% of input data
batch_size = 2

seq = iaa.Sequential([sometimes(iaa.Affine(scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                                           translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                                           rotate=(-30, 30),
                                           order=3,
                                           cval=0)),
                      sometimes(iaa.AddToHueAndSaturation((-50, 50), per_channel=True)),
                      iaa.Fliplr(0.5), # horizontally flip 50% of the images
                      iaa.Flipud(0.5),
                    ])
                    

train_generator = DataGenerator(dir='train', img_col='sat', mask_col='mask', img_augmentation=seq, shuffle=True, batch_size=batch_size)
val_generator = DataGenerator(dir='val', img_col='sat', mask_col='mask', shuffle=False, batch_size=batch_size)


# In[10]:


train_generator.df


# In[11]:


pair = next(train_generator)  # sat image, mask image


# In[12]:


pair[0].shape


# In[13]:


pair[1].shape


# In[14]:


plt.imshow(np.reshape(pair[0][1,:,:], (1024, 1024,3)))


# In[15]:


plt.imshow(np.reshape(pair[1][1,:,:], (1024, 1024)),vmin=0, vmax=1)


# In[16]:


pair = next(val_generator)  # sat image, mask image


# In[17]:


pair[0][1,:,:,:].shape


# In[18]:


pair[1][1,:,:].shape


# In[19]:


pair[0].shape


# In[20]:


pair[1].shape


# In[21]:


plt.imshow(np.reshape(pair[0][1,:,:], (1024, 1024,3)))


# In[22]:


plt.imshow(np.reshape(pair[1][1,:,:], (1024, 1024)),vmin=0, vmax=1)


# In[23]:


# import cv2
# cv2.imshow('My Image', np.reshape(pair[0], (1024, 1024,3)))

# # 按下任意鍵則關閉所有視窗
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[24]:


# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

# # reduces learning rate on plateau
# lr_reducer =  (factor=0.1,
#                                cooldown= 10,
#                                patience=10,verbose =1,
#                                min_lr=0.1e-5)
# mode_autosave = ModelCheckpoint("./weights/road_crop.efficientnetb0imgsize.h5",monitor='val_iou_score', 
#                                    mode = 'max', save_best_only=True, verbose=1, period =10)

# # stop learining as metric on validatopn stop increasing
# early_stopping = EarlyStopping(patience=10, verbose=1, mode = 'auto') 

# # tensorboard for monitoring logs
# tensorboard = TensorBoard(log_dir='./logs/tenboard', histogram_freq=0,
#                           write_graph=True, write_images=False)

# callbacks = [mode_autosave, lr_reducer, tensorboard, early_stopping]


# In[25]:


from segmentation_models import Unet
from tensorflow.keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score

# def plot_training_history(history):
#     """
#     Plots model training history 
#     """
#     fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
#     ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
#     ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
#     ax_loss.legend()
#     ax_acc.plot(history.epoch, history.history["iou_score"], label="Train iou")
#     ax_acc.plot(history.epoch, history.history["val_iou_score"], label="Validation iou")
#     ax_acc.legend()
    
model = Unet('resnet34', input_shape=(1024, 1024, 3), encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze = False)
model.summary()


# In[26]:


model.output


# In[27]:


model.output_shape


# In[28]:


model.compile(optimizer = optimizers.Adam(lr=0.00002),
                    loss=bce_dice_loss, metrics=[iou_score])

from tensorflow.keras.callbacks import ModelCheckpoint
filepath="weights-improvement-{epoch:02d}-{val_iou_score:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(generator=train_generator, steps_per_epoch=int(np.ceil(len(train_generator.df))/batch_size),
                                  validation_data=val_generator, epochs=100,callbacks=callbacks_list)  
# history = model.fit_generator(train_generator, shuffle =True,
#                   epochs=50, workers=4, use_multiprocessing=True,
#                   validation_data = val_generator, 
#                   verbose = 1, callbacks=callbacks)
# # plotting history
# plot_training_history(history)


# In[29]:


plt.plot(history.history['iou_score'], label='iou_score')
plt.plot(history.history['val_iou_score'], label='val_iou_score')
plt.xlabel('Epoch')
plt.ylabel('iou')
# plt.ylim([0, 1])
plt.legend(loc='lower right')
# test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
print('iou_score=',history.history['iou_score'][-1],"   ","val_iou_score=",history.history['val_iou_score'][-1])


# In[30]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.ylim([0.3, 1])
plt.legend(loc='upper right')
# test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
print('loss=',history.history['loss'][-1],"   ","val_loss=",history.history['val_loss'][-1])


# In[31]:


test_generator = DataGenerator(dir='test', img_col='sat', mask_col='mask', shuffle=False, batch_size=batch_size)
test_generator.df


# In[32]:


predict_test=model.predict_generator(test_generator, verbose=1)
predict_test.size


# In[33]:


def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv


# In[34]:


import cv2
name = [i.split('\\')[-1].split('_')[0] for i in test_generator.df['filepath']]
for i in range(21):
    predict = (np.reshape(predict_test[i,:,:], (1024, 1024)))
    predict = convert_binary(predict,0.2)
    filename = "C:\\Users\\TESWISE\\Desktop\\test\\" + name[i] + "_mask.png"
    cv2.imwrite(filename, predict) 


# In[35]:


model.save('C:\\Users\\TESWISE\\Desktop\\test\\APIP_unet.h5')


# In[ ]:




