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
        return False if augmenter.name in ["UnnamedAddToHueAndSaturation"] else default


# In[5]:


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


# In[6]:


train_generator.df


# In[7]:


pair = next(train_generator)  # sat image, mask image


# In[8]:


pair[0][1,:,:].shape


# In[9]:


pair[1][1,:,:].shape


# In[10]:


plt.imshow(np.reshape(pair[0][1,:,:], (1024, 1024,3)))


# In[11]:


plt.imshow(np.reshape(pair[1][1,:,:], (1024, 1024)),vmin=0, vmax=1)


# In[12]:


pair = next(val_generator)  # sat image, mask image


# In[13]:


pair[0][1,:,:,:].shape


# In[14]:


pair[1][1,:,:].shape


# In[15]:


plt.imshow(np.reshape(pair[0][1,:,:], (1024, 1024,3)))


# In[16]:


plt.imshow(np.reshape(pair[1][1,:,:], (1024, 1024)),vmin=0, vmax=1)


# In[17]:


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):

    x = Conv2D(nb_filter, kernel_size=kernel_size, strides=strides, padding=padding,
               kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    identity = input
    # 如果維度不同，則使用1x1卷積進行調整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=regularizers.l2(0.00001))(input)

    return add([identity, residual])

def basic_block(nb_filter, strides=(1, 1)):

    def f(input):
        
        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))

        return shortcut(input, residual)

    return f

def residual_block(nb_filter, repetitions, is_first_layer=False):

    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = basic_block(nb_filter, strides)(input)
        return input

    return f


# In[18]:


def D_LinkNet(input_shape=(1024,1024,3), nclass=1):

    input_ = Input(shape=input_shape)
    
    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = residual_block(64, 3, is_first_layer=True)(pool1)
    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)
    
    conv3 = residual_block(128, 4, is_first_layer=True)(pool2)
    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv3)
    
    conv4 = residual_block(256, 6, is_first_layer=True)(pool3)
    pool4 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv4)
         
    conv5 = residual_block(512, 3, is_first_layer=True)(pool4)
    
    conv6 = Conv2D(512, 3, dilation_rate=(1, 1), activation='relu', padding='same')(conv5)
    conv6 = Dropout(0.1)(conv6)
    conv7 = Conv2D(512, 3, dilation_rate=(2, 2), activation='relu', padding='same')(conv6)
    conv8 = Conv2D(512, 3, dilation_rate=(4, 4), activation='relu', padding='same')(conv7)
    conv9 = Conv2D(512, 3, dilation_rate=(8, 8), activation='relu', padding='same')(conv8)
    conv9_1 = Conv2D(512, 3, dilation_rate=(16, 16), activation='relu', padding='same')(conv9)
    conv9_1 = Dropout(0.2)(conv9_1)
    
    merge = add([conv5,conv6,conv7,conv8,conv9,conv9_1])
    
#up-scaling
    conv10 = Conv2D(128, 1, activation='relu', padding='same')(merge)
    conv10 = Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same')(conv10)
    conv10 = Dropout(0.2)(conv10)
    conv10 = Conv2D(256, 1, activation='relu', padding='same')(conv10)
    merge1 = add([conv10,conv4])
    
    conv11 = Conv2D(64, 1, activation='relu', padding='same')(merge1)
    conv11 = Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, 1, activation='relu', padding='same')(conv11)
    merge2 = add([conv11,conv3])
    
    conv12 = Conv2D(32, 1, activation='relu', padding='same')(merge2)
    conv12 = Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='same')(conv12)
    conv12 = Dropout(0.1)(conv12)
    conv12 = Conv2D(64, 1, activation='relu', padding='same')(conv12)
    merge3 = add([conv12,conv2])
    
    conv13 = Conv2D(16, 1, activation='relu', padding='same')(merge3)
    conv13 = Conv2DTranspose(16, 3, strides=(2, 2), activation='relu', padding='same')(conv13)
    conv13 = Conv2D(64, 1, activation='relu', padding='same')(conv13)

    conv14 = Conv2DTranspose(32, 4, strides=(2, 2), activation='relu', padding='same')(conv13)
    output_ = Conv2D(nclass, 3, activation='sigmoid', padding='same')(conv14)
#     up9 = UpSampling2D(size=2)(conv9)
#     up9 = Conv2D(512, 2, activation='relu', padding='same')(up9)
#     merge10 = concatenate([conv4 , up9], axis=3)
#     conv10 = Conv2D(512, 3, activation='relu', padding='same')(merge10)
#     conv10 = Conv2D(512, 3, activation='relu', padding='same')(conv10)
    
    
#     up11 = UpSampling2D(size=2)(conv10)
#     up11 = Conv2D(256, 2, activation='relu', padding='same')(up11)
#     merge11 = concatenate([conv3 , up11], axis=3)
#     conv12 = Conv2D(256, 3, activation='relu', padding='same')(merge11)
#     conv12 = Conv2D(256, 3, activation='relu', padding='same')(conv12)
    
#     up13 = UpSampling2D(size=2)(conv12)
#     up13 = Conv2D(128, 2, activation='relu', padding='same')(up13)
#     merge13 = concatenate([conv2 , up13], axis=3)
#     conv14 = Conv2D(128, 3, activation='relu', padding='same')(merge13)
#     conv14 = Conv2D(128, 3, activation='relu', padding='same')(conv14)
    
#     up14 = UpSampling2D(size=2)(conv14)
#     up14 = Conv2D(64, 2, activation='relu', padding='same')(up14)
#     merge14 = concatenate([conv1 , up14], axis=3)
#     conv15 = Conv2D(64, 3, activation='relu', padding='same')(merge14)
#     conv15 = Conv2D(64, 3, activation='relu', padding='same')(conv15)
    
#     conv16 = Conv2D(32, 4, activation='relu', padding='same')(conv15)
#     up15 = UpSampling2D(size=2)(conv16)
#     up15 = Conv2D(32, 2, activation='relu', padding='same')(up15)

#     output_ = Conv2D(nclass, 3, activation='sigmoid', padding='same')(up15)

    model = Model(inputs=input_, outputs=output_)
    model.summary()

    return model


# In[19]:


if __name__ == '__main__':
    net_final = D_LinkNet()


# In[20]:


net_final.outputs


# In[21]:


from segmentation_models.metrics import iou_score
from segmentation_models.losses import  bce_dice_loss

net_final.compile(optimizer=optimizers.Adam(lr=0.00002),
                  loss = bce_dice_loss,
                  metrics=[iou_score])


# In[22]:


from tensorflow.keras.callbacks import ModelCheckpoint
filepath="weights-improvement-{epoch:02d}-{val_iou_score:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]


# In[23]:


history = net_final.fit_generator(generator=train_generator, steps_per_epoch=int(np.ceil(len(train_generator.df))/batch_size),
                                  validation_data=val_generator, epochs=60,callbacks=callbacks_list) 


# In[24]:


plt.plot(history.history['iou_score'], label='iou_score')
plt.plot(history.history['val_iou_score'], label='val_iou_score')
plt.xlabel('Epoch')
plt.ylabel('iou')
# plt.ylim([0, 1])
plt.legend(loc='lower right')
# test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
print('iou_score=',history.history['iou_score'][-1],"   ","val_iou_score=",history.history['val_iou_score'][-1])


# In[25]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.ylim([0.3, 1])
plt.legend(loc='upper right')
# test_loss, test_acc = net_final.evaluate([X_test],Y_test, verbose=2)
print('loss=',history.history['loss'][-1],"   ","val_loss=",history.history['val_loss'][-1])


# In[40]:


test_generator = DataGenerator(dir='test', img_col='sat', mask_col='mask', shuffle=False, batch_size=1)
test_generator.df


# In[41]:


pair = next(test_generator)  # sat image, mask image
pair[0].shape


# In[42]:


import os  
os.environ["CUDA_VISIBLE_DEVICES"] = ""  
predict_test=net_final.predict_generator(test_generator, verbose=1)
predict_test.size


# In[43]:


def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    
    return final_conv


# In[44]:


# predict_test = convert_binary(predict_test, 0.2)


# In[45]:


predict_test.shape


# In[46]:


plt.imshow((np.reshape(predict_test[14,:,:], (1024, 1024))))


# In[47]:


import cv2
name = [i.split('\\')[-1].split('_')[0] for i in test_generator.df['filepath']]
for i in range(21):
    predict = (np.reshape(predict_test[i,:,:], (1024, 1024)))
    predict = convert_binary(predict,0.05)
    filename = "C:\\Users\\USER\\Machine Learning course\\ADIP\\Unet\\test\\" + name[i] + "_mask.png"
    cv2.imwrite(filename, predict) 


# In[48]:


import cv2
cv2.imshow('My Image', np.reshape(predict_test[14,:,:], (1024, 1024)))

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[49]:


import cv2
predict_test = np.reshape(predict_test[14,:,:], (1024, 1024))
predict_test = convert_binary(predict_test,0.05)
predict_test


# In[50]:


plt.imshow(predict_test)


# In[37]:


cv2.imshow('My Image', predict_test)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()





