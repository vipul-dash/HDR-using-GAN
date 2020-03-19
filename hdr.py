import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers as tfl
img =cv2.imread("test.jpeg")
print(img.shape)
original_img=np.asarray(img)
#print(original_img)

#hYPER parameters 

epsilon=0.00005
class hdrGAN:
    def __init__(self,generator,discriminator,inputimg):
        self.generator=generator
        self.discriminator=discriminator
        self.img=inputimg
        self.input_dim=self.img.shape
    def generator(input_dim,output_dim,img):
           with tf.variable_scope("generator")  as scope: 

            model=tf.keras.Sequential()
       ############################################################ down sampling layers ############################
       
       
               # first block 
       
            model.add(tfl.Conv2D(filters=64,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block1'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())

            #second block

            model.add(tfl.Conv2D(filters=128,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block2'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())

            #third  block

            model.add(tfl.Conv2D(filters=256,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block3'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())



            #fourth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block4'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())
             
            #fifth block

            model.add(tfl.Conv2D(filters=512,kernel_size=(10,10),strides=(2,2),padding='same',name='conv_block5'))
            model.add(tfl.BatchNormalization(epsilon=epsilon))
            model.add(tfl.PRelu())
             
           
               

                         
               

