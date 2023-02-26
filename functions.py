#!/usr/bin/python3
import mtcnn
import numpy as np
from keras import backend as K
from keras.utils.data_utils import get_file
from scipy import spatial
import os
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import PIL
import shutil
import tensorflow as tf



model=tf.keras.models.load_model("./faceai.h5")



def preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def to_np(fpath):
    img=cv2.imread(fpath)
    img=cv2.resize(img,(224,224))
    img=np.asarray(img,dtype='float32')
    return np.expand_dims(img,axis=0).tolist()
    


def generate_embeddings(face):

    face=preprocess_input(face,version=2)
    embeddings_known=model.predict(face)
    return embeddings_known


def compare(embedding_known,embedding_unknown,limit=0.45):
    dist=spatial.distance.cosine(embedding_known.flatten(),embedding_unknown.flatten())
    if dist>limit:
        return 0
    else:
        return 1
    
'''
check_face=to_np('./check.jpg')
true_face=to_np('./true.jpg')
print(compare(generate_embeddings(check_face),generate_embeddings(true_face)))
'''
