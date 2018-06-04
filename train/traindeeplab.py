from __future__ import absolute_import
from __future__ import print_function
from modelV3 import Deeplabv3
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras import utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import scipy.ndimage as ndi
import numpy as np
import pandas as pd
import sys
import os

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='constant', cval=0.):

    theta = np.deg2rad(rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x,h,w

def rotate_xy(x, y, angle, w, h):
    angle=np.deg2rad(angle)
    origin = np.array([[x, y, 1]])
    left = np.array([[1, 0, 0], [0, -1, 0], [-0.5 * w, 0.5 * h, 1]])

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    mid = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
    right = np.array([[1, 0, 0], [0, -1, 0], [0.5 * w, 0.5 * h, 1]])
    return np.dot(origin, np.dot(np.dot(left, mid), right))


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def makelabel(orilabel):
    newlabel = np.zeros((orilabel.shape[0], orilabel.shape[1], 14))
    layer_idx = np.arange(orilabel.shape[0]).reshape(orilabel.shape[0], 1)
    component_idx = np.tile(np.arange(orilabel.shape[1]), (orilabel.shape[0], 1))
    newlabel[layer_idx, component_idx, orilabel] = 1
    return newlabel
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(np.random.randint(start, stop))
    return random_list

def dataGen(batch_size, paths, newLandmark, landmarkNum):
    # to determine dimensions
    pathlen = len(paths)
    while (1):
        samples = random_int_list(0, pathlen - 1, batch_size)
        X = np.zeros((batch_size,512,512,3))
        Y = np.zeros((batch_size,512,512,landmarkNum))
        k=0
        for i in samples:
            imagePath = paths[i]
            rg=np.random.uniform(-45,45)
            ytemp=np.zeros((512,512))
            # ytemp=np.reshape(ytemp,(-1,1))
            xtemp=np.zeros((512,512,3))
            img = image.load_img('train/' + imagePath)
            x = image.img_to_array(img)
            rg=np.random.uniform(-45,45)
            rg=np.random.randint(0,2)
            x = preprocess_input(x)
            x,h,w=random_rotation(x,rg,0,1,2)
            xtemp[0:x.shape[0], 0:x.shape[1], :] = x[:, :, :]
            X[k]=xtemp
            for j in range(landmarkNum-1):
                if newLandmark[i][2*j]!=-1 and newLandmark[i][2*j+1]!=-1:
                    x1=int(newLandmark[i][2*j])
                    y1=int(newLandmark[i][2*j+1])
                    af=rotate_xy(x1,y1,rg,w,h)
                    x1=int(af[0,0])
                    y1=int(af[0,1])
                    if x1>=0 and y1>=0 and x1<=w and y1<=h:
                        if x1<=0:
                            x1=0
                        if y1<=0:
                            y1=0
                        if x1>=512:
                            x1=511
                        if y1>=512:
                            y1=511
                        ytemp[x1][y1]=j+1
            Y[k] = utils.to_categorical(ytemp, landmarkNum)
            k=k+1
        yield (X, Y)


if __name__ == '__main__':
    closeType = sys.argv[1]                    #The model should be trained for each kind of cloth
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
    batchSize = 4
    landmarks = pd.read_csv('train/Annotations/train.csv')
    landmarks = landmarks[landmarks['image_category'] == closeType]
    imgPath = landmarks.iloc[:, 0]
    imgPath = imgPath.tolist()
    lists = []
    cw = np.array([0.1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    if closeType == 'blouse':
        availableLandmark = [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
        cw = np.array([0.1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    elif closeType == 'outwear':
        availableLandmark = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        cw = np.array([0.1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    elif closeType == 'trousers':
        availableLandmark = [17, 18, 21, 22, 23, 24, 25]
        cw = np.array([0.1, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    elif closeType == 'skirt':
        availableLandmark = [17, 18, 19, 20]
        cw = np.array([0.1, 1000, 1000, 1000, 1000])
    elif closeType == 'dress':
        availableLandmark = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20]
        cw = np.array([0.1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    landmarNum = len(availableLandmark)
    for idx in range(len(landmarks)):
        lands = landmarks.iloc[idx, availableLandmark]
        li = []
        for data in lands.str.split('_'):
            li.extend([int(i) for i in data])
        lists.append(li)
    lists = np.array(lists)
    newLists = np.zeros((lists.shape[0],landmarNum*2))
    for i in range(lists.shape[0]):
        k = 0
        for j in range(lists.shape[1]):
            if j == 0:
                newLists[i][k] = lists[i][j]
                k = k + 1
                continue
            elif (j + 1) % 3 == 0:
                continue
            else:
                newLists[i][k] = lists[i][j]
                k = k + 1
    mod = Deeplabv3(weights=None,classes=landmarNum+1,backbone='xception')
    mod.load_weights('v3model_'+closeType+'.h5',by_name=True)
    adam = Adam(lr=0.0001,decay=0.0001)
    mod.compile(optimizer=adam, loss=weighted_categorical_crossentropy(cw))
    checkpointer = ModelCheckpoint(filepath=closeType+'_model_{epoch:03d}_argu.h5', verbose=1, save_best_only=False,save_weights_only=True)
    mod.fit_generator(dataGen(batchSize, paths=imgPath, newLandmark=newLists, landmarkNum=landmarNum+1),
                      steps_per_epoch=int(newLists.shape[0]/batchSize), epochs=3000, callbacks=[checkpointer],use_multiprocessing=False)
