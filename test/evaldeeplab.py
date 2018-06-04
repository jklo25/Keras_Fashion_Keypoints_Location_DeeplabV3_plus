from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
from modelV3 import Deeplabv3
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
if __name__ == '__main__':
    landmarks = pd.read_csv('test/vali/answer.csv')
    category='blouse'
    landmarks = landmarks[landmarks['image_category'] == category]
    imgPath = landmarks.iloc[:, 0]
    imgPath = imgPath.tolist()
    lists = []
    if category == 'blouse':
        availableLandmark = [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
    elif category == 'outwear':
        availableLandmark = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    elif category == 'trousers':
        availableLandmark = [17, 18, 21, 22, 23, 24, 25]
    elif category == 'skirt':
        availableLandmark = [17, 18, 19, 20]
    elif category == 'dress':
        availableLandmark = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20]
    landmarNum = len(availableLandmark)
    for idx in range(len(landmarks)):
        lands = landmarks.iloc[idx, availableLandmark]
        li = []
        for data in lands.str.split('_'):
            li.extend([int(i) for i in data])
        lists.append(li)
    lists = np.array(lists)

    dicts = {'blouse': 15, 'skirt': 6, 'dress': 17, 'outwear': 16, 'trousers': 9}
    df = pd.DataFrame(columns=[i for i in range(dicts[category])])
    print(df.shape)
    print(len(landmarks))
    paths = landmarks.iloc[:, 0]
    paths = paths.tolist()
    if category=='blouse':
        clsnum=14
    model = Deeplabv3(weights=None, classes=clsnum, backbone='xception')
    model.load_weights('../train/'+category+'_model.h5')
    baifenbi=0.0
    n = 0
    for imageNum in range(len(landmarks)):
        # toShow = mpimg.imread('test/vali/'+paths[imageNum])
        img = image.load_img('test/vali/'+paths[imageNum])
        x = image.img_to_array(img)
        x = preprocess_input(x)
        xtemp = np.zeros((512, 512, 3))
        xtemp[0:x.shape[0], 0:x.shape[1], :] = x[:, :, :]
        xtemp = np.expand_dims(xtemp, axis=0)
        result = model.predict(xtemp)
        heatmap = result[0]
        # print(toShow.shape)
        if category == 'blouse':
            s = np.sqrt(np.square(lists[imageNum][15] - lists[imageNum][18]) + np.square(
                lists[imageNum][16] - lists[imageNum][19]))
        elif category == 'outwear':
            s = np.sqrt(np.square(lists[imageNum][12] - lists[imageNum][15]) + np.square(
                lists[imageNum][13] - lists[imageNum][16]))
        elif category == 'trousers':
            s = np.sqrt(np.square(lists[imageNum][0] - lists[imageNum][3]) + np.square(
                lists[imageNum][1] - lists[imageNum][4]))
        elif category == 'skirt':
            s = np.sqrt(np.square(lists[imageNum][0] - lists[imageNum][3]) + np.square(
                lists[imageNum][1] - lists[imageNum][4]))
        elif category == 'dress':
            s = np.sqrt(np.square(lists[imageNum][15] - lists[imageNum][18]) + np.square(
                lists[imageNum][16] - lists[imageNum][19]))
        k=1
        for i in range(int(lists.shape[1] / 3)):
            if lists[imageNum][3*i+2]==-1:
                k=k+1
            else:
                sq = heatmap[:, :, k]
                w, h = sq.shape
                index = int(np.argmax(sq))
                x1 = int(index / w)
                y1 = index % h
                d=np.sqrt(np.square(lists[imageNum][3*i]-x1)+np.square(lists[imageNum][3*i+1]-y1))
                if(s>0):
                    baifenbi=baifenbi+d/s
                k=k+1
                n=n+1
        print(baifenbi/n)
    print(baifenbi/n)
