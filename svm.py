from sklearn import svm

import cv2
import numpy as np
import os
import pickle

import pandas as pd

width = 227
height = 227
channel = 3


def preprocessImage(file_name):
    print("reading - {}".format(file_name))
    image = cv2.imread('crops/' + file_name)
    im = cv2.resize(image, (width, height)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

csv_filepath = 'labels.csv'

if os.path.isfile(csv_filepath):
    df = pd.read_csv('labels.csv',index_col=0)
# else:
#     with open('crops/labels2.txt') as f:
#         content = f.readlines()

#     labels = [ int(s.strip()) for s in content ]

#     file_names = [ f for f in os.listdir('crops/') if f.endswith(".jpg")]

#     zipped = list(zip(file_names, labels ))

#     df2 = pd.DataFrame(zipped)
#     df2.columns = ['name', 'label']
#     df.columns  = ['name', 'label']

#     df = df.append(df2, ignore_index=True)
#     df.to_csv('labels.csv')

# df_merged = df.append(df2, ignore_index=True)
# df_merged.to_csv('labels_merged.csv')

# Realign dataframe
# file_names = [ f for f in os.listdir('crops/') if f.endswith(".jpg")]
# labels = df['label'].as_matrix()
# zipped = list(zip(file_names, labels ))
# df = pd.DataFrame(zipped)


from bottleneck_features import create_googlenet,get_squeezenet


images_filepath = 'data_train_squeezenet.npy'
extracted_features = 'data_features_squeezenet.npy'

if os.path.isfile(extracted_features):
    out = np.load(extracted_features)
else:
    if os.path.isfile(images_filepath):
        data = np.load(images_filepath)
    else:
        images_count = df.shape[0]
        data = np.empty([images_count, channel, width,height])
        for i, file_name in enumerate(df.ix[:,0]):
            data[i] = preprocessImage(file_name)
        np.save(images_filepath, data)
    model = get_squeezenet(nb_classes=1000,
     path_to_weights='model/squeezenet_weights_th_dim_ordering_th_kernels.h5',dim_ordering='th')
    out = model.predict(data)
    np.save(extracted_features, out)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from keras.utils.np_utils import to_categorical

y = to_categorical(df.ix[:,1])
X = out

classifier = OneVsRestClassifier(LinearSVC(random_state=0))
classifier.fit(X, y)

pickle.dump(classifier, open('classifier_squeezenet.pkl', 'wb'))
y_pred = classifier.predict(X)