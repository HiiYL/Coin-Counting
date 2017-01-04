"""
coinCounting.py

YOUR WORKING FUNCTION

"""
import numpy as np


from skimage import data
from skimage.feature import match_template

from skimage import data
from skimage import morphology
from skimage import exposure
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.filters import sobel


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from bottleneck_features import create_googlenet,get_squeezenet


import cv2
import pickle

# images_count = 66
width = 227
height = 227
channel = 3

# model = create_googlenet('googlenet_weights.h5')
model = get_squeezenet(nb_classes=1000,
 path_to_weights='model/squeezenet_weights_th_dim_ordering_th_kernels.h5',
 dim_ordering='th')
# you are allowed to import other Python packages above
##########################

labels = np.array([])


def preprocessImage(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = cv2.resize(image, (width, height)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


coin_lookup_table = { 0 : 0.05, 1 : 0.05, 2 : 0.1, 3 : 0.1, 4 : 0.2, 5: 0.2, 6: 0.5, 7: 0.5}
coin_type_lookup_table = { 0 : "old 5 sen",
 1 :"new 5 sen", 2 : "old 10 sen",
  3 : "new 10 sen", 4 :"old 20 sen", 5: "new 20 sen", 6: "old 50 sen", 7: "new 50 sen"}
classifier = pickle.load(open('classifier_squeezenet.pkl','rb'))


def processAndFindRegions(coin_num,image, invert=False):
    img = cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )


    # cv2.imwrite('{}.jpg'.format(200 + coin_num), img)

    img = cv2.medianBlur(img,3)

    # cv2.imwrite('{} - blurred.jpg'.format(200 + coin_num), img)

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=20,maxRadius=50)

    total_value = 0

    circles = np.uint16(np.around(circles))

    modified_image = image.copy()
    for label, i in enumerate(circles[0,:]):

        cv2.circle(modified_image,(i[0],i[1]),i[2],(0,255,0),2)

        x = i[0]
        y = i[1]
        radius = i[2]

        cropped_image = image[ y - radius : y + radius , x - radius : x + radius ]



        image_processed = preprocessImage(cropped_image)
        out = model.predict(image_processed)
        coin = np.argmax(classifier.predict(out))
        value = coin_lookup_table[coin]
        # print("Value of coin #{} of image #{} is [ {} - {} ] ".format(label, coin_num, coin, coin_type_lookup_table[coin]))

        # cv2.imshow('detected circles',  cropped_image)
        # cv2.waitKey(0)

        # cv2.imwrite('crops/{} - {}.jpg'.format(200 + coin_num, 100 + label),cropped_image)

        cv2.circle(modified_image,(i[0],i[1]),2,(0,0,255),3)
        # global labels
        # labels = np.append(labels,coin)
        total_value += value

    # cv2.imshow('detected circles',  modified_image)
    # cv2.waitKey(0)

    cv2.imwrite('{}.jpg'.format(200 + coin_num), modified_image)

    # np.savetxt('coins.txt', labels)

    return total_value





def coinCount(coinMat, i):
    # Inputs
    # coinMat: 4-D numpy array of row*col*3*numImages, 
    #          numImage denote number of images in coin set (10 in this case)
    # i: coin set image number (1, 2, ... 10)
    # Output
    # ans: Total value of the coins in the image, in float type
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    image = coinMat[:,:,:,i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    total_value = processAndFindRegions(i, image, invert=False)

    ans = total_value
    # END OF YOUR CODE
    #########################################################################
    return round(ans,2)