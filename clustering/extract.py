#!/usr/bin/env python3
#
# Copyright 2017 Zegami Ltd

"""Preprocess images using Keras pre-trained models."""

import argparse
import csv
import os
import glob

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

import numpy as np
import pandas
from sklearn.cluster import KMeans


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')


parser = argparse.ArgumentParser(prog='Feature extractor')
parser.add_argument('source', default='None', help='Path to the source metadata file')
parser.add_argument('model', default='ResNet50', nargs="?", type=named_model, help='Name of the pre-trained model to use')

pargs = parser.parse_args()

source_dir = os.path.dirname(pargs.source)


def main():
    try:
        # read the source file
        # data = pandas.read_csv(pargs.source, sep='\t')
        image_list = []
        
        feature_list = []
        
        frame_list = [f for f in glob.glob("{}\\*.jpg".format(pargs.source))]
        
        for path in frame_list: #assuming jpg
            if os.path.isfile(path):
                print('File exist: {}'.format(path))
                try:
                    # load image setting the image size to 224 x 224
                    img = image.load_img(path, target_size=(224, 224))
                    
                    # convert image to numpy array
                    x = image.img_to_array(img)
                    
                    # the image is now in an array of shape (3, 224, 224)
                    # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
    
                    # extract the features
                    features = pargs.model.predict(x)[0]
                    feature_np = np.array(features)
                    
                    feature_list.append(feature_np.flatten()) 
                except Exception as ex:
                    # skip all exceptions for now
                    print(ex)
                    pass
                except Exception as ex:
                    # skip all exceptions for now
                    print(ex)
                    pass

        if len (feature_list) >0 :
            feature_list_np = np.array(feature_list)
            kmeans = KMeans(n_clusters=5, random_state=0).fit(feature_list_np)


    except EnvironmentError as e:
        print(e)

if __name__ == '__main__':
    main()
