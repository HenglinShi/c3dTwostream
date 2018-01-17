'''
Created on Aug 24, 2017

@author: hshi
'''
import sys
import datetime
import shutil
import os
from PythonLayers.DataLayers.UCF101.RgbFrameAndOpticalFlowDataLayerConfiguration import RgbFrameAndOpticalFlowDataLayerConfiguration
from BatchLoaders.UCF101.RgbFrameAndOpticalFlowMultiProcessBatchLoader import RgbFrameAndOpticalFLowMultiProcessBatchLoader
caffe_root = '../../../3rd/caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cPickle
import numpy as np
from caffe.proto import caffe_pb2
import multiprocessing

import random as rd

from skimage.io import imshow, imread
from skimage.transform import resize as imresize
import ConfigParser


def loadVideoNamesFile(videoNamesFilePath):
    with open(videoNamesFilePath, 'r') as videoNamesFile:
        videoNames = videoNamesFile.read()
    
    videoNames = videoNames.split('\r')
    videoNames = filter(None, videoNames)
    return videoNames
    
def getSampleNum(videoNamesFilePath):
    videoNames = loadVideoNamesFile(videoNamesFilePath)
    videoNum = len(videoNames)
    return videoNum


def getOutFrameSize(configPath):
    cf = ConfigParser.ConfigParser()
    cf.read(configPath)
    outHeight = 0
    outWidth = 0
    
    toResize = cf.getboolean('resize', 'enable')
    if toResize:
        resizeHeight = cf.getint('resize', 'height')
        resizeWidth = cf.getint('resize', 'width')
        
        outHeight = resizeHeight
        outWidth = resizeWidth
        
    toCrop = cf.getboolean('crop', 'enable')
    if toCrop:
        cropHeight = cf.getint('crop', 'cropBottom') - cf.getint('crop', 'cropTop')
        cropWidth = cf.getint('crop', 'cropRight') - cf.getint('crop', 'cropLeft')
        
        outHeight = cropHeight
        outWidth = cropWidth
        
    return outHeight, outWidth
def main():
    

    # Prepare the mean file
    configPath_train = './train.config'
    configPath_test = './test.config'
    solverPath = 'c3d_ucf101_finetuning_solver.prototxt'
    weightPath = './twoStreamInit.caffemodel'
    
    
    config_train = RgbFrameAndOpticalFlowDataLayerConfiguration(configPath_train)
    config_test = RgbFrameAndOpticalFlowDataLayerConfiguration(configPath_test)
    
    maxTrainingEpoches = config_train.maxTrainingEpoches
    videoNumPerBatch_train = config_train.videoNumPerBatch
    videoNumPerBatch_test = config_test.videoNumPerBatch
    sampleNum_train = getSampleNum(config_train.videoNamesFilePath)
    sampleNum_test = getSampleNum(config_test.videoNamesFilePath)


    batchNumPerEpoch_train = np.ceil(sampleNum_train * 1.0/videoNumPerBatch_train)
    batchNumPerEpoch_test = np.ceil(sampleNum_test *1.0/videoNumPerBatch_test)
    
    losses_train = np.zeros(maxTrainingEpoches)
    losses_test = np.zeros(maxTrainingEpoches)
    
    accuracies_train = np.zeros(maxTrainingEpoches)
    accuracies_test = np.zeros(maxTrainingEpoches)



    solver = None
    solver = caffe.SGDSolver(solverPath)
    #solver.net.copy_from(weightPath)
    
    
    
    batchLoader_test = RgbFrameAndOpticalFLowMultiProcessBatchLoader(config_test)
    
    



    

    

    

    
    for i in range(maxTrainingEpoches):
        
        pred = np.zeros(batchNumPerEpoch_test * videoNumPerBatch_test)
        gt = np.ones(batchNumPerEpoch_test * videoNumPerBatch_test)
    
        for j in range(batchNumPerEpoch_test):
            sampleBatch_test, \
            labelBatch_test, \
            clipMarkerBatch_test = batchLoader_test.getNextBatch()
            
            solver.net.blobs['data'].data[...] = sampleBatch_test
            solver.net.blobs['label'].data[...] = labelBatch_test
            solver.net.blobs['cm'].data[...] = clipMarkerBatch_test
            solver.net.forward(start='conv1a_new') #change
            
            print solver.net.blobs['fc8'].data.argmax(1)
            print solver.net.blobs['label'].data.reshape([-1])
            losses_test[i] += solver.net.blobs['loss'].data
            pred[j * videoNumPerBatch_test : videoNumPerBatch_test*(j+1)] = solver.net.blobs['fc8'].data.argmax(1)
            gt[j * videoNumPerBatch_test : videoNumPerBatch_test*(j+1)] = solver.net.blobs['label'].data.reshape([-1])
        
        pred = pred[0:sampleNum_test]
        gt = gt[0:sampleNum_test]
        
        accuracies_test[i] = (sum(pred==gt) * 1.0)/sampleNum_test
        print accuracies_test[i]
    
        batchLoader_test.reset()
    
    
    
        for j in range(batchNumPerEpoch_train):
            solver.step(1)
            print solver.net.blobs['fc8'].data.argmax(1)
            print solver.net.blobs['label'].data.reshape([-1])
            print solver.net.blobs['loss'].data
            losses_train[i] += solver.net.blobs['loss'].data






if __name__ == '__main__':
    main()