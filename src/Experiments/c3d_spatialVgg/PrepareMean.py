'''
Created on Aug 24, 2017

@author: hshi
'''
import sys

import os
caffe_root = '../../../3rd/caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import numpy as np
import ConfigParser

def getSampleNum(videoList):
    return 0

def binaryProto2mat(inPath, outShape):
    mBlob = caffe.proto.caffe_pb2.BlobProto()
    mData = open(inPath , 'rb' ).read()
    mBlob.ParseFromString(mData)
    
    mData = np.array(mBlob.diff)
    mData = mData.reshape(outShape)
    return mData

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
    dataLayerConfigPath = './train.config'
    cf = ConfigParser.ConfigParser()
    cf.read(dataLayerConfigPath)
    
    cropTop = cf.getint('crop', 'cropTop')
    cropBottom = cf.getint('crop', 'cropBottom')
    cropLeft = cf.getint('crop', 'cropLeft')
    cropRight = cf.getint('crop', 'cropRight')
    stackDepth = cf.getint('datalayer', 'stackDepth')
    outChannel = cf.getint('datalayer', 'channel')
    
    
    outHeight, outWidth = getOutFrameSize(dataLayerConfigPath)
    
    
    
    # Prepare Mean and save to numpy
    meanBinaryProtoPath = 'train01_16_128_171_mean.binaryproto'
    rgbMean = binaryProto2mat(meanBinaryProtoPath, [1,3,16,128,171])
    rgbMean = rgbMean[:,:,:,cropTop:cropBottom, cropLeft:cropRight]
    mMean = np.zeros([1, outChannel, stackDepth,outHeight,outWidth])
    mMean[:,0:3, :, :, :] = rgbMean
    mMean[:,3:5, :, :, :] = 128
    mMeanPath = cf.get('normalization', 'meanFilePath')
    np.save(mMeanPath, mMean)