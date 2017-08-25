'''
Created on Aug 24, 2017

@author: hshi
'''
import sys
import os
from PythonLayers.DataLayers.UCF101.RgbFrameAndOpticalFlowDataLayerConfiguration import RgbFrameAndOpticalFlowDataLayerConfiguration
caffe_root = '../../../../3rd/caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from BatchLoaders.UCF101.RgbFrameAndOpticalFlowMultiProcessBatchLoader import RgbFrameAndOpticalFLowMultiProcessBatchLoader


class RgbFrameAndOpticalFlowDataLayer(caffe.Layer):
    '''
    classdocs
    '''


    def setup(self, bottom, top):
        
        self.top_names = ['data', 'label', 'clipMarker']
        
        
        # Using configParers
        params = eval(self.param_str)
        self.configFilePath = params['configFilePath']

        
        self.layerConfig = RgbFrameAndOpticalFlowDataLayerConfiguration(self.configFilePath)

        self.mBatchLoader = RgbFrameAndOpticalFLowMultiProcessBatchLoader(self.layerConfig)

        
  
        
        top[0].reshape(self.layerConfig.batchSize,  
                       self.layerConfig.frameChannel, 
                       self.layerConfig.stackDepth, 
                       self.layerConfig.outWidth, 
                       self.layerConfig.outWidth)# (N*T) x C x H x W
        
        top[1].reshape(self.layerConfig.videoNumPerBatch, 
                       1) # N x 1
        
        top[2].reshape(self.layerConfig.samplingFrameStackNumPerVideo, 
                       self.layerConfig.videoNumPerBatch) # T x N
    
    def forward(self, bottom, top):
        sampleBatch, labelBatch, clipMarkerBatch = self.mBatchLoader.getNextBatch()

        top[0].data[...] = sampleBatch
        top[1].data[...] = labelBatch
        top[2].data[...] = clipMarkerBatch

    def reshape(self, bottom, top):
        pass
    
    def backward(self, bottom, top):
        pass