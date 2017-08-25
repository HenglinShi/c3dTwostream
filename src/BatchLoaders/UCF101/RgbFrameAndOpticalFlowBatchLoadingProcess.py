'''
Created on Aug 24, 2017

@author: hshi
'''
import os
import multiprocessing
import numpy as np
from skimage.io import imshow, imread
from skimage.transform import resize as imresize
import scipy.io as sio

class RgbFrameAndOpticalFlowBatchLoadingProcess(multiprocessing.Process):
    '''
    classdocs
    '''


    def __init__(self, 
                 dataLayerConfig,
                 labelNames,
                 videoNameBatchQueue,
                 dataBatchQueue):
        '''
        Constructor
        '''
        multiprocessing.Process.__init__(self)
        
        self.videoNameBatchQueue = videoNameBatchQueue
        self.dataBatchQueue = dataBatchQueue

        self.dataDir = dataLayerConfig.dataDir
        self.rgbDataDir = os.path.join(self.dataDir, 'jpegs_256')
        self.opticalFlowDir_u = os.path.join(self.dataDir, 'tvl1_flow', 'u')
        self.opticalFlowDir_v = os.path.join(self.dataDir, 'tvl1_flow', 'v')
        
        
        self.frameHeight_out = dataLayerConfig.outHeight
        self.frameWidth_out = dataLayerConfig.outWidth
        self.stackDepth = dataLayerConfig.stackDepth
        self.frameChannel = dataLayerConfig.frameChannel
        
        
        self.videoNumPerBatch = dataLayerConfig.videoNumPerBatch
        self.samplingFrameStackNumPerVideo = dataLayerConfig.samplingFrameStackNumPerVideo
        
        
        self.labelNames = labelNames
        self.labelTypeNum = len(self.labelNames)
        
        
        self.enableMeanNormalization = dataLayerConfig.enableMeanNormalization
        if self.enableMeanNormalization:
            self.mean = np.load(dataLayerConfig.meanFilePath)
        
        self.enableScaleNormalization = dataLayerConfig.enableScaleNormalization
        if self.enableScaleNormalization:
            self.scale = np.load(dataLayerConfig.scaleFilePath)
        
        self.enableResize = dataLayerConfig.enableResize
        if self.enableResize:
            self.resizeHeight = dataLayerConfig.resizeHeight
            self.resizeWidth = dataLayerConfig.resizeWidth
            
        self.enableCrop = dataLayerConfig.enableCrop
        if self.enableCrop:
            
            self.cropTop = dataLayerConfig.cropTop
            self.cropBottom = dataLayerConfig.cropBottom
            self.cropLeft = dataLayerConfig.cropLeft
            self.cropRight = dataLayerConfig.cropRight
        
        self.batchSize = dataLayerConfig.batchSize
        
        self.sampleBatch = np.zeros(shape = (self.batchSize, 
                                             self.frameChannel,
                                             self.stackDepth, 
                                             self.frameHeight_out, 
                                             self.frameWidth_out))
        

        self.labelBatch = np.zeros(shape=(self.videoNumPerBatch, 1))
        
        self.clipMarkerBatch = np.ones(shape=(self.samplingFrameStackNumPerVideo, self.videoNumPerBatch))
        
        self.clipMarkerBatch[0,:] = 0
        
    def run(self):
        
        while True:
            currentBatchVideoNames = self.videoNameBatchQueue.get()
            
            if currentBatchVideoNames is not None:
                for videoIte in range(len(currentBatchVideoNames)):
                    currentVideoName = currentBatchVideoNames[videoIte]
                    self.labelBatch[videoIte] = self.labelNames.index(currentVideoName[2:-8])
                    
                    currentVideoRgbFramesDirPath = os.path.join(self.rgbDataDir, currentVideoName)
                    currentVideoOpticalFlowDirPath_u = os.path.join(self.opticalFlowDir_u, currentVideoName)
                    currentVideoOpticalFlowDirPath_v = os.path.join(self.opticalFlowDir_v, currentVideoName)
                    
                    #currentVideoFramesDirPath = os.path.join(self.dataDir, currentVideoName)
                
                    currentVideoFrameNames = os.listdir(currentVideoRgbFramesDirPath)
                    currentVideoFrameNames.sort()
                    
                    frameNumOfCurrentVideo = len(currentVideoFrameNames) - 1
            
                    sampleFrameLocs = np.linspace(0, frameNumOfCurrentVideo - 1, self.stackDepth * self.samplingFrameStackNumPerVideo, dtype='int')
            
                    for stackIte in range(self.samplingFrameStackNumPerVideo):
                

                        stackFrameInds = sampleFrameLocs[stackIte * self.stackDepth : (stackIte + 1) * self.stackDepth]
                
                        for frameIte in range(self.stackDepth):
                            currentFrame = imread(os.path.join(currentVideoRgbFramesDirPath, currentVideoFrameNames[stackFrameInds[frameIte]]))
                            flow_u = imread(os.path.join(currentVideoOpticalFlowDirPath_u, currentVideoFrameNames[stackFrameInds[frameIte]]))
                            flow_v = imread(os.path.join(currentVideoOpticalFlowDirPath_v, currentVideoFrameNames[stackFrameInds[frameIte]]))
                                
                            if self.enableResize:
                                currentFrame = imresize(currentFrame.astype('float64'), [self.resizeHeight, self.resizeWidth])
                                flow_u = imresize(flow_u.astype('float64'), [self.resizeHeight, self.resizeWidth])
                                flow_v = imresize(flow_v.astype('float64'), [self.resizeHeight, self.resizeWidth])
                             
                            if self.enableCrop:
                                currentFrame = currentFrame[self.cropTop:self.cropBottom,self.cropLeft:self.cropRight,:]    
                                flow_u = flow_u[self.cropTop:self.cropBottom,self.cropLeft:self.cropRight]  
                                flow_v = flow_v[self.cropTop:self.cropBottom,self.cropLeft:self.cropRight]  
                                
                            self.sampleBatch[videoIte + stackIte * self.videoNumPerBatch, 0:3, frameIte, :, :] = currentFrame.transpose([2,0,1])
                            self.sampleBatch[videoIte + stackIte * self.videoNumPerBatch, 3, frameIte, :, :] = flow_u
                            self.sampleBatch[videoIte + stackIte * self.videoNumPerBatch, 4, frameIte, :, :] = flow_v
            
                            
                            
            
                if self.enableMeanNormalization:
                    self.sampleBatch -= self.mean
                            
                if self.enableScaleNormalization:
                    self.sampleBatch /= self.scale
                            
                            


                        
                
                self.dataBatchQueue.put({'sample': self.sampleBatch,
                                         'label': self.labelBatch,
                                         'clipMarker': self.clipMarkerBatch})
            
  
   
