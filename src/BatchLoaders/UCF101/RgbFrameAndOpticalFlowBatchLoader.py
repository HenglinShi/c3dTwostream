'''
Created on Aug 24, 2017

@author: hshi
'''
import os
import random as rd
import numpy as np
from skimage.io import imshow, imread
from skimage.transform import resize as imresize

class RgbFrameAndOpticalFlowBatchLoader(object):
    '''
    classdocs
    '''


    def __init__(self, 
                 videoNumPerBatch, 
                 samplingFrameStackNumPerVideo, 
                 dataDir, 
                 videoNamesFilePath, 
                 labelNamesFilePath,
                 stackDepth,
                 frameHeight, 
                 frameWidth,
                 frameChannel):
        
        self.videoNumPerBatch = videoNumPerBatch
        self.samplingFrameStackNumPerVideo = samplingFrameStackNumPerVideo
        self.dataDir = dataDir
        self.rgbDataDir = os.path.join(self.dataDir, 'jpegs_256')
        self.opticalFlowDir_u = os.path.join(self.dataDir, 'tvl1_flow', 'u')
        self.opticalFlowDir_v = os.path.join(self.dataDir, 'tvl1_flow', 'v')
        
        self.videoNamesFilePath = videoNamesFilePath
        self.labelNamesFilePath = labelNamesFilePath
        self.stackDepth = stackDepth
        self.frameHeight = frameHeight
        self.frameWidth = frameWidth
        self.frameChannel = 5
        
        

        self.batchSize = self.videoNumPerBatch * self.samplingFrameStackNumPerVideo
        
        self.videoNames = self.loadVideoNamesFile()
        self.labelNames = self.loadLabelNamesFile()
        self.videoNum = len(self.videoNames)
        self.labelNum = len(self.labelNames)
        
        self.currentVideoInd = 0
        rd.shuffle(self.videoNames)
        
        
        self.halfStackDepth = (self.stackDepth/2)
        
        
        
        self.sampleBatch = np.zeros(shape = (self.batchSize, 
                                             self.frameChannel,
                                             self.stackDepth, 
                                             self.frameHeight, 
                                             self.frameWidth))
        
        self.labelBatch = np.zeros(shape=(self.videoNumPerBatch,
                                          1))
        
        self.clipMarkerBatch = np.ones(shape=(self.samplingFrameStackNumPerVideo, 
                                               self.videoNumPerBatch))
        self.clipMarkerBatch[0,:] = 0
        

    
    def shuffleFileList(self):
        rd.shuffle(self.videoNames)
    def resetCurrentVideoInd(self):
        self.currentVideoInd = 0
    
    def reset(self):
        self.shuffleFileList()
        self.resetCurrentVideoInd()
        
    def getNextBatch(self):
        for i in range(self.videoNumPerBatch):
            if self.currentVideoInd == self.videoNum:
                self.currentVideoInd = 0
                rd.shuffle(self.videoNames)
                
            currentVideoName = self.videoNames[self.currentVideoInd]
            
            currentVideoRgbFramesDirPath = os.path.join(self.rgbDataDir, currentVideoName)
            currentVideoOpticalFlowDirPath_u = os.path.join(self.opticalFlowDir_u, currentVideoName)
            currentVideoOpticalFlowDirPath_v = os.path.join(self.opticalFlowDir_v, currentVideoName)
            
            
            self.labelBatch[i] = self.labelNames.index(currentVideoName[2:-8])
                
            currentVideoFrameNames = os.listdir(currentVideoRgbFramesDirPath)
            currentVideoFrameNames.sort()
            frameNumOfCurrentVideo = len(currentVideoFrameNames) - 1
            
            sampleFrameLocs = np.linspace(0, frameNumOfCurrentVideo - 1, self.stackDepth * self.samplingFrameStackNumPerVideo, dtype='int')
            
            for stackIte in range(self.samplingFrameStackNumPerVideo):
                

                stackFrameInds = sampleFrameLocs[stackIte * self.stackDepth : (stackIte + 1) * self.stackDepth]
                
                for frameIte in range(self.stackDepth):
                    currentFrame = imread(os.path.join(currentVideoRgbFramesDirPath, currentVideoFrameNames[stackFrameInds[frameIte]]))
                                        
                    currentFrame = imresize(currentFrame.astype('float64'), [self.frameHeight, self.frameWidth])
                    self.sampleBatch[i + stackIte * self.videoNumPerBatch, 0:3, frameIte, :, :] = currentFrame.transpose([2,0,1])
            

                    flow_u = imread(os.path.join(currentVideoOpticalFlowDirPath_u, 'frame' + currentVideoFrameNames[stackFrameInds[frameIte]]))
                    flow_u = imresize(flow_u.astype('float64'), [self.frameHeight, self.frameWidth])
                    flow_u = flow_u - 128
                            
                    flow_v = imread(os.path.join(currentVideoOpticalFlowDirPath_v, 'frame' + currentVideoFrameNames[stackFrameInds[frameIte]]))
                    flow_v = imresize(flow_v.astype('float64'), [self.frameHeight, self.frameWidth])
                    flow_v = flow_v - 128
                            
                    self.sampleBatch[i + stackIte * self.videoNumPerBatch, 3, frameIte, :, :] = flow_u
                    self.sampleBatch[i + stackIte * self.videoNumPerBatch, 4, frameIte, :, :] = flow_v
                            
                    #print i, stackIte,i + stackIte * self.videoNumPerBatch, frameIte

                
            self.currentVideoInd += 1
            
        return self.sampleBatch, self.labelBatch, self.clipMarkerBatch

    def getVideoNum(self):
        return self.videoNum
    
    
    def loadLabelNamesFile(self, labelNamesFilePath = None):
        
        if labelNamesFilePath is None:
            labelNamesFilePath = self.labelNamesFilePath
            
        with open(labelNamesFilePath, 'r') as labelNamesFile:
            labelNames = labelNamesFile.read()
        
        labelNames = labelNames.split('\r')
        labelNames = filter(None, labelNames)
        return labelNames
        
    def loadVideoNamesFile(self, videoNamesFilePath = None):
        
        if videoNamesFilePath is None:
            videoNamesFilePath = self.videoNamesFilePath 
            
        with open(videoNamesFilePath, 'r') as videoNamesFile:
            videoNames = videoNamesFile.read()
        
        videoNames = videoNames.split('\r')
        videoNames = filter(None, videoNames)
        
        
        return videoNames   
        