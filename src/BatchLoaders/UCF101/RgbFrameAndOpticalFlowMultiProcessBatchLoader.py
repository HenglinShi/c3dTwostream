'''
Created on Aug 24, 2017

@author: hshi
'''
import multiprocessing
import random as rd
from BatchLoaders.UCF101.RgbFrameAndOpticalFlowBatchLoadingProcess import RgbFrameAndOpticalFlowBatchLoadingProcess
from BatchLoaders.UCF101.BatchShedulingProcess import BatchShedulingProcess
class RgbFrameAndOpticalFLowMultiProcessBatchLoader(object):
    '''
    classdocs
    '''


    def __init__(self, 
                 dataLayerConfig):
        '''
        Constructor
        '''
        
        self.videoNumPerBatch = dataLayerConfig.videoNumPerBatch
        self.samplingFrameStackNumPerVideo = dataLayerConfig.samplingFrameStackNumPerVideo
        self.dataDir = dataLayerConfig.dataDir
        self.videoNamesFilePath = dataLayerConfig.videoNamesFilePath
        self.labelNamesFilePath = dataLayerConfig.labelNamesFilePath
        
        
        self.videoNameList = self.loadVideoNamesFile(self.videoNamesFilePath)
        self.labelNames = self.loadLabelNamesFile(self.labelNamesFilePath)
        
        rd.shuffle(self.videoNameList)
        
        self.shedulingQueueSize = dataLayerConfig.shedulingQueueSize
        self.dataBatchQueueSize = dataLayerConfig.dataBatchQueueSize
        
        self.dataBatchQueue = multiprocessing.Queue(self.dataBatchQueueSize)
        self.videoNameBatchQueue = multiprocessing.Queue(self.shedulingQueueSize)
        
        self.batchLoadingProcessNum = dataLayerConfig.batchLoadingProcessNum
        dataLoadingProcesses = []
        
        for i in range(self.batchLoadingProcessNum):
            dataLoadingProcesses.append(RgbFrameAndOpticalFlowBatchLoadingProcess(dataLayerConfig,
                                                                                  self.labelNames,
                 self.videoNameBatchQueue,
                 self.dataBatchQueue))
                                                            
            dataLoadingProcesses[i].start()
        
        
        videoNamesFiller = BatchShedulingProcess(self.videoNameList,
                                                 self.videoNumPerBatch,
                                                 self.videoNameBatchQueue)
        #videoNamesFiller.daemon = True
        videoNamesFiller.start()
        
    def getNextBatch(self):
        
        dataBatch = None
        
        while (dataBatch is None):
            dataBatch = self.dataBatchQueue.get()
            
        return dataBatch['sample'], dataBatch['label'], dataBatch['clipMarker']
    
    

    
    
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
