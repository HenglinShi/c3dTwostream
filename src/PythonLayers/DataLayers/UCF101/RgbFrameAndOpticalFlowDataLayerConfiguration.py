'''
Created on Aug 24, 2017

@author: hshi
'''
import ConfigParser
class RgbFrameAndOpticalFlowDataLayerConfiguration(object):
    '''
    classdocs
    '''


    def __init__(self, configFilePath):
        '''
        Constructor
        '''
        self.configFilePath = configFilePath
        cf = ConfigParser.ConfigParser()
        cf.read(self.configFilePath)
        
        self.dataDir = cf.get("config", "dataDir")
        self.videoNamesFilePath = cf.get("config", "videoNamesFilePath")
        self.labelNamesFilePath = cf.get("config", "labelNamesFilePath")
        
        self.samplingFrameStackNumPerVideo = cf.getint("config", "samplingFrameStackNumPerVideo")
        self.stackDepth = cf.getint("config", "stackDepth")
        
        self.frameChannel = cf.getint("config", "channel")
        self.outHeight = cf.getint("config", "srcHeight")
        self.outWidth = cf.getint("config", "srcWidth")
        
        self.enableResize = cf.getboolean('config', 'enableResize')
        if self.enableResize:
            #self.cropMode = cf.getboolean('config', 'cropMode')
            self.resizeHeight = cf.getint('config', 'resizeHeight')
            self.resizeWidth = cf.getint('config', 'resizeWidth')
            
            self.outHeight = self.resizeHeight
            self.outWidth = self.resizeWidth
            
        self.enableCrop = cf.getboolean('config', 'enableCrop')
        
        if self.enableCrop:
            
            self.cropTop = cf.getint('config', 'cropTop')
            self.cropBottom = cf.getint('config', 'cropBottom')
            self.cropLeft = cf.getint('config', 'cropLeft')
            self.cropRight = cf.getint('config', 'cropRight')
            
            self.outHeight = self.cropBottom - self.cropTop
            self.outWidth = self.cropRight - self.cropLeft
            
        
        self.videoNumPerBatch = cf.getint("config", "videoNumPerBatch")
       
        self.shedulingQueueSize = cf.getint("config", "shedulingQueueSize")
        self.dataBatchQueueSize = cf.getint("config", "dataBatchQueueSize")
        self.batchLoadingProcessNum = cf.getint("config", "batchLoadingProcessNum")
    
    
        self.batchSize = self.videoNumPerBatch * self.samplingFrameStackNumPerVideo
        
        self.enableMeanNormalization = cf.getboolean('config', 'enableMeanNormalization')
        self.meanFilePath = cf.get('config', 'meanFilePath')
        
        self.enableScaleNormalization = cf.getboolean('config', 'enableScaleNormalization')
        self.scaleFilePath = cf.get('config', 'scaleFilePath')
        
        self.maxTrainingEpoches = cf.getint('config', 'maxTrainingEpoches')