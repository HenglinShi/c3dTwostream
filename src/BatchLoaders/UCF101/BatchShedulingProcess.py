'''
Created on Aug 24, 2017

@author: hshi
'''
import multiprocessing
import random as rd

class BatchShedulingProcess(multiprocessing.Process):
    '''
    classdocs
    '''


    def __init__(self, 
                 videoNameList,
                 videoNumPerBatch,
                 videoNameBatchQueue):
        '''
        Constructor
        '''
        
        multiprocessing.Process.__init__(self)
        self.videoNameBatchQueue = videoNameBatchQueue
        #self.dataBatchQueue = dataBatchQueue
        self.videoNameList = videoNameList
        self.videoNumPerBatch = videoNumPerBatch
        
        self.videoNum = len(self.videoNameList)
        self.currentVideoInd = 0
        
    def run(self):
        while True:
            currentVideoNamesBatch = list()
            
            for _ in range(self.videoNumPerBatch):
                if self.currentVideoInd == self.videoNum:
                    rd.shuffle(self.videoNameList)
                    self.currentVideoInd = 0
                
                currentVideoNamesBatch.append(self.videoNameList[self.currentVideoInd])
                
                self.currentVideoInd += 1
                
            self.videoNameBatchQueue.put(currentVideoNamesBatch)
 
 

        