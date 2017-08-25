'''
Created on Aug 24, 2017

@author: hshi
'''
import numpy as np
import sys
import os
caffe_root = '../../3rd/caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

class PermuteLayer(caffe.Layer):
    '''
    classdocs
    '''


    def setup(self, bottom, top):
        
        self.top_names = ['top']
        
        params = eval(self.param_str)
        self.permuteIndex = np.asarray(params['permuteIndex'])
        self.bottomShape = bottom[0].data.shape
        
        if len(self.permuteIndex) == 2:
            top[0].reshape(self.bottomShape[self.permuteIndex[0]], 
                           self.bottomShape[self.permuteIndex[1]])
            
        elif len(self.permuteIndex) == 3:
            top[0].reshape(self.bottomShape[self.permuteIndex[0]], 
                           self.bottomShape[self.permuteIndex[1]],
                           self.bottomShape[self.permuteIndex[2]])
            
        elif len(self.permuteIndex) == 4:
            top[0].reshape(self.bottomShape[self.permuteIndex[0]],
                           self.bottomShape[self.permuteIndex[1]],
                           self.bottomShape[self.permuteIndex[2]],
                           self.bottomShape[self.permuteIndex[3]])
            
        else: 
            pass
        
    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data.transpose(self.permuteIndex)

    def reshape(self, bottom, top):
        pass
    
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff.transpose(self.permuteIndex)
        


