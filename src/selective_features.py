"""Compute only the features consumed by a frozen policy, preserving definitions."""
import cv2
import numpy as np
from src.information_score import InformationScorer,Information

class SelectiveScorer(InformationScorer):
    def __init__(self,config,active):
        super().__init__(config);self.active=active
    def __call__(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
        gray=cv2.resize(gray,(self.config['width'],self.config['height']),interpolation=cv2.INTER_AREA)
        first=self.previous is None
        edge=motion=difference=0.
        if self.active[0]:edge=float(np.count_nonzero(cv2.Canny(gray,*self.config['canny']))/gray.size)
        if not first and self.active[1]:
            flow=cv2.calcOpticalFlowFarneback(self.previous,gray,None,.5,3,15,3,5,1.2,0)
            motion=float(np.linalg.norm(flow,axis=-1).mean())
        if not first and self.active[2]:
            difference=float(np.abs(gray.astype(np.float32)-self.previous.astype(np.float32)).mean()/255)
        self.previous=gray.copy()
        scaled=[float(np.clip(x/self.config[k],0,1)) for x,k in zip((edge,motion,difference),
                ('edge_scale','motion_scale','difference_scale'))]
        return Information(edge,motion,difference,*scaled,float(np.dot(scaled,self.config['weights'])),first)
