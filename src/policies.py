"""Explicit validation-tuned, causal frame-precision policies."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class Policy:
    name:str
    weights:tuple=(.3,.4,.3)
    scales:tuple=(.20,8.,.25)
    thresholds:tuple=(.30,.65)
    fixed:int=0
    anchor:int=0
    period:int=0

    def assign(self,features):
        output=[];scores=[]
        for i,f in enumerate(features):
            if self.fixed:
                score=0.;bits=self.fixed
            else:
                values=np.asarray([f['edge_raw'],f['motion_raw'],f['difference_raw']])
                score=float(np.dot(np.clip(values/np.asarray(self.scales),0,1),self.weights))
                lo,hi=self.thresholds
                bits=4 if score<lo else 8 if score<hi else 16
                if i==0 and self.anchor:bits=max(bits,self.anchor)
                if self.period and i%self.period==0:bits=16
            output.append(bits);scores.append(score)
        return output,scores

    def to_dict(self):return asdict(self)

def candidates(scales):
    return [Policy('original'),
        Policy('edges_only',(1.,0.,0.),scales,(.40,.85),anchor=8),
        Policy('difference_only',(0.,0.,1.),scales,(.40,.85),anchor=8),
        Policy('motion',(.1,.7,.2),scales,(.30,.75)),
        Policy('difference',(.1,.2,.7),scales,(.30,.75)),
        Policy('balanced',(1/3,1/3,1/3),scales,(.30,.75)),
        Policy('edges',(.7,.1,.2),scales,(.30,.75)),
        Policy('conservative',(.2,.4,.4),scales,(.15,.60),anchor=16),
        Policy('budget',(.2,.4,.4),scales,(.55,.90),anchor=8),
        Policy('motion_anchor',(.1,.7,.2),scales,(.40,.85),anchor=16),
        Policy('difference_anchor',(.1,.2,.7),scales,(.40,.85),anchor=16),
        Policy('periodic',(.2,.4,.4),scales,(.50,.90),period=8)]

def fit_scales(feature_lists):
    values=np.asarray([[f['edge_raw'],f['motion_raw'],f['difference_raw']]
                       for features in feature_lists for f in features])
    return tuple(np.maximum(np.quantile(values,.95,axis=0),[.01,.1,.01]).tolist())
