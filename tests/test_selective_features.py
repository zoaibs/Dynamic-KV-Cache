import json
import numpy as np
import pytest
from src.information_score import InformationScorer
from src.selective_features import SelectiveScorer
from scripts.resources import ROOT

@pytest.mark.parametrize('active',[(True,False,False),(False,False,True),(True,True,True)])
def test_selected_feature_definitions_match_reference(active):
    config=json.loads((ROOT/'configs/default.json').read_text())['score']
    a=InformationScorer(config);b=SelectiveScorer(config,active)
    rng=np.random.default_rng(44)
    for _ in range(3):
        image=rng.integers(0,256,(120,160,3),dtype=np.uint8)
        x,y=a(image),b(image)
        for flag,key in zip(active,['edge_raw','motion_raw','difference_raw']):
            if flag:assert getattr(x,key)==getattr(y,key)
