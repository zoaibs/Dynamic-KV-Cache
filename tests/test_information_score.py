import json
import cv2
import numpy as np
import pytest
from scripts.resources import ROOT
from src.information_score import InformationScorer
from src.precision_scheduler import precision_for_score

@pytest.fixture
def config():
    return json.loads((ROOT / 'configs/default.json').read_text())['score']

def test_first_missing_and_static(config):
    s = InformationScorer(config)
    assert s(None).missing and s.previous is None
    black = np.zeros((120, 160), np.uint8)
    first = s(black)
    assert first.first_frame and first.score == first.motion == first.difference == 0
    assert s(None).missing
    assert s(black).score == 0

def test_canny_matches_pixel_fraction(config):
    im = np.zeros((120, 160), np.uint8)
    im[30:90, 40:120] = 255
    result = InformationScorer(config)(im)
    expected = np.count_nonzero(cv2.Canny(im, 100, 200)) / im.size
    assert result.edge_raw == expected
    assert result.edge == min(1, expected / config['edge_scale'])

def test_difference_and_clipping(config):
    s = InformationScorer(config)
    s(np.zeros((120, 160), np.uint8))
    r = s(np.full((120, 160), 255, np.uint8))
    assert r.difference_raw == r.difference == 1
    assert 0 <= r.score <= 1

def test_flow_on_translated_texture(config):
    rng = np.random.default_rng(123)
    a = cv2.GaussianBlur(rng.integers(0, 256, (120, 160), dtype=np.uint8), (5, 5), 0)
    b = np.roll(a, 3, axis=1)
    s = InformationScorer(config)
    s(a)
    r = s(b)
    assert r.motion_raw > 1.5
    assert r.score == pytest.approx(0.3*r.edge + 0.4*r.motion + 0.3*r.difference)

@pytest.mark.parametrize('score,bits', [(0,4),(.299999,4),(.30,8),(.649999,8),(.65,16),(1,16)])
def test_thresholds(score, bits):
    assert precision_for_score(score) == bits

@pytest.mark.parametrize('score', [float('nan'), float('inf'), -1, 1.01])
def test_invalid_scores(score):
    with pytest.raises(ValueError):
        precision_for_score(score)

def test_invalid_config(config):
    with pytest.raises(ValueError):
        InformationScorer(dict(config, motion_scale=0))
    with pytest.raises(ValueError):
        InformationScorer(dict(config, weights=[1,1,1]))
