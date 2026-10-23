"""Exactly Canny edge density, Farneback magnitude and grayscale difference."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass(frozen=True)
class Information:
    edge_raw: float
    motion_raw: float
    difference_raw: float
    edge: float
    motion: float
    difference: float
    score: float
    first_frame: bool
    missing: bool = False

    def to_dict(self):
        return asdict(self)

class InformationScorer:
    def __init__(self, config):
        self.config = config
        for field in ('edge_scale', 'motion_scale', 'difference_scale'):
            if config[field] <= 0:
                raise ValueError('Scales must be positive')
        w = np.asarray(config['weights'], dtype=float)
        if len(w) != 3 or np.any(w < 0) or not np.isclose(w.sum(), 1):
            raise ValueError('Three nonnegative weights must sum to one')
        self.previous = None

    def reset(self):
        self.previous = None

    def __call__(self, frame):
        if frame is None:
            # A missing observation never advances the temporal reference.
            return Information(0, 0, 0, 0, 0, 0, 0, self.previous is None, True)
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError('Expected uint8 grayscale or BGR image')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.resize(gray, (self.config['width'], self.config['height']),
                          interpolation=cv2.INTER_AREA)
        edge = float(np.count_nonzero(cv2.Canny(gray, *self.config['canny'])) / gray.size)
        first = self.previous is None
        motion = difference = 0.0
        if not first:
            flow = cv2.calcOpticalFlowFarneback(self.previous, gray, None,
                                               0.5, 3, 15, 3, 5, 1.2, 0)
            motion = float(np.linalg.norm(flow, axis=-1).mean())
            difference = float(np.abs(gray.astype(np.float32) -
                                      self.previous.astype(np.float32)).mean() / 255)
        self.previous = gray.copy()
        features = [float(np.clip(value / self.config[field], 0, 1))
                    for value, field in zip((edge, motion, difference),
                                           ('edge_scale', 'motion_scale', 'difference_scale'))]
        score = float(np.dot(features, self.config['weights']))
        return Information(edge, motion, difference, *features, score, first)
