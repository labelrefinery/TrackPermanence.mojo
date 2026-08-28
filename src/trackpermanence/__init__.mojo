"""TrackPermanence.mojo — pure-Mojo inference for 'Offline Tracking with
Object Permanence' (arXiv:2310.01288): Re-ID of tracklets across occlusions
and completion of the occluded poses, trained in TrackPermanence.py."""

from .csvio import assign_frames, read_tracker_csv, write_tracker_csv
from .features import LocalFrame, completion_frame, hermite_prior, motion_features, time_queries
from .lft import load_lft
from .model import ModelConfig, completion_poses, decode_config, reid_logit
from .plugin import PluginConfig, PluginResult, link_tracklets, run_plugin
from .tensor import Tensor
from .types import BoxState, Tracklet
