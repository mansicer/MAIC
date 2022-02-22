REGISTRY = {}

from .rnn_agent import RNNAgent
from .maic_agent import MAICAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY['maic'] = MAICAgent
