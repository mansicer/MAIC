REGISTRY = {}

from .basic_controller import BasicMAC
from .maic_controller import MAICMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['maic_mac'] = MAICMAC
