"""
GeoSemDet -- geometry- and semantics-guided end-to-end detection.

Built on DEIM (https://github.com/ShihuaHuang95/DEIM), which builds on D-FINE and RT-DETR.
"""

# imported for their side effect of populating the component registry
from . import optim
from . import data
from . import deim
from . import extre_module

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)
