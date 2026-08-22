"""
Detector components.

Upstream: DEIM (Huang et al., 2025) / D-FINE (Peng et al., 2024) / RT-DETR (Zhao et al., 2024).
GeoSemDet adds GSA (geometry-steered attention) and SRG (semantic reliability gating)
on top of the D-FINE decoder.
"""

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder, SimpleEncoder

from .dfine_decoder import DFINETransformer
from .gsa_decoder import GSATransformer          # + GSA
from .srg_decoder import SRGTransformer          # + SRG
from .geosemdet_decoder import GeoSemDetTransformer  # + GSA + SRG (the released model)

from .text_adapter import TextAdapter

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion
