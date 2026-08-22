"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Dict

from ._solver import BaseSolver
from .det_solver import DetSolver

TASKS: Dict[str, BaseSolver] = {
    'detection': DetSolver,
}
