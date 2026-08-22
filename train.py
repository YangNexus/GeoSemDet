"""
GeoSemDet training / evaluation entry point.

    python train.py -c configs/geosemdet_pcb.yml
    python train.py -c configs/geosemdet_concrete12.yml --test-only -r <checkpoint>

Derived from DEIM (Copyright (c) 2024 The DEIM Authors), itself modified from
RT-DETR (Copyright (c) 2023 lyuwenyu). Both Apache-2.0; see NOTICE.
"""
import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # select GPUs, e.g. "0,2,3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse

from engine.logger_module import get_logger
from engine.extre_module.torch_utils import check_cuda
from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    check_cuda()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    cfg_str = json.dumps(cfg.__dict__, indent=4, ensure_ascii=False)
    print(GREEN + cfg_str + RESET)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        if args.path:
            solver.val_pt_onnx_engine(args.mode)
        else:
            solver.val()
    else:
        solver.fit(cfg_str)

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    parser.add_argument('-p', '--path', type=str, help='onnx/engine or prune-model path in test-only')
    parser.add_argument('-m', '--mode', type=str, default='det', choices=['det', 'mask'], help='engine model mode(det/mask) in test-only')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
