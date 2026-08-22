"""
Model assembly for GeoSemDet.

``DEIMGraph`` builds a detector from a YOLO-style layer graph declared in a YAML file
(``backbone`` / ``encoder`` / ``decoder`` sections); ``GeoSemDet`` extends it with the
cached category-text bank consumed by SRG and routes the depth prior to GSA.
"""

import re, yaml, contextlib
from pathlib import Path
from functools import partial
from typing import Optional
from ..core import register
from ..misc.dist_utils import is_dist_available_and_initialized
from ..backbone.common import FrozenBatchNorm2d

import torch
import torch.nn as nn

from engine.logger_module import get_logger
from engine.misc.text_bank import load_text_bank, summarize_text_bank
from engine.deim.text_adapter import TextAdapter

from engine.backbone.hgnetv2 import StemBlock, HG_Stage, filter_loadable_state_dict
from engine.deim.hybrid_encoder import ConvNormLayer_fuse, SCDown, TransformerEncoderBlock
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.gsa_decoder import GSATransformer
from engine.deim.srg_decoder import SRGTransformer
from engine.deim.geosemdet_decoder import GeoSemDetTransformer

from engine.extre_module.ultralytics_nn.conv import Concat, Conv
from engine.extre_module.custom_nn.module.mambaout import MambaOut
from engine.extre_module.custom_nn.module.dsm import DSM
from engine.extre_module.custom_nn.module.srg_gate import SRGEncoderGate

RED, GREEN, BLUE, YELLOW, ORANGE, CYAN, MAGENTA, LAVENDER, GOLD, RESET = "[91m", "[92m", "[94m", "[93m", "[38;5;208m", "[96m", "[95m", "[38;5;147m", "[38;5;220m", "[0m"
logger = get_logger(__name__)

__all__ = ['DEIMGraph', 'GeoSemDet']

@register(force=True)  # force: re-importing this module must not double-register
class DEIMGraph(nn.Module):   
    __share__ = ['num_classes', 'eval_spatial_size']
    def __init__(self, \
        yaml_path,   
        pretrained=None,
        freeze_stem_only=False,
        freeze_at=-1,
        freeze_norm=False,
        num_classes=80,   
        eval_spatial_size=(640, 640)  
    ):
        super().__init__()  
        d = yaml_load(yaml_path)    
        backbone, encoder, decoder, self.save = parse_model(d, ch=3, nc=num_classes, eval_spatial_size=eval_spatial_size, verbose=True)    
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder  

        # print(self.backbone.state_dict().keys())    
     
        if freeze_at >= 0:  
            self._freeze_parameters(self.backbone[0])
            if not freeze_stem_only:     
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i + 1]) 
 
        if freeze_norm:
            self._freeze_norm(self.backbone)  
 
        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                state = torch.load(pretrained, map_location='cpu')
                logger.info(f"Loaded stage1 {pretrained} HGNetV2 from local file.")

                filtered_state = filter_loadable_state_dict(self.backbone.state_dict(), state)
                logger.info(RED + f'Loading Pretrained State Dict Key Names:{filtered_state.keys()}' + RESET)
   
                self.backbone.load_state_dict(filtered_state, strict=False)  

            except (Exception, KeyboardInterrupt) as e:    
                if (is_dist_available_and_initialized() and torch.distributed.get_rank() == 0) or (not is_dist_available_and_initialized()): 
                    logger.error(f"Loading Backbone Pretrained Weight Error. Message:{str(e)}")
                exit()
    
    def forward(self, x, targets=None):
        depth_prior = None
        if isinstance(x, dict) and getattr(self.decoder, "supports_depth_prior", False):
            depth_prior = x.get("depth_geom", x.get("raw_depth_aligned", x.get("raw_npy", x.get("npy"))))
            x = x["rgb"]

        y = [] 
        for idx, m in enumerate(list(self.backbone.children()) + list(self.encoder.children())):     
            # print(idx, m.f, m.i)
            if m.f != -1:  # if not from previous layer     
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers   
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output    
  
        decoder_feats = [y[j] for j in self.decoder.f]
        if getattr(self.decoder, "supports_depth_prior", False):
            x = self.decoder(decoder_feats, targets=targets, depth_prior=depth_prior)
        else:
            x = self.decoder(decoder_feats, targets)    
        return x    

    def deploy(self, ):   
        self.eval() 
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
   
    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):   
            m = FrozenBatchNorm2d(m.num_features)   
        else:     
            for name, child in m.named_children():
                _child = self._freeze_norm(child)   
                if _child is not child:
                    setattr(m, name, _child)   
        return m   
    
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False    
   
  
@register(force=True)
class GeoSemDet(DEIMGraph):
    __share__ = ['num_classes', 'eval_spatial_size', 'text_cache_file']

    def __init__(
        self,
        yaml_path,
        pretrained=None,
        freeze_stem_only=False,
        freeze_at=-1,
        freeze_norm=False,
        num_classes=80,
        eval_spatial_size=(640, 640),
        img_dim=256,
        text_dim=512,
        text_adapter_layers=1,
        text_cache_file: Optional[str] = None,
        expected_categories=None,
    ):
        super().__init__(
            yaml_path=yaml_path,
            pretrained=pretrained,
            freeze_stem_only=freeze_stem_only,
            freeze_at=freeze_at,
            freeze_norm=freeze_norm,
            num_classes=num_classes,
            eval_spatial_size=eval_spatial_size,
        )
        self.text_adapter = TextAdapter(text_dim, img_dim, text_adapter_layers)
        self.text_cache_file = text_cache_file
        self.expected_categories = expected_categories
        self.register_buffer("class_text_feats", torch.empty(0), persistent=False)
        self.register_buffer("class_text_feats_all", torch.empty(0), persistent=False)
        self.register_buffer("anchor_text_feats", torch.empty(0), persistent=False)
        self.text_categories: list[str] = []
        self.anchor_names: list[str] = []
        self.class_descriptions: list[list[str]] = []
        self._forward_text_log_emitted = False
        if text_cache_file:
            self._load_text_cache(text_cache_file)

    def _load_text_cache(self, text_cache_file: str) -> None:
        text_cache = load_text_bank(
            torch.load(Path(text_cache_file), map_location="cpu"),
            text_cache_file,
            expected_categories=self.expected_categories,
        )
        if text_cache["num_classes"] != len(text_cache["categories"]):
            raise ValueError(
                f"text bank num_classes mismatch: metadata={text_cache['num_classes']}, "
                f"categories={len(text_cache['categories'])}"
            )
        self.class_text_feats = text_cache["class_text_feats"]
        self.class_text_feats_all = text_cache["class_text_feats_all"]
        self.anchor_text_feats = text_cache["anchor_text_feats"]
        self.text_categories = text_cache["categories"]
        self.anchor_names = text_cache["anchor_names"]
        self.class_descriptions = text_cache["class_descriptions"]
        logger.info(
            ORANGE + "text bank loaded for model: "
            + summarize_text_bank(text_cache, text_cache_file) + RESET
        )

    def _adapt_text_feats(self, batch_size: int, device: torch.device):
        if self.class_text_feats.numel() == 0 or self.anchor_text_feats.numel() == 0:
            raise RuntimeError("GeoSemDet requires class and anchor text features from text_cache_file.")
        class_text_feats = self.text_adapter(self.class_text_feats.to(device))
        anchor_text_feats = self.text_adapter(self.anchor_text_feats.to(device))
        class_text_feats = class_text_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        anchor_text_feats = anchor_text_feats.unsqueeze(0).repeat(batch_size, 1, 1)
        return class_text_feats, anchor_text_feats

    def _collect_gate_stats(self, device: torch.device) -> dict:
        gate_gammas = []
        for module in self.modules():
            if isinstance(module, SRGEncoderGate):
                gate_gammas.append(module.gamma.detach())
        if not gate_gammas:
            return {}
        stats = {}
        gamma_stack = torch.stack([gamma.to(device=device).float() for gamma in gate_gammas])
        stats["srg_gamma_mean"] = gamma_stack.mean()
        for index, gamma in enumerate(gate_gammas):
            stats[f"srg_gamma_{index}"] = gamma.to(device=device).float()
        return stats

    def forward(self, x, targets=None):
        y = []
        depth_prior = None
        if isinstance(x, dict):
            depth_prior = x.get("depth_geom", x.get("raw_depth_aligned", x.get("raw_npy", x.get("npy"))))
            x = x["rgb"]
            bs = x.shape[0]
            device = x.device
        else:
            bs = x.shape[0]
            device = x.device

        class_text_feats, anchor_text_feats = self._adapt_text_feats(bs, device)

        for idx, m in enumerate(list(self.backbone.children()) + list(self.encoder.children())):
            if isinstance(m, SRGEncoderGate):
                x = m(x, anchor_text_feats)
                y.append(x if m.i in self.save else None)
                continue

            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        if not self._forward_text_log_emitted:
            logger.info(
                ORANGE + "SRG text features: "
                f"class_shape={tuple(class_text_feats.shape)}, anchor_shape={tuple(anchor_text_feats.shape)}, "
                f"device={class_text_feats.device}, dtype={class_text_feats.dtype}" + RESET
            )
            self._forward_text_log_emitted = True

        decoder_feats = [y[j] for j in self.decoder.f]
        if getattr(self.decoder, "supports_depth_prior", False):
            out = self.decoder(
                decoder_feats,
                targets=targets,
                class_text_feats=class_text_feats,
                depth_prior=depth_prior,
            )
        else:
            out = self.decoder(decoder_feats, targets=targets, class_text_feats=class_text_feats)
        srg_stats = out.setdefault("srg_stats", {})
        srg_stats.update(self._collect_gate_stats(device))
        return out

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file. 

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.    

    Returns:
        (dict): YAML data and file name.
    """   
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"  
    with open(file, errors="ignore", encoding="utf-8") as f:   
        s = f.read()  # string

        # Remove special characters 
        if not s.isprintable():  
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
     
        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files) 
        if append_filename: 
            data["yaml_file"] = str(file)
        return data 
   
def parse_module(d, i, f, m, args, ch, nc=None, eval_spatial_size=None):   
    import ast   
    try: 
        if m == 'node_mode':    
            m = d[m] 
            if len(args) > 0:
                if args[0] == 'head_channel':
                    args[0] = int(d[args[0]])
        t = m
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
    except:   
        pass

    selfatt, selfatt_args_index = None, -1     
    if type(args) is list:
        for j, a in enumerate(args):  
            if isinstance(a, str):
                with contextlib.suppress(ValueError):  
                    try: 
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)   
                    except:   
                        args[j] = a
            elif type(a) is dict:
                if 'module' in a:    
                    module_ = a['module']
                    try:  
                        module_ = getattr(torch.nn, module_[3:]) if 'nn.' in module_ else globals()[module_]
                    except Exception as e:     
                        raise Exception(f'{module_} is maybe not import in task.py, please check. {e}')
                    module_param = a.get('param', {}) 
                    for k in module_param:
                        p = module_param[k]
                        try:
                            module_param[k] = locals()[p] if p in locals() else (getattr(torch.nn, p[3:]) if 'nn.' in p else ast.literal_eval(p))
                        except:
                            module_param[k] = p   
                    args[j] = partial(module_, **module_param)
                if 'selfatt' in a:     
                    selfatt = a['selfatt']   
                    selfatt_args_index = j     
    if selfatt_args_index != -1:
        args.pop(j)
    
    c2 = ch[-1]
    if m in {StemBlock, HG_Stage}:
        c1, cmid, c2 = ch[f], args[0], args[1]
        args = [c1, cmid, c2, *args[2:]]
    elif m in {ConvNormLayer_fuse, SCDown}:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {TransformerEncoderBlock}:
        c2 = ch[f]
        args = [c2, *args]
    elif m is Concat:
        c2 = sum(ch[x] for x in f)
    elif m is Conv:
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {MambaOut, DSM, SRGEncoderGate}:  # merge-node blocks and the SRG encoder gate
        c1, c2 = ch[f], args[0]
        args = [c1, c2, *args[1:]]
    elif m in {DFINETransformer, GSATransformer, SRGTransformer, GeoSemDetTransformer}:
        args["feat_channels"] = [ch[x] for x in f]
        args["num_classes"] = nc
        args["eval_spatial_size"] = eval_spatial_size
    else:
        c2 = ch[f]

    if isinstance(m, type) == False and type(m) is not str:   
        m_ = m
    else:     
        if type(args) is dict:
            m_ = m(**args)
        elif type(args) is list:
            m_ = m(*args)  # module 
        else:
            m_ = m(*args[0], **args[1])
        t = str(m)[8:-2].replace('__main__.', '')  # module type   
    m_.np = sum(x.numel() for x in m_.parameters())  # number params
    m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type     
 
    return m_, c2, t, args

def parse_model(d, ch, nc, eval_spatial_size, verbose=True): 
    if verbose:  
        logger.info(ORANGE + f"{'':>3}{'from':>10}{'params':>10}  {'module':<60}{'arguments':<30}" + RESET)    
    layer_index, ch = 0, [ch]
    backbone_layers, encoder_layers, decoder_model, save, c2 = [], [], None, [], ch[-1]  # layers, savelist, ch out   
 
    if verbose:
        logger.info(BLUE + "-"*40 + "BackBone" + "-"*40 + RESET)  
    for f, m, args in d["backbone"]:
   
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch, eval_spatial_size=eval_spatial_size)
   
        if verbose:
            logger.info(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print     
        
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        backbone_layers.append(m_)
        if layer_index == 0:   
            ch = []
        ch.append(c2)

        layer_index += 1
   
    if verbose:
        logger.info(BLUE + "-"*40 + "Enchoder" + "-"*40 + RESET)  
    for f, m, args in d["encoder"]:
    
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch, eval_spatial_size=eval_spatial_size)

        if verbose:
            logger.info(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
   
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 
        encoder_layers.append(m_) 
        ch.append(c2)

        layer_index += 1
    
    if verbose:
        logger.info(BLUE + "-"*40 + "Decoder" + "-"*40 + RESET)
    for f, m, args in d["decoder"]:    
 
        m_, c2, t, args = parse_module(d, layer_index, f, m, args, ch, nc, eval_spatial_size)
    
        if verbose:
            logger.info(ORANGE + f"{layer_index:>3}{str(f):>10}{m_.np:10.0f}  {t:<60}{str(args):<30}" + RESET)  # print
    
        save.extend(x % layer_index for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        decoder_model = m_   
        ch.append(c2)    
    
    # print(ch)
    return nn.Sequential(*backbone_layers), nn.Sequential(*encoder_layers), decoder_model, sorted(save)
