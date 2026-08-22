import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .denoising import get_contrastive_denoising_training_group
from .dfine_decoder import DFINETransformer
from .dfine_utils import distance2bbox, weighting_function
from .utils import inverse_sigmoid

__all__ = ["SRGTransformer"]


class SRGTextScoreHead(nn.Module):
    def __init__(self, hidden_dim: int, text_scale: float = 10.0, proj_mode: str = "free"):
        super().__init__()
        self.proj_mode = str(proj_mode).lower()
        if self.proj_mode == "identity":
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.constant_(self.proj.bias, 0)
        self.register_buffer("text_scale", torch.tensor(float(text_scale)), persistent=False)

    def forward(self, query_feats: torch.Tensor, class_text_feats: torch.Tensor) -> torch.Tensor:
        if class_text_feats.dim() == 2:
            class_text_feats = class_text_feats.unsqueeze(0)
        if class_text_feats.shape[0] == 1 and query_feats.shape[0] > 1:
            class_text_feats = class_text_feats.repeat(query_feats.shape[0], 1, 1)
        if class_text_feats.shape[0] != query_feats.shape[0]:
            raise ValueError(
                f"class_text_feats batch mismatch: got {class_text_feats.shape[0]}, expected {query_feats.shape[0]}"
            )
        image_norm = F.normalize(self.proj(query_feats), p=2, dim=-1)
        text_norm = F.normalize(class_text_feats.to(query_feats.device), p=2, dim=-1)
        return (image_norm @ text_norm.transpose(2, 1)) * self.text_scale.to(query_feats.dtype)


class SRGAdaptiveLogitFusion(nn.Module):
    def __init__(self, alpha_max: float, hidden_dim: int = 16):
        super().__init__()
        if alpha_max <= 0:
            raise ValueError(f"alpha_max must be positive, got {alpha_max}")
        self.alpha_max = float(alpha_max)
        self.gate = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _reliability_features(closed_logits: torch.Tensor, text_logits: torch.Tensor) -> torch.Tensor:
        closed_prob = closed_logits.detach().sigmoid()
        text_prob = text_logits.detach().sigmoid()

        topk = torch.topk(closed_prob, k=min(2, closed_prob.shape[-1]), dim=-1).values
        visual_conf = topk[..., :1]
        if topk.shape[-1] == 1:
            visual_margin = visual_conf
        else:
            visual_margin = topk[..., :1] - topk[..., 1:2]

        text_conf = text_prob.max(dim=-1, keepdim=True).values
        agreement = F.cosine_similarity(closed_prob, text_prob, dim=-1, eps=1e-6).unsqueeze(-1)
        return torch.cat([visual_conf, visual_margin, text_conf, agreement], dim=-1)

    def forward(self, closed_logits: torch.Tensor, text_logits: torch.Tensor, base_alpha: torch.Tensor) -> torch.Tensor:
        features = self._reliability_features(closed_logits, text_logits).to(closed_logits.dtype)
        alpha_factor = 2.0 * torch.sigmoid(self.gate(features))
        text_alpha = base_alpha.to(closed_logits.dtype) * alpha_factor
        return text_alpha.clamp(max=self.alpha_max)


class SRGDecoderWrapper(nn.Module):
    def __init__(self, base_decoder: nn.Module) -> None:
        super().__init__()
        self.hidden_dim = base_decoder.hidden_dim
        self.num_layers = base_decoder.num_layers
        self.layer_scale = base_decoder.layer_scale
        self.num_head = base_decoder.num_head
        self.eval_idx = base_decoder.eval_idx
        self.up = base_decoder.up
        self.reg_scale = base_decoder.reg_scale
        self.reg_max = base_decoder.reg_max
        self.layers = base_decoder.layers
        self.lqe_layers = base_decoder.lqe_layers

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = torch.nn.functional.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]])

    @staticmethod
    def _run_closed_head(head: nn.Module, output: torch.Tensor) -> torch.Tensor:
        return head(output)

    @staticmethod
    def _run_text_head(head: nn.Module, output: torch.Tensor, class_text_feats: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if head is None or isinstance(head, nn.Identity) or class_text_feats is None:
            return None
        return head(output, class_text_feats)

    def _combine_logits(
        self,
        closed_logits: torch.Tensor,
        text_logits: Optional[torch.Tensor],
        alpha: torch.Tensor,
        fuse_text_logits: bool,
        fusion_mode: str,
        text_logit_fusion: Optional[nn.Module],
    ) -> torch.Tensor:
        if text_logits is None or not fuse_text_logits:
            return closed_logits
        if fusion_mode == "global":
            return closed_logits + alpha.to(closed_logits.dtype) * text_logits
        if fusion_mode == "adaptive":
            if text_logit_fusion is None:
                raise RuntimeError("adaptive text fusion requires text_logit_fusion.")
            text_alpha = text_logit_fusion(closed_logits, text_logits, alpha)
            return closed_logits + text_alpha.to(closed_logits.dtype) * text_logits
        raise ValueError(f"Unsupported SRG fusion_mode: {fusion_mode}")

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        class_text_feats,
        bbox_head,
        closed_score_head,
        text_score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        alpha,
        fuse_text_logits: bool,
        fusion_mode: str,
        text_logit_fusion: Optional[nn.Module],
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        depth_context=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_closed_logits = []
        dec_out_text_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        project = weighting_function(self.reg_max, up, reg_scale) if not hasattr(self, "project") else self.project
        ref_points_detach = torch.sigmoid(ref_points_unact)

        pre_logits = pre_closed_logits = pre_text_logits = None
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = torch.nn.functional.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = torch.nn.functional.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            if depth_context is None:
                output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)
            else:
                output = layer(
                    output,
                    ref_points_input,
                    value,
                    spatial_shapes,
                    attn_mask,
                    query_pos_embed,
                    depth_context=depth_context,
                )

            if i == 0:
                pre_bboxes = torch.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                if self.training or not isinstance(closed_score_head[0], nn.Identity):
                    pre_closed_logits = self._run_closed_head(closed_score_head[0], output)
                    pre_text_logits = self._run_text_head(text_score_head[0], output, class_text_feats)
                    pre_logits = self._combine_logits(
                        pre_closed_logits,
                        pre_text_logits,
                        alpha,
                        fuse_text_logits,
                        fusion_mode,
                        text_logit_fusion,
                    )
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                closed_scores = self._run_closed_head(closed_score_head[i], output)
                closed_scores = self.lqe_layers[i](closed_scores, pred_corners)
                text_scores = self._run_text_head(text_score_head[i], output, class_text_feats)
                scores = self._combine_logits(
                    closed_scores,
                    text_scores,
                    alpha,
                    fuse_text_logits,
                    fusion_mode,
                    text_logit_fusion,
                )
                dec_out_logits.append(scores)
                dec_out_closed_logits.append(closed_scores)
                dec_out_text_logits.append(text_scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        stacked_text_logits = None
        if dec_out_text_logits and dec_out_text_logits[0] is not None:
            stacked_text_logits = torch.stack(dec_out_text_logits)

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_closed_logits),
            stacked_text_logits,
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_logits,
            pre_closed_logits,
            pre_text_logits,
        )


@register()
class SRGTransformer(DFINETransformer):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_text_logits: bool = True,
        fuse_text_logits: bool = True,
        text_scale: float = 10.0,
        alpha_init: float = 0.05,
        alpha_max: float = 0.1,
   fusion_mode: str = "global",
        text_proj_mode: str = "free",
    ):
        assert query_select_method in ("default", "one2many"), "agnostic query selection is not supported in SRGTransformer"
        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=num_levels,
            num_points=num_points,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learn_query_content=learn_query_content,
            eval_spatial_size=eval_spatial_size,
            eval_idx=eval_idx,
            eps=eps,
            aux_loss=aux_loss,
            cross_attn_method=cross_attn_method,
            query_select_method=query_select_method,
            reg_max=reg_max,
            reg_scale=reg_scale,
            layer_scale=layer_scale,
            mlp_act=mlp_act,
        )
        self.use_text_logits = bool(use_text_logits)
        self.fuse_text_logits = bool(fuse_text_logits)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"global", "adaptive"}:
            raise ValueError(f"fusion_mode must be 'global' or 'adaptive', got {fusion_mode}")
        self.alpha_max = float(alpha_max)
        if self.alpha_max <= 0:
            raise ValueError(f"alpha_max must be positive, got {self.alpha_max}")
        scaled_dim = round(layer_scale * hidden_dim)
        if self.use_text_logits:
            self.enc_text_score_head = SRGTextScoreHead(hidden_dim, text_scale=text_scale, proj_mode=text_proj_mode)
            self.dec_text_score_head = nn.ModuleList(
    [SRGTextScoreHead(hidden_dim, text_scale=text_scale, proj_mode=text_proj_mode) for _ in range(self.eval_idx + 1)]
        + [SRGTextScoreHead(scaled_dim, text_scale=text_scale, proj_mode=text_proj_mode) for _ in range(num_layers - self.eval_idx - 1)]
            )
        else:
            self.enc_text_score_head = nn.Identity()
            self.dec_text_score_head = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        alpha_ratio = min(max(float(alpha_init) / self.alpha_max, 1e-6), 1 - 1e-6)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(alpha_ratio / (1.0 - alpha_ratio))))
        self.text_logit_fusion = SRGAdaptiveLogitFusion(self.alpha_max) if self.fusion_mode == "adaptive" else None
        self.decoder = SRGDecoderWrapper(self.decoder)

    def load_state_dict(self, state_dict, strict: bool = True):
        if strict and self.text_logit_fusion is not None:
            current_state = self.state_dict()
            state_dict = dict(state_dict)
            for key in current_state:
                if key.startswith("text_logit_fusion.") and key not in state_dict:
                    state_dict[key] = current_state[key]
        return super().load_state_dict(state_dict, strict=strict)

    @property
    def srg_alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.alpha_raw)

    @staticmethod
    def _prepare_text_feats(text_feats: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        if text_feats.dim() == 2:
            text_feats = text_feats.unsqueeze(0)
        text_feats = text_feats.to(device)
        if text_feats.shape[0] == 1 and batch_size > 1:
            text_feats = text_feats.repeat(batch_size, 1, 1)
        if text_feats.shape[0] != batch_size:
            raise ValueError(f"text_feats batch mismatch: got {text_feats.shape[0]}, expected {batch_size}")
        return text_feats

    def _text_alpha(self, closed_logits: torch.Tensor, text_logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if text_logits is None or not self.fuse_text_logits:
            return None
        if self.fusion_mode == "global":
            return self.srg_alpha.to(closed_logits.dtype)
        if self.fusion_mode == "adaptive":
            if self.text_logit_fusion is None:
                raise RuntimeError("adaptive text fusion requires text_logit_fusion.")
            return self.text_logit_fusion(closed_logits, text_logits, self.srg_alpha)
        raise ValueError(f"Unsupported SRG fusion_mode: {self.fusion_mode}")

    def _combine_logits(self, closed_logits: torch.Tensor, text_logits: Optional[torch.Tensor]) -> torch.Tensor:
        text_alpha = self._text_alpha(closed_logits, text_logits)
        if text_alpha is None:
            return closed_logits
        return closed_logits + text_alpha.to(closed_logits.dtype) * text_logits

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        class_text_feats: Optional[torch.Tensor],
        denoising_logits=None,
        denoising_bbox_unact=None,
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        output_memory = self.enc_output(memory)
        enc_outputs_logits_closed = self.enc_score_head(output_memory)
        enc_outputs_text_logits = (
            self.enc_text_score_head(output_memory, class_text_feats)
            if self.use_text_logits and class_text_feats is not None
            else None
        )
        enc_outputs_logits = self._combine_logits(enc_outputs_logits_closed, enc_outputs_text_logits)

        enc_topk_memory, enc_topk_closed_logits, enc_topk_anchors = self._select_topk(
            output_memory, enc_outputs_logits_closed, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list, enc_topk_closed_logits_list, enc_topk_text_logits_list = [], [], [], []
        if self.training:
            enc_topk_bboxes_list.append(torch.sigmoid(enc_topk_bbox_unact))
            enc_topk_closed_logits_list.append(enc_topk_closed_logits)
            if enc_outputs_text_logits is not None:
                enc_topk_text_logits = enc_outputs_text_logits.gather(
                    dim=1,
                    index=torch.topk(enc_outputs_logits_closed.max(-1).values, self.num_queries, dim=-1)[1]
                    .unsqueeze(-1)
                    .repeat(1, 1, enc_outputs_text_logits.shape[-1]),
                )
            else:
                enc_topk_text_logits = None
            enc_topk_text_logits_list.append(enc_topk_text_logits)
            enc_topk_logits_list.append(self._combine_logits(enc_topk_closed_logits, enc_topk_text_logits))

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return (
            content,
            enc_topk_bbox_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            enc_topk_closed_logits_list,
            enc_topk_text_logits_list,
            enc_outputs_logits,
            enc_outputs_logits_closed,
            enc_outputs_text_logits,
        )

    def _build_srg_stats(self, closed_logits: torch.Tensor, text_logits: Optional[torch.Tensor]) -> dict:
        stats = {"srg_alpha": self.srg_alpha.detach()}
        stats["mean_closed_logit_norm"] = closed_logits.detach().float().norm(dim=-1).mean()
        if text_logits is not None:
            stats["mean_text_logit_norm"] = text_logits.detach().float().norm(dim=-1).mean()
            text_alpha = self._text_alpha(closed_logits, text_logits)
            if text_alpha is None:
                stats["mean_effective_alpha"] = torch.zeros((), device=closed_logits.device)
                stats["mean_text_delta_norm"] = torch.zeros((), device=closed_logits.device)
            else:
                stats["mean_effective_alpha"] = text_alpha.detach().float().mean()
                stats["mean_text_delta_norm"] = (text_alpha * text_logits).detach().float().norm(dim=-1).mean()
        else:
            stats["mean_text_logit_norm"] = torch.zeros((), device=closed_logits.device)
            stats["mean_effective_alpha"] = torch.zeros((), device=closed_logits.device)
            stats["mean_text_delta_norm"] = torch.zeros((), device=closed_logits.device)
        return stats

    def forward(self, feats, targets=None, class_text_feats=None, depth_prior=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        depth_context = None
        if depth_prior is not None:
            if not hasattr(self, "_build_depth_context"):
                raise RuntimeError("depth_prior was provided, but this SRG decoder does not support depth context.")
            depth_context = self._build_depth_context(depth_prior, spatial_shapes)

        if self.use_text_logits:
            if class_text_feats is None:
                raise RuntimeError("SRGTransformer requires class_text_feats when use_text_logits=True.")
            class_text_feats = self._prepare_text_feats(class_text_feats, memory.shape[0], memory.device)

        if self.training and self.num_denoising > 0:
            if targets is None:
                raise RuntimeError("Training with denoising requires targets.")
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        (
            init_ref_contents,
            init_ref_points_unact,
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            enc_topk_closed_logits_list,
            enc_topk_text_logits_list,
            enc_outputs_logits,
            enc_outputs_logits_closed,
            enc_outputs_text_logits,
        ) = self._get_decoder_input(memory, spatial_shapes, class_text_feats, denoising_logits, denoising_bbox_unact)

        (
            out_bboxes,
            out_logits,
            out_logits_closed,
            out_text_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            pre_logits_closed,
            pre_text_logits,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            class_text_feats,
            self.dec_bbox_head,
            self.dec_score_head,
            self.dec_text_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            self.srg_alpha,
            self.fuse_text_logits,
            self.fusion_mode,
            self.text_logit_fusion,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            depth_context=depth_context,
        )

        dn_pre_text_logits = dn_out_text_logits = None
        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_logits_closed, pre_logits_closed = torch.split(pre_logits_closed, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_logits_closed, out_logits_closed = torch.split(out_logits_closed, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            if out_text_logits is not None:
                dn_pre_text_logits, pre_text_logits = torch.split(pre_text_logits, dn_meta["dn_num_split"], dim=1)
                dn_out_text_logits, out_text_logits = torch.split(out_text_logits, dn_meta["dn_num_split"], dim=2)

        final_text_logits = out_text_logits[-1] if out_text_logits is not None else None
        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_logits_closed": out_logits_closed[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
            if final_text_logits is not None:
                out["pred_text_logits"] = final_text_logits
        else:
            out = {
                "pred_logits": out_logits[-1],
                "pred_logits_closed": out_logits_closed[-1],
                "pred_boxes": out_bboxes[-1],
            }
            if final_text_logits is not None:
                out["pred_text_logits"] = final_text_logits

        out["srg_stats"] = self._build_srg_stats(out["pred_logits_closed"], out.get("pred_text_logits"))

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss_srg(
                out_logits[:-1],
                out_logits_closed[:-1],
                out_text_logits[:-1] if out_text_logits is not None else None,
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss_srg_enc(
                enc_topk_logits_list,
                enc_topk_closed_logits_list,
                enc_topk_text_logits_list,
                enc_topk_bboxes_list,
            )
            out["pre_outputs"] = {
                "pred_logits": pre_logits,
                "pred_logits_closed": pre_logits_closed,
                "pred_boxes": pre_bboxes,
            }
            if pre_text_logits is not None:
                out["pre_outputs"]["pred_text_logits"] = pre_text_logits
            out["enc_meta"] = {"class_agnostic": False}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss_srg(
                    dn_out_logits,
                    dn_out_logits_closed,
                    dn_out_text_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {
                    "pred_logits": dn_pre_logits,
                    "pred_logits_closed": dn_pre_logits_closed,
                    "pred_boxes": dn_pre_bboxes,
                }
                if dn_pre_text_logits is not None:
                    out["dn_pre_outputs"]["pred_text_logits"] = dn_pre_text_logits
                out["dn_meta"] = dn_meta

            out["enc_outputs_logits"] = enc_outputs_logits
            out["enc_outputs_logits_closed"] = enc_outputs_logits_closed
            if enc_outputs_text_logits is not None:
                out["enc_outputs_text_logits"] = enc_outputs_text_logits

        return out

    @torch.jit.unused
    def _set_aux_loss_srg(
        self,
        outputs_class,
        outputs_closed_class,
        outputs_text_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        aux = []
        for i, (a, ac, b, c, d) in enumerate(
            zip(outputs_class, outputs_closed_class, outputs_coord, outputs_corners, outputs_ref)
        ):
            item = {
                "pred_logits": a,
                "pred_logits_closed": ac,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            if outputs_text_class is not None:
                item["pred_text_logits"] = outputs_text_class[i]
            aux.append(item)
        return aux

    @torch.jit.unused
    def _set_aux_loss_srg_enc(self, outputs_class, outputs_closed_class, outputs_text_class, outputs_coord):
        aux = []
        for i, (a, ac, b) in enumerate(zip(outputs_class, outputs_closed_class, outputs_coord)):
            item = {"pred_logits": a, "pred_logits_closed": ac, "pred_boxes": b}
            if outputs_text_class and outputs_text_class[i] is not None:
                item["pred_text_logits"] = outputs_text_class[i]
            aux.append(item)
        return aux
