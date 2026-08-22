"""
Depth-guided D-FINE decoder.

The implementation keeps the original D-FINE decoder parameter hierarchy and
only injects an aligned raw-depth geometry bias before deformable attention
softmax.
"""

import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dfine_decoder import (
    DFINETransformer,
    LQE,
    MSDeformableAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from .dfine_utils import distance2bbox, weighting_function
from .utils import inverse_sigmoid
from ..core import register

__all__ = ["GSATransformer", "GSADeformableAttention"]


class GSADeformableAttention(MSDeformableAttention):
    """MSDeformableAttention with decoder-side depth consistency bias."""

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
        depth_lambda_init=-8.0,
        depth_sigma_init=0.2,
        depth_unit_scale=1.0,
        eps=1e-6,
    ):
        super().__init__(embed_dim, num_heads, num_levels, num_points, method, offset_scale)
        self.depth_lambda_raw = nn.Parameter(torch.tensor(float(depth_lambda_init)))
        sigma_raw = math.log(math.expm1(float(depth_sigma_init)))
        self.depth_sigma_raw = nn.Parameter(torch.tensor(sigma_raw))
        self.depth_unit_scale = float(depth_unit_scale)
        self.eps = float(eps)
        self.last_depth_bias_shape = None

    @property
    def depth_lambda(self):
        return F.softplus(self.depth_lambda_raw)

    @property
    def depth_sigma(self):
        return F.softplus(self.depth_sigma_raw) + self.eps

    def build_depth_context(self, depth_prior, spatial_shapes):
        """Build aligned raw-depth and valid-mask pyramids for decoder attention."""
        if depth_prior is None:
            return None
        if not torch.is_tensor(depth_prior):
            raise TypeError(f"depth_prior must be a torch.Tensor, got {type(depth_prior)}")

        depth = depth_prior.float()
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4:
            raise ValueError(f"depth_prior must have shape [B,1,H,W] or [B,H,W], got {tuple(depth.shape)}")
        if depth.shape[1] != 1:
            raise ValueError(f"depth_prior must have one channel, got {depth.shape[1]}")

        depth = depth * self.depth_unit_scale
        valid = torch.isfinite(depth) & (depth > 0)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        depth_levels = []
        valid_levels = []
        for h, w in spatial_shapes:
            size = (int(h), int(w))
            depth_levels.append(F.interpolate(depth, size=size, mode="bilinear", align_corners=False))
            valid_levels.append(F.interpolate(valid.float(), size=size, mode="nearest") > 0.5)
        return {"depth": depth_levels, "valid": valid_levels}

    @staticmethod
    def _reference_xy(reference_points, num_levels):
        if reference_points.dim() == 3:
            ref_xy = reference_points[..., :2]
            ref_xy = ref_xy[:, :, None, :].expand(-1, -1, num_levels, -1)
        elif reference_points.dim() == 4:
            ref_xy = reference_points[..., :2]
            if ref_xy.shape[2] == 1 and num_levels > 1:
                ref_xy = ref_xy.expand(-1, -1, num_levels, -1)
            elif ref_xy.shape[2] != num_levels:
                raise ValueError(
                    f"reference_points levels must be 1 or {num_levels}, got {ref_xy.shape[2]}"
                )
        else:
            raise ValueError(
                "reference_points must have shape [B,Q,2/4] or [B,Q,L,2/4], "
                f"got {tuple(reference_points.shape)}"
            )
        return ref_xy

    @staticmethod
    def _sample_map(map_level, xy):
        grid = xy * 2.0 - 1.0
        return F.grid_sample(
            map_level,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    def _sample_depth_bias(self, depth_context, reference_points, sampling_locations, query_dtype):
        if depth_context is None:
            return None

        bs, len_q, num_heads, total_points, _ = sampling_locations.shape
        ref_xy = self._reference_xy(reference_points, self.num_levels)
        sampling_by_level = sampling_locations.split(self.num_points_list, dim=3)

        bias_levels = []
        valid_levels = []
        for level, points in enumerate(self.num_points_list):
            sample_xy = sampling_by_level[level]
            in_bounds = (
                (sample_xy[..., 0] >= 0.0)
                & (sample_xy[..., 0] <= 1.0)
                & (sample_xy[..., 1] >= 0.0)
                & (sample_xy[..., 1] <= 1.0)
            )

            depth_level = depth_context["depth"][level].to(device=sample_xy.device, dtype=torch.float32)
            valid_level = depth_context["valid"][level].to(device=sample_xy.device)

            sample_grid = sample_xy.reshape(bs, len_q * num_heads, points, 2)
            sample_depth = self._sample_map(depth_level, sample_grid)
            sample_depth = sample_depth.reshape(bs, 1, len_q, num_heads, points).permute(0, 2, 3, 4, 1)
            sample_depth = sample_depth.squeeze(-1)

            sample_valid = self._sample_map(valid_level.float(), sample_grid)
            sample_valid = sample_valid.reshape(bs, 1, len_q, num_heads, points).permute(0, 2, 3, 4, 1)
            sample_valid = sample_valid.squeeze(-1) > 0.5

            ref_grid = ref_xy[:, :, level, :].reshape(bs, len_q, 1, 2)
            ref_depth = self._sample_map(depth_level, ref_grid).reshape(bs, 1, len_q, 1).permute(0, 2, 1, 3)
            ref_depth = ref_depth.reshape(bs, len_q, 1, 1)

            ref_valid = self._sample_map(valid_level.float(), ref_grid).reshape(bs, 1, len_q, 1).permute(0, 2, 1, 3)
            ref_valid = ref_valid.reshape(bs, len_q, 1, 1) > 0.5

            sample_depth = torch.nan_to_num(sample_depth, nan=0.0, posinf=0.0, neginf=0.0)
            ref_depth = torch.nan_to_num(ref_depth, nan=0.0, posinf=0.0, neginf=0.0)

            level_bias = -torch.abs(sample_depth - ref_depth.detach()) / self.depth_sigma
            valid_bias = sample_valid & ref_valid & in_bounds
            level_bias = torch.where(valid_bias, level_bias, torch.zeros_like(level_bias))
            level_bias = level_bias.clamp(min=-20.0, max=0.0)

            bias_levels.append(level_bias)
            valid_levels.append(valid_bias)

        depth_bias = torch.cat(bias_levels, dim=-1).to(dtype=query_dtype)
        self.last_depth_bias_shape = tuple(depth_bias.shape)
        if depth_bias.shape != (bs, len_q, num_heads, total_points):
            raise ValueError(
                f"depth_bias shape {tuple(depth_bias.shape)} does not match attention logits "
                f"{(bs, len_q, num_heads, total_points)}"
            )
        return depth_bias

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int],
        depth_context=None,
    ):
        bs, len_q = query.shape[:2]

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, len_q, self.num_heads, sum(self.num_points_list), 2)

        attn_logits = self.attention_weights(query).reshape(bs, len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            if reference_points.dim() == 3:
                reference_points_for_offsets = reference_points[:, :, None, :].expand(-1, -1, self.num_levels, -1)
            elif reference_points.dim() == 4 and reference_points.shape[2] == 1:
                reference_points_for_offsets = reference_points.expand(-1, -1, self.num_levels, -1)
            elif reference_points.dim() == 4 and reference_points.shape[2] == self.num_levels:
                reference_points_for_offsets = reference_points
            else:
                raise ValueError(
                    f"reference_points must be [B,Q,2] or [B,Q,L,2], got {tuple(reference_points.shape)}"
                )

            offset_levels = sampling_offsets.split(self.num_points_list, dim=3)
            loc_levels = []
            for level, offset_level in enumerate(offset_levels):
                h, w = value_spatial_shapes[level]
                normalizer = torch.tensor([w, h], device=query.device, dtype=query.dtype).reshape(1, 1, 1, 1, 2)
                ref_xy = reference_points_for_offsets[:, :, level, :].reshape(bs, len_q, 1, 1, 2)
                loc_levels.append(ref_xy + offset_level / normalizer)
            sampling_locations = torch.cat(loc_levels, dim=3)
        elif reference_points.shape[-1] == 4:
            if reference_points.dim() == 3:
                reference_points_for_offsets = reference_points[:, :, None, :]
            elif reference_points.dim() == 4 and reference_points.shape[2] in (1, self.num_levels):
                reference_points_for_offsets = reference_points
            else:
                raise ValueError(
                    f"reference_points must be [B,Q,4] or [B,Q,L,4], got {tuple(reference_points.shape)}"
                )

            offset_levels = sampling_offsets.split(self.num_points_list, dim=3)
            scale_levels = self.num_points_scale.to(device=query.device, dtype=query.dtype).split(self.num_points_list)
            loc_levels = []
            for level, (offset_level, scale_level) in enumerate(zip(offset_levels, scale_levels)):
                ref_level_idx = 0 if reference_points_for_offsets.shape[2] == 1 else level
                ref_level = reference_points_for_offsets[:, :, ref_level_idx, :]
                ref_xy = ref_level[:, :, None, None, :2]
                ref_wh = ref_level[:, :, None, None, 2:]
                scale_level = scale_level.reshape(1, 1, 1, -1, 1)
                offset = offset_level * scale_level * ref_wh * self.offset_scale
                loc_levels.append(ref_xy + offset)
            sampling_locations = torch.cat(loc_levels, dim=3)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        depth_bias = self._sample_depth_bias(depth_context, reference_points, sampling_locations, query.dtype)
        if depth_bias is not None:
            attn_logits = attn_logits + self.depth_lambda.to(dtype=query.dtype) * depth_bias

        attention_weights = F.softmax(attn_logits, dim=-1)
        output = self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )
        return output


class GSATransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
        depth_lambda_init=-8.0,
        depth_sigma_init=0.2,
        depth_unit_scale=1.0,
    ):
        super().__init__(
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            activation,
            n_levels,
            n_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        if layer_scale is not None:
            d_model = round(layer_scale * d_model)
            dim_feedforward = round(layer_scale * dim_feedforward)
        self.cross_attn = GSADeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
            depth_lambda_init=depth_lambda_init,
            depth_sigma_init=depth_sigma_init,
            depth_unit_scale=depth_unit_scale,
        )

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
        depth_context=None,
    ):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
            depth_context=depth_context,
        )
        target = self.gateway(target, self.dropout2(target2))

        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))
        return target


class GSATransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
    ):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
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
        dec_out_pred_corners = []
        dec_out_refs = []
        project = self.project if hasattr(self, "project") else weighting_function(self.reg_max, up, reg_scale)

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

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
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


@register()
class GSATransformer(DFINETransformer):
    __share__ = ["num_classes", "eval_spatial_size"]
    supports_depth_prior = True

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
        depth_lambda_init=-8.0,
        depth_sigma_init=0.2,
        depth_unit_scale=1.0,
    ):
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
        decoder_layer = GSATransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            depth_lambda_init=depth_lambda_init,
            depth_sigma_init=depth_sigma_init,
            depth_unit_scale=depth_unit_scale,
        )
        decoder_layer_wide = GSATransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            depth_lambda_init=depth_lambda_init,
            depth_sigma_init=depth_sigma_init,
            depth_unit_scale=depth_unit_scale,
        )
        self.decoder = GSATransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
        )

    def _build_depth_context(self, depth_prior, spatial_shapes):
        if depth_prior is None:
            return None
        first_layer = self.decoder.layers[0]
        return first_layer.cross_attn.build_depth_context(depth_prior, spatial_shapes)

    def forward(self, feats, targets=None, depth_prior=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        depth_context = self._build_depth_context(depth_prior, spatial_shapes)

        if self.training and self.num_denoising > 0:
            from .denoising import get_contrastive_denoising_training_group

            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=1.0,
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, enc_outputs_logits = (
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        )

        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            depth_context=depth_context,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

            out["enc_outputs_logits"] = enc_outputs_logits

        return out
