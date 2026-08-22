from ..core import register
from .gsa_decoder import GSATransformerDecoder, GSATransformerDecoderLayer
from .srg_decoder import SRGTransformer, SRGDecoderWrapper

__all__ = ["GeoSemDetTransformer"]


@register()
class GeoSemDetTransformer(SRGTransformer):
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
        use_text_logits: bool = True,
        fuse_text_logits: bool = True,
        text_scale: float = 10.0,
        alpha_init: float = 0.05,
        alpha_max: float = 0.1,
        fusion_mode: str = "global",
        text_proj_mode: str = "free",
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
            use_text_logits=use_text_logits,
            fuse_text_logits=fuse_text_logits,
            text_scale=text_scale,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
            fusion_mode=fusion_mode,
            text_proj_mode=text_proj_mode,
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
        depth_decoder = GSATransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            self.eval_idx,
            layer_scale,
            act=activation,
        )
        self.decoder = SRGDecoderWrapper(depth_decoder)

    def _build_depth_context(self, depth_prior, spatial_shapes):
        first_layer = self.decoder.layers[0]
        return first_layer.cross_attn.build_depth_context(depth_prior, spatial_shapes)
