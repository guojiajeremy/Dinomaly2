"""
ONNX export script for Dinomaly2D.
Supports all encoders from vit_encoder:
  dinov1, dinov2, dinov3, dino, mae, ibot, beit, beitv2, deit, tips, digpt, moco
"""

import torch
import torch.nn as nn
import argparse

from models.uad import Dinomaly
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention2
from functools import partial


def build_model(encoder_name, la=False, lc=2, dropout=0.4, cr=True):
    """Build Dinomaly model matching dinomaly_2D.py configuration."""
    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    # Determine fuse layers from lc
    if lc == 0:
        fuse_layer_encoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
        fuse_layer_decoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif lc == 1:
        fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    elif lc == 2:
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    elif lc == 3:
        fuse_layer_encoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
    elif lc == 4:
        fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elif lc == 11:
        fuse_layer_encoder = [[7]]
        fuse_layer_decoder = [[7]]
    elif lc == 12:
        fuse_layer_encoder = [[3], [7]]
        fuse_layer_decoder = [[3], [7]]
    elif lc == 14:
        fuse_layer_encoder = [[1], [3], [5], [7]]
        fuse_layer_decoder = [[1], [3], [5], [7]]
    else:
        raise ValueError(f"lc={lc} not supported")

    # Bottleneck
    bottleneck = []
    bottleneck.append(nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)))
    bottleneck.append(nn.Sequential(
        nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
        nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout)
    ))
    bottleneck = nn.ModuleList(bottleneck)

    # Decoder: 8 blocks
    decoder = []
    attn_cls = partial(LinearAttention2, eps=1e-8) if la else Attention
    for i in range(8):
        blk = VitBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=attn_cls
        )
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = Dinomaly(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=target_layers,
        remove_class_token=False,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        context_aware_recenter=cr,
    )

    return model


def export_onnx(model, image_size, output_path, device='cpu'):
    """Export model to ONNX format."""
    model = model.to(device).eval()

    # Derive patch_size from actual encoder
    patch_size = getattr(model.encoder, 'patch_embed', None)
    if patch_size is not None:
        ps = patch_size.patch_size
        patch_size = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)
    else:
        patch_size = 16

    side = int(image_size // patch_size)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # ONNX wrapper: stacks feature lists into single tensors
    class Dinomaly2D_ONNX(nn.Module):
        def __init__(self, model, patch_size, side):
            super().__init__()
            self.model = model
            self.patch_size = patch_size
            self.side = side
            self.encoder = model.encoder
            self.bottleneck = model.bottleneck
            self.decoder = model.decoder
            self.target_layers = model.target_layers
            self.fuse_layer_encoder = model.fuse_layer_encoder
            self.fuse_layer_decoder = model.fuse_layer_decoder
            self.remove_class_token = model.remove_class_token
            self.context_aware_recenter = model.context_aware_recenter
            self.num_register_tokens = getattr(self.encoder, 'num_register_tokens', 0)

        def forward(self, x):
            import torch.nn.functional as F

            # Encoder forward (no torch.no_grad for ONNX graph)
            x_enc = self.encoder.prepare_tokens(x)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                x_enc = blk(x_enc)
                if i in self.target_layers:
                    en_list.append(x_enc)

            # Bottleneck: fuse encoder features -> bottleneck -> decoder
            if self.remove_class_token:
                fused = [e[:, 1 + self.num_register_tokens:, :] for e in en_list]
            else:
                fused = en_list
            x = self.model.fuse_feature([fused[idx] for idx in [0, 1, 2, 3, 4, 5, 6, 7]])

            for i, blk in enumerate(self.bottleneck):
                x = blk(x)

            # Decoder
            de_list = []
            for i, blk in enumerate(self.decoder):
                x = blk(x)
                de_list.append(x)
            de_list = de_list[::-1]

            # Fuse features per layer group
            en = [self.model.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
            de = [self.model.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

            # Remove class token from decoder features only
            # Encoder features are handled in the context-aware recenter block below
            if not self.remove_class_token:
                de = [d[:, 1 + self.num_register_tokens:, :] for d in de]

            # Context-aware recenter for encoder features
            if self.context_aware_recenter:
                en_centered = [e[:, 1 + self.num_register_tokens:, :] - e[:, :1, :] for e in en]
                en_centered = [F.layer_norm(e, normalized_shape=(e.shape[-1],), eps=1e-8) for e in en_centered]
            else:
                en_centered = [e[:, 1 + self.num_register_tokens:, :] for e in en]

            # Use non-recentered features for reshape (they have correct seq_len)
            en_reshape = [e[:, 1 + self.num_register_tokens:, :] for e in en]

            # Reshape to spatial [B, C, H, W] using concrete side dimension
            bs = x.shape[0]
            h = w = self.side
            en = [e.permute(0, 2, 1).reshape(bs, -1, h, w).contiguous() for e in en_reshape]
            de = [d.permute(0, 2, 1).reshape(bs, -1, h, w).contiguous() for d in de]

                    # Return all groups as separate outputs (dynamic count per lc value)
            outputs = []
            for i in range(len(en)):
                outputs.append(en[i])
            for i in range(len(de)):
                outputs.append(de[i])
            return tuple(outputs)

    onnx_model = Dinomaly2D_ONNX(model, patch_size, side)
    onnx_model = onnx_model.to(device)

    # Fixed shape - no dynamic_shapes
    num_en_groups = len(model.fuse_layer_encoder)
    num_de_groups = len(model.fuse_layer_decoder)
    output_names = []
    for i in range(num_en_groups):
        output_names.append(f'encoder_features_{i}')
    for i in range(num_de_groups):
        output_names.append(f'decoder_features_{i}')

    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        external_data=False,
        verbose=False,
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"  Input: (batch, 3, {image_size}, {image_size})")
    print(f"  Output: {', '.join(output_names)}")


def load_checkpoint(model, checkpoint_path):
    """Load trained checkpoint into model."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    # Handle DataParallel/DistributedDataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.' prefix if exists
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Export Dinomaly2D model to ONNX')
    parser.add_argument('--checkpoint', type=str, default=r"./saved_results/dinomaly2_visa_uni_dinov2sr_i448392_en29_3bn2564e_dp4_la_lc2_llp09f01_car_it40k_sadam2e42e3_wd1e4_w1h_b16_s1/model.pth",
                        help='Path to trained model.pth checkpoint')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14',
                        help='Encoder name (dinov2reg_vit_small_14, dinov2reg_vit_base_14, etc.)')
    parser.add_argument('--image_size', type=int, default=392,
                        help='Input image size')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (auto-generated if not specified)')
    parser.add_argument('--la', type=int, default=1,
                        help='Linear Attention (1=yes, 0=no)')
    parser.add_argument('--lc', type=int, default=2,
                        help='Loose Constraint groups (0=layer-to-layer, 1=1group, 2=2group, etc.)')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--cr', type=int, default=1,
                        help='Context-aware recenter (1=yes, 0=no)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for export (cpu/cuda)')

    args = parser.parse_args()

    # Auto-generate output path: replace .pth with .onnx in same directory
    if args.output is None:
        args.output = args.checkpoint.replace('.pth', '.onnx')

    print(f"Building model: encoder={args.encoder}, image_size={args.image_size}")
    model = build_model(
        encoder_name=args.encoder,
        la=args.la == 1,
        lc=args.lc,
        dropout=args.dropout,
        cr=args.cr == 1,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint)

    print(f"Exporting to ONNX: {args.output}")
    export_onnx(model, args.image_size, args.output, device=args.device)


if __name__ == '__main__':
    main()
