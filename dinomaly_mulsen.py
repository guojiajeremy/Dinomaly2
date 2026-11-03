# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset_mulsen import get_data_transforms, TrainDataset, TestDataset
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import Dinomaly
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
from utils import evaluation_batch_rgbinfra, global_cosine, global_cosine_hm_percent, WarmupCosineScheduler
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list, args):
    setup_seed(1)

    total_iters = args.total_iters
    batch_size = 8
    image_size = args.image_size
    crop_size = args.crop_size

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_data = TrainDataset(class_name=item, dataset_path=args.data_path,
                                  transform=data_transform, gt_transform=gt_transform, depth=False)
        test_data = TestDataset(class_name=item, dataset_path=args.data_path,
                                transform=data_transform, gt_transform=gt_transform, depth=False)
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    encoder_name = args.backbone
    # encoder_name = 'dinov2reg_vit_small_14'
    # encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    # encoder_name = 'dinov2_vit_base_14'
    # encoder_name = 'dino_vit_base_16'
    # encoder_name = 'ibot_vit_base_16'
    # encoder_name = 'mae_vit_base_16'
    # encoder_name = 'beitv2_vit_base_16'
    # encoder_name = 'beit_vit_base_16'
    # encoder_name = 'digpt_vit_base_16'
    # encoder_name = 'deit_vit_base_16'

    if args.lc == 0:  # layer to layer
        fuse_layer_encoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
        fuse_layer_decoder = [[0], [1], [2], [3], [4], [5], [6], [7]]
    elif args.lc == 1:  # one group
        fuse_layer_encoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    elif args.lc == 2:  # two group
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    elif args.lc == 3:  # three group
        fuse_layer_encoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1, 2], [3, 4, 5], [6, 7]]
    elif args.lc == 4:  # four group
        fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
        fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
    elif args.lc == 11:  # sparse, one layer
        fuse_layer_encoder = [[7]]
        fuse_layer_decoder = [[7]]
    elif args.lc == 12:  # sparse, two layers
        fuse_layer_encoder = [[3], [7]]
        fuse_layer_decoder = [[3], [7]]
    elif args.lc == 14:  # sparse, four layers
        fuse_layer_encoder = [[1], [3], [5], [7]]
        fuse_layer_decoder = [[1], [3], [5], [7]]
    else:
        raise "loose constraint value not supported"

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
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    dropout = args.dropout
    bottleneck.append(nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)))
    bottleneck.append(nn.Sequential(nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
                                    nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout)))

    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=partial(LinearAttention2, eps=1e-8) if args.la else Attention)
        # blk = ConvBlock(dim=embed_dim, kernel_size=1, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = Dinomaly(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                     remove_class_token=False,
                     fuse_layer_encoder=fuse_layer_encoder,
                     fuse_layer_decoder=fuse_layer_decoder,
                     context_aware_recenter=args.cr)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    model.init_weights()

    optimizer = StableAdamW([{'params': bottleneck[0].parameters(), 'lr': 2e-4},
                             {'params': bottleneck[1].parameters()},
                             {'params': decoder.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
    lr_scheduler = WarmupCosineScheduler(optimizer, final_ratio=args.lr_decay_ratio, total_epochs=total_iters,
                                         warmup_epochs=100)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for (img, infra), label in train_dataloader:
            img = img.to(device)
            infra = infra.to(device)

            en, de = model(torch.cat([img, infra], dim=0))

            p_final = args.ll_ratio
            p = min(p_final * it / 1000, p_final)
            if args.ll:
                loss = global_cosine_hm_percent(en, de, p=p, factor=args.ll_factor)
            else:
                loss = global_cosine(en, de)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % 5000 == 0:
                # torch.save(ader_model.state_dict(), os.path.join(args.save_dir, args.save_name, 'ader_model.pth'))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_rgb_list, ap_px_rgb_list, f1_px_rgb_list, aupro_px_rgb_list = [], [], [], []
                auroc_px_infra_list, ap_px_infra_list, f1_px_infra_list, aupro_px_infra_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch_rgbinfra(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp = results['object_level']
                    auroc_px_rgb, ap_px_rgb, f1_px_rgb, aupro_px_rgb = results['pixel_rgb']
                    auroc_px_infra, ap_px_infra, f1_px_infra, aupro_px_infra = results['pixel_infra']

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)

                    auroc_px_rgb_list.append(auroc_px_rgb)
                    ap_px_rgb_list.append(ap_px_rgb)
                    f1_px_rgb_list.append(f1_px_rgb)
                    aupro_px_rgb_list.append(aupro_px_rgb)

                    auroc_px_infra_list.append(auroc_px_infra)
                    ap_px_infra_list.append(ap_px_infra)
                    f1_px_infra_list.append(f1_px_infra)
                    aupro_px_infra_list.append(aupro_px_infra)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                        'RGB-AUROC:{:.4f}, RGB-AP:{:.4f}, RGB-F1:{:.4f}, RGB-AUPRO:{:.4f}, '
                        'Infra-AUROC:{:.4f}, Infra-AP:{:.4f}, Infra-F1:{:.4f}, Infra-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px_rgb, ap_px_rgb, f1_px_rgb, aupro_px_rgb,
                            auroc_px_infra, ap_px_infra, f1_px_infra, aupro_px_infra))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f},'
                    ' P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f},'
                    'Infra-AUROC:{:.4f}, Infra-AP:{:.4f}, Infra-F1:{:.4f}, Infra-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_rgb_list), np.mean(ap_px_rgb_list), np.mean(f1_px_rgb_list),
                        np.mean(aupro_px_rgb_list),
                        np.mean(auroc_px_infra_list), np.mean(ap_px_infra_list), np.mean(f1_px_infra_list),
                        np.mean(aupro_px_infra_list),
                    ))
                model.train()

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--data_path', type=str, default='/Data1/guojia/mvtec_anomaly_detection')
    parser.add_argument('--data_path', type=str, default='../MulSen_AD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='dinomaly2_mulsen_uni')
    parser.add_argument('--backbone', type=str, default='dinov2reg_vit_large_14')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for Noisy Bottleneck')
    parser.add_argument('--la', type=int, default=1,
                        help='Linear Attention. 1 for yes, 0 for no.')
    parser.add_argument('--lc', type=int, default=2,
                        help='Loose Constraint. 1 for 1 group, 2 for 2 group, 0 for layer-to-layer.')
    parser.add_argument('--ll', type=int, default=1,
                        help='Loose Loss. 1 for yes, 0 for no.')
    parser.add_argument('--ll_ratio', type=float, default=0.9,
                        help='The ratio of discarded regions in Loose Loss. 0.9 (90%) by default.')
    parser.add_argument('--ll_factor', type=float, default=0.1,
                        help='The ratio gradients of the discarded regions. 0.1 by default.')
    parser.add_argument('--cr', type=int, default=1,
                        help='Context-aware recentering. 1 for yes, 0 for no.')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--total_iters', type=int, default=20000)
    parser.add_argument('--lr_decay_ratio', type=float, default=1.)
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()
    #
    item_list = ["capsule", "cotton", "cube", "spring_pad", "screw", "screen", "piggy", "nut", "flat_pad",
                 'plastic_cylinder', "zipper", "button_cell", "toothbrush", "solar_panel", "light",
                 ]

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    print_fn(args)
    train(item_list, args)
