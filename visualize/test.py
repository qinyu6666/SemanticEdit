import argparse
import shutil
import numpy as np
import imageio
import torch

from visualize.utils import generate, cubic_spline_interpolate
import pprint

from models.semantic_stylegan import SemanticGenerator, DualBranchDiscriminator

def make_model1(args, verbose=True):
    if verbose:
        print(f"Initializing model with arguments:")
        pprint.pprint(vars(args))
    model = SemanticGenerator(args.size, args.latent, args.n_mlp,
        channel_multiplier=args.channel_multiplier, seg_dim=args.seg_dim,
        local_layers=args.local_layers, local_channel=args.local_channel,
        base_layers=args.base_layers, depth_layers=args.depth_layers,
        coarse_size=args.coarse_size, coarse_channel=args.coarse_channel, min_feat_size=args.min_feat_size,
        residual_refine=args.residual_refine, detach_texture=args.detach_texture,
        transparent_dims=args.transparent_dims)
    return model

def main():
    print('start 5.1')
    ckpt = torch.load('../pretrained/CelebAMask-HQ-512x512.pt')
    model = make_model1(ckpt['args'])
    model.to('cuda')
    model.eval()
    model.load_state_dict(ckpt['g_ema'])
    mean_latent = model.style(torch.randn(1000, model.style_dim, device='cuda')).mean(0)

    print("Generating original image ...")
    with torch.no_grad():

        styles = model.style(torch.randn(1, model.style_dim, device='cuda'))
        styles = .7 * styles + (1 - .7) * mean_latent.unsqueeze(0)  # [1，512]

        if styles.ndim == 2:
            assert styles.size(1) == model.style_dim
            styles = styles.unsqueeze(1).repeat(1, model.n_latent, 1)  # [1，28，512]
        images, segs = generate(model, styles, mean_latent=mean_latent, randomize_noise=False)


if __name__ == '__main__':
    main()


