import os
import shutil
import copy
import math
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from imageio import imwrite, mimwrite
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
import sys
from visualize.utils import generate
sys.path.append('D:\\item2\\SemanticStyleGAN-main/models')
sys.path.append('D:\\item2\\SemanticStyleGAN-main/criteria')
sys.path.append('D:\\item2\\SemanticStyleGAN-main')

from criteria.lpips import lpips

from models import make_model
from visualize.utils import tensor2image, tensor2seg,tensor2segmaskeye

from spiga000.inference.config import ModelConfig
from spiga000.inference.framework import SPIGAFramework
from spiga000.demo.visualize.plotter import Plotter



def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def get_transformation(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transform


def calc_lpips_loss(im1, im2):
    img_gen_resize = F.adaptive_avg_pool2d(im1, (256, 256))
    target_img_tensor_resize = F.adaptive_avg_pool2d(im2, (256, 256))
    p_loss = percept(img_gen_resize, target_img_tensor_resize).mean()
    return p_loss

# def scale_landmark(landmarks, scale_factor= 1.2):
#     center = torch.mean(landmarks, axis=0)
#     shifted_landmarks = landmarks - center
#
#     scaled_shifted_landmarks = shifted_landmarks * scale_factor
#
#     scaled_landmarks = scaled_shifted_landmarks + center
#
#     return scaled_landmarks

def optimize_latent(args, g_ema, target_img_tensor):
    noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    for noise in noises:
        noise.requires_grad = True

    # initialization
    with torch.no_grad():
        noise_sample = torch.randn(10000, 512, device=device)
        latent_mean = g_ema.style(noise_sample).mean(0)
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(args.batch_size, 1)  # 1,512
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)  # 1,28,512

    latent_in.requires_grad = True
    if args.no_noises:
        optimizer = optim.Adam([latent_in], lr=args.lr)
    else:
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    latent_path = [latent_in.detach().clone()]
    pbar = tqdm(range(args.step))
    for i in pbar:
        optimizer.param_groups[0]['lr'] = get_lr(float(i) / args.step, args.lr)

        img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises)

        p_loss = calc_lpips_loss(img_gen, target_img_tensor)
        mse_loss = F.mse_loss(img_gen, target_img_tensor)
        n_loss = torch.mean(torch.stack([noise.pow(2).mean() for noise in noises]))

        if args.w_plus == True:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(0).repeat(latent_in.size(0), g_ema.n_latent, 1))
        else:
            latent_mean_loss = F.mse_loss(latent_in, latent_mean.repeat(latent_in.size(0), 1))

        # main loss function
        loss = (n_loss * args.noise_regularize +
                p_loss * args.lambda_lpips +
                mse_loss * args.lambda_mse +
                latent_mean_loss * args.lambda_mean)

        pbar.set_description(
            # f'perc: {p_loss.item():.4f} '
            f'noise: {n_loss.item():.4f} mse: {mse_loss.item():.4f}  latent: {latent_mean_loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # noise_normalize_(noises)
        latent_path.append(latent_in.detach().clone())

    return latent_path, noises


# def optimize_weights(args, g_ema, target_img_tensor, latent_in, noises=None):
#     for p in g_ema.parameters():
#         p.requires_grad = True
#     optimizer = optim.Adam(g_ema.parameters(), lr=args.lr_g)
#
#     pbar = tqdm(range(args.finetune_step))
#     for i in pbar:
#         img_gen, _ = g_ema([latent_in], input_is_latent=True, randomize_noise=False, noise=noises)
#
#         # p_loss = calc_lpips_loss(img_gen, target_img_tensor)
#         mse_loss = F.mse_loss(img_gen, target_img_tensor)
#
#         # main loss function
#         # loss = (p_loss * args.lambda_lpips +
#         #         mse_loss * args.lambda_mse
#         #         )
#         #
#         # pbar.set_description(f'perc: {p_loss.item():.4f} mse: {mse_loss.item():.4f}')
#
#         # optimizer.zero_grad()
#         # loss.backward()
#         # optimizer.step()
#
#     return g_ema


def get_landmark(img, p):
    """get landmark with dlib
    :return: np.array shape=(98, 2)
    """
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #
    # dets = face_cascade.detectMultiScale(img,scaleFactor = 1.1, minNeighbors = 5)

    # for k, det in enumerate(dets):
    #     feature = p.inference(img, [det])
    feature = p.inference(img)

    # lm = np.array(feature['landmarks'][0])
    lm = feature['landmarks'][0]

    return lm


def split_ldm(ldm):
    x = []
    y = []
    for p in ldm:
        x.append(p[0])
        y.append(p[1])
    return x,y

# def optimize_lantentface(args, g_ema, w_latent,origin_tensor_left,origin_tensor_right,face_predictor,origin_img_tensor,seg_gen):
def optimize_lantentface(args, g_ema, w_latent, eye_width_target, face_predictor,origin_img_tensor, seg_gen):

    # W向量 + W_deta向量 一起生成
    w_latent_deta = torch.zeros((512), device='cuda') # TODO:28
    # w_latent_deta = torch.randn(512, device='cuda')
    # w_latent_deta = torch.zeros((1,28,512), device='cuda') # TODO:28
    eye_width_target = eye_width_target.detach() # 左眼x[60]
    # origin_tensor_right = origin_tensor_right.detach() # 右眼角 x[72]
    w_latent.requires_grad = False
    w_latent_deta.requires_grad = True
    optimizer = optim.Adam([w_latent_deta], lr=args.lr)
    # latent_path = [w_latent_deta.detach().clone()]
    latent_path = [w_latent_deta]

    # noises = g_ema.render_net.get_noise(noise=None, randomize_noise=False)
    # for noise in noises:
    #     noise.requires_grad = True

    pbar = tqdm(range(args.step))
    for i in pbar:
        w_latent = w_latent.detach()
        optimizer.param_groups[0]['lr'] = get_lr(float(i) / args.step, args.lr)

        w_latent[0,6] = w_latent[0,6] + w_latent_deta #TODO
        # img_gen_new, segs_new = g_ema([w_latent + w_latent_deta], input_is_latent=True, randomize_noise=False)
        # w_latent = [w_latent + w_latent_deta]
        img_gen_new, segs_new = g_ema([w_latent], input_is_latent=True, randomize_noise=False)






        # 用于看效果的，可以去掉
        img_gen_np1 = tensor2image(img_gen_new.clone().detach()).squeeze()
        # 用于看效果的，可以去掉
        img_gen_new = img_gen_new.permute(0, 2, 3, 1).squeeze()
        img_gen_new = img_gen_new * 127.5 + 127.5   # 512，512，3
        # img_gen_newl = img_gen_newl.detach()  # 使 required_grad = False，才可以.numpy()
        segs_new = tensor2segmaskeye(segs_new)
        seg_mask_all = seg_gen.int() & segs_new.int() # 1，512，512
        ldm_new = get_landmark(img_gen_new, face_predictor)
        # 看一看效果,可以去掉
        landmarkss = np.array(ldm_new.clone().cpu().detach())
        canvas1 = copy.deepcopy(img_gen_np1)
        canvas1 = np.ascontiguousarray(canvas1)
        ploter = Plotter()
        canvas1 = ploter.landmarks.draw_landmarks(canvas1, landmarkss)
        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        axs.imshow(canvas1)
        #结束看效果，可以去掉

        x_new, y_new = split_ldm(ldm_new)
        x_new_lefteye = x_new[60]
        x_new_righteye = x_new[72]
        # x_new_lefteye = y_new[62] + y_new[61]
        # x_new_righteye = y_new[69] + y_new[70]

        # ldm_ref_eyeleft_new = [ldm_new[i] for i in range(60,68)]
        # l2_loss_lefteye = F.mse_loss(eye_width_target,ldm_ref_eyeleft_new)

        eye_width_new = (x_new[64] - x_new[60]) + (x_new[72] - x_new[68])


        l1_loss_eye_width = F.l1_loss(eye_width_new,eye_width_target)
        # l1_loss_righteye = F.l1_loss(x_new_righteye,origin_tensor_right)

        # 两眼去除，然后再MSE，
        origin_img_tensor_no_eye = origin_img_tensor.permute(2,0,1) * seg_mask_all.cuda()  # new

        img_gen_newl_no_eye = img_gen_new.permute(2,0,1) * seg_mask_all.cuda()  # new

        # 看一看no_eye 原始的图片
        origin_img_tensor_no_eye_look = origin_img_tensor.permute(2,0,1) * seg_gen.cuda()
        origin_img_tensor_no_eye_look = origin_img_tensor_no_eye_look.permute(1,2,0).cpu().numpy()
        origin_img_tensor_no_eye_look = origin_img_tensor_no_eye_look.astype(np.uint8)
        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        axs.imshow(origin_img_tensor_no_eye_look)

        # 看一看no_eye 现在的图片
        img_gen_newl_no_eye_np = img_gen_newl_no_eye.permute(1, 2, 0).clone().detach().cpu().numpy()
        img_gen_newl_no_eye_np = img_gen_newl_no_eye_np.astype(np.uint8)
        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        axs.imshow(img_gen_newl_no_eye_np)

        mask_mse_loss = F.mse_loss(img_gen_newl_no_eye,origin_img_tensor_no_eye)  # 带有mask的  new

        # mask_loss = (F.mse_loss(img_gen_new, origin_img_tensor, reduction='none') * seg_mask_all.cuda()).mean()

        p_loss = calc_lpips_loss(img_gen_new.permute(2,0,1).unsqueeze(0), origin_img_tensor.permute(2,0,1).unsqueeze(0))

        # n_loss = torch.mean(torch.stack([noise.pow(2).mean() for noise in noises]))

        # w_norm_loss = F.normalize(w_latent_deta, dim=0).mean()  # ///////////////////////////////////////////
        w_norm_loss = torch.norm(w_latent_deta,dim=0)
        # main loss function
        # loss = (l1_loss * 0.1 + mask_loss * 50)
        # loss = l1_loss_lefteye * 0.1 + l1_loss_righteye * 0.1
        # loss = l1_loss_lefteye * 0.1 + l1_loss_righteye * 0.1 + mask_mse_loss * 50 + p_loss + n_loss * 10
        # loss = l1_loss_lefteye * 0.1 + l1_loss_righteye * 0.1 + mask_mse_loss * 30 + p_loss
        # loss = l1_loss_lefteye * 0.1 + l1_loss_righteye * 0.1 + w_norm_loss + p_loss
        # loss = l1_loss_eye_width * 0.1 + w_norm_loss + p_loss
        loss = l1_loss_eye_width * 0.1 + p_loss * 0.1

        # loss = l1_loss

        pbar.set_description(
            f' l1_mse: {l1_loss_eye_width.item():.4f} p_loss: {p_loss.item():.4f} mask_loss: {mask_mse_loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # noise_normalize_(noises)
        latent_path.append(w_latent_deta)

    return latent_path,img_gen_new

if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parse_boolean = lambda x: not x in ["False", "false", "0"]
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--imgdir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--no_noises', type=parse_boolean, default=True)
    parser.add_argument('--w_plus', type=parse_boolean, default=True, help='optimize in w+ space, otherwise w space')

    parser.add_argument('--save_steps', type=parse_boolean, default=False,
                        help='if to save intermediate optimization results')

    parser.add_argument('--truncation', type=float, default=1,
                        help='truncation tricky, trade-off between quality and diversity')

    parser.add_argument('--lr', type=float, default=0.03)# 默认值 0.1
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--step', type=int, default=300, help='latent optimization steps')
    parser.add_argument('--finetune_step', type=int, default=0,
                        help='pivotal tuning inversion (PTI) steps (200-400 should give good result)')
    parser.add_argument('--noise_regularize', type=float, default=10)
    parser.add_argument('--lambda_mse', type=float, default=0.1) # TODO 大
    parser.add_argument('--lambda_lpips', type=float, default=1.0) # TODO 大
    parser.add_argument('--lambda_mean', type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    print("Loading model ...")
    ckpt = torch.load(args.ckpt)
    # ckpt = torch.load('pretrained/CelebAMask-HQ-512x512.pt')

    g_ema = make_model(ckpt['args'])
    g_ema.to(device)
    g_ema.eval()
    g_ema.load_state_dict(ckpt['g_ema'])

    percept = lpips.LPIPS(net_type='vgg').to(device)

    img_list = sorted(os.listdir(args.imgdir))

    transform = get_transformation(args)

    for image_name in img_list:
        img_path = os.path.join(args.imgdir, image_name)

        # load target image
        target_pil = Image.open(img_path).resize((args.size, args.size), resample=Image.LANCZOS)
        origin_img_tensor = transform(target_pil).unsqueeze(0).to(device)
        #
        # #先看一看
        # img_tensor_new = origin_img_tensor.clone().squeeze(0).permute(1, 2, 0)
        # img_tensor_np = img_tensor_new.detach().cpu().numpy().copy()  # 要加.copy()，不共享以前的
        # img_tensor_np = (img_tensor_np * 255).astype(np.uint8)
        # b, g, r = cv2.split(img_tensor_np)
        # img_tensor_np = cv2.merge([r, g, b])

        # img = cv2.imread(img_path)
        # img = cv2.resize(img, (args.size, args.size))
        # b, g, r = cv2.split(img)
        # img = cv2.merge([r, g, b])

        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        axs.imshow(target_pil)

        # origin_img_tensor = torch.tensor(img, dtype=torch.float)
        # origin_img_tensor = origin_img_tensor.permute(2,0,1).unsqueeze(0).to(device)

        latent_path, noises = optimize_latent(args, g_ema, origin_img_tensor)

        w_latent = latent_path[-1]
        # save results
        with torch.no_grad():
            img_gen, seg_gen = g_ema([w_latent], input_is_latent=True, randomize_noise=False, noise=noises)
            img_gen_np = tensor2image(img_gen).squeeze() # 这个tensor没梯度，可以不用detach()
            imwrite(os.path.join(args.outdir, 'recon/', image_name), img_gen_np)

        face_predictor = SPIGAFramework(ModelConfig('wflw'), gpus=[0])

        # ldm_ref = get_landmark(img_gen, face_predictor)


        # 新加的看看landmark效果
        # img_tensor_new = img_gen.clone().squeeze(0).permute(1, 2, 0)
        # img_tensor_np = img_tensor_new.detach().cpu().numpy().copy()  # 要加.copy()，不共享以前的
        # img_tensor_np = (img_tensor_np * 255).astype(np.uint8)
        #
        # img_tensor_np = cv2.cvtColor(img_tensor_np, cv2.COLOR_BGR2RGB)
        #
        # ldm_ref = get_landmark(img_gen, face_predictor)
        # ldm_ref_np = ldm_ref.clone().detach().cpu().numpy().astype(np.uint8)
        #
        # for b in ldm_ref_np:
        #     cv2.circle(img_tensor_np, (int(b[0]), int(b[1])), 2, (255, 255, 255), 4)
        # # cv2.imshow('img', img_tensor_np)
        # # cv2.waitKey(0)
        # fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        # axs.imshow(img_tensor_np)
        # print('aaa')
        # 结束看效果

        # 新加的看看landmark效果
        # img_tensor_new = img_gen.clone().squeeze(0).permute(1, 2, 0)
        # img_tensor_new = img_gen.squeeze(0).permute(1, 2, 0)
        # img_gen_np = tensor2image(img_gen).squeeze()

        # img_tensor_np = img_tensor_new.detach().cpu().numpy().copy()
        # img_tensor_np = img_tensor_np.astype(np.uint8)
        # b, g, r = cv2.split(img_tensor_np)
        # img_tensor_np = cv2.merge([r, g, b])

        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        axs.imshow(img_gen_np)

        # img_gen = img_gen[0].permute(1,2,0) # 一会儿试试
        # img_gen= img_gen.squeeze(0).permute(1, 2, 0)
        img_gen = img_gen.permute(0, 2, 3, 1).squeeze()
        img_gen = img_gen * 127.5 + 127.5

        # img_gen = torch.tensor(img_gen_np, dtype=torch.float).cuda() # 正常运行，就是梯度断了

        ldm_ref = get_landmark(img_gen, face_predictor)

        landmarkss = np.array(ldm_ref.cpu().detach())
        # landmarkss = ldm_ref.cpu().detach().numpy()

        canvas = copy.deepcopy(img_gen_np)
        canvas = np.ascontiguousarray(canvas)
        ploter = Plotter()
        canvas = ploter.landmarks.draw_landmarks(canvas,landmarkss)

        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
        axs.imshow(canvas)
        print('aaa')
        # 结束看效果

        x_cur, y_cur = split_ldm(ldm_ref)
        # left_eye_slider = 10
        # x_cur_60_left = x_cur[60] - left_eye_slider  # slider 左眼角 target
        # # x_cur_62_left = y_cur[61] + y_cur[62] + 2* left_eye_slider  # slider 左眼顶点 target
        #
        # x_cur_60_right = x_cur[72] + left_eye_slider
        # # x_cur_70_right = y_cur[70] + y_cur[69] + 2* left_eye_slider

        eye_width = (x_cur[64] - x_cur[60]) + (x_cur[72] - x_cur[68])
        slider_eye = 1.2
        eye_width_target = slider_eye * eye_width

        # ldm_ref_eyeleft = [ldm_ref[i] for i in range(60,68)]  扯
        #
        # ldm_ref_eyeleft_scaled = scale_landmark(ldm_ref_eyeleft, scale_factor=1.4) 扯

        seg_gen = tensor2segmaskeye(seg_gen)


        # latent_path,img_gen_newl = optimize_lantentface(args,g_ema,w_latent,x_cur_60_left,x_cur_60_right,face_predictor,img_gen,seg_gen)
        latent_path,img_gen_newl = optimize_lantentface(args,g_ema,w_latent,eye_width_target,face_predictor,img_gen,seg_gen)

        img_gen_newl = img_gen_newl.detach().cpu().clamp(0,255).numpy()
        img_gen_newl = img_gen_newl.astype(np.uint8)

        imwrite(os.path.join(args.outdir, 'recon/', image_name), img_gen_newl)


        # 可视化landmark看看，描点
        # 中间模型出现的图 可视化看看，还有lantent_path[-1]看看

        # 做loss数值差别大时，归一化









