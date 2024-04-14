import math
import torch
import sys
sys.path.append(".")
from ddpm_diffusion.Condition_Model import UNet1
from typing import Dict
from ddpm_diffusion.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from ddpm_diffusion.Diffusion_Model import UNet
from utils.Scheduler import GradualWarmupScheduler
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from nilearn import plotting
from nilearn import datasets

atlas = datasets.fetch_atlas_msdl()
def Heatmap(matrix,target_age):
    labels = ['L Aud', 'R Aud', 'Striate', 'L DMN',
              'Med DMN', 'Front DMN', 'R DMN', 'Occ post',
              'Motor', 'R DLPFC', 'R Front pol', 'R Par',
              'R Post Temp', 'Basal', 'L Par', 'L DLPFC',
              'L Front pol', 'L IPS', 'R IPS', 'L LOC',
              'Vis', 'R LOC', 'D ACC', 'V ACC', 'R A Ins',
              'L STS', 'R STS', 'L TPJ', 'Broca', 'Sup Front S',
              'R TPJ', 'R Pars Op', 'Cereb', 'Dors PCC', 'L Ins',
              'Cing', 'R Ins', 'L Ant IPS', 'R Ant IPS']

    plt.figure(figsize=(10, 10))

    np.fill_diagonal(matrix, 0)
    plt.imshow(matrix, interpolation="nearest", cmap="RdBu_r",
               vmax=0.8, vmin=-0.8)
    plt.title(target_age)
    # plt.colorbar()
    # And display the labels
    x_ticks = plt.xticks(range(len(labels)), labels, rotation=90)
    y_ticks = plt.yticks(range(len(labels)), labels)
    plt.show()
    return

def conection(matrix,atlas,target_age):
    coords = atlas.region_coords
    plotting.plot_connectome(matrix, coords,
                             edge_threshold="80%", colorbar=True)
    plt.title(target_age)
    plotting.show()
    return
def ones(matrix):
    # 计算最大值
    max_val = torch.max(matrix)
    # 计算最小值
    min_val = torch.min(matrix)
    normalized_matrix =2* (matrix - min_val) / (max_val - min_val)-1
    return normalized_matrix


def eval(modelConfig: Dict,image,source_age,target_age,RL=False):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        input_size = (64, 64)
        cropped_image = image
        # 结束裁切
        # 加入年龄通道
        cropped_image = np.expand_dims(cropped_image, axis=0)
        cropped_image = torch.tensor(cropped_image)
        cropped_image = cropped_image.to(torch.float32)
        cropped_image = cropped_image.to(device)
        # orig_size = cropped_image.shape[:2]

        # cropped_image = transforms.ToTensor()(cropped_image)  # 转换为tensor格式
        source_age_channel = torch.full_like(cropped_image[:1, :, :], source_age / 100).to(device)
        target_age_channel = torch.full_like(cropped_image[:1, :, :], target_age / 100).to(device)
        input_tensor = torch.cat([cropped_image, source_age_channel, target_age_channel], dim=0).unsqueeze(0)

        # image = transforms.ToTensor()(image)
        ckpt0 = torch.load("../model/best_unet_model_MCI(1000)_1.pth", map_location=device)
        Unet = UNet1().to(device)
        Unet.load_state_dict(ckpt0)
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        if RL:
            ckpt = torch.load("../RL/RL_model/rLmodel_3",map_location=device)
            model.to(device)
            model.load_state_dict(ckpt)
        else:
            trainer = GaussianDiffusionTrainer(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
            ckpt = torch.load("../model/best_ddpm_model_MCI(1000)_2.pth", map_location=device)
            trainer.load_state_dict(ckpt)
            model=trainer.model
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            Unet,model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        noisyImage_d = torch.randn(
            size=[modelConfig["batch_size"], 1, 39, 39], device=device)
        noisyImage = (noisyImage_d + noisyImage_d.transpose(2,3))/math.sqrt(2)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage,input_tensor)
        sampledImgs = ones(sampledImgs)
        return sampledImgs
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])


modelConfig = {
    "state": "test",  # or eval
    "epoch": 10,
    "batch_size": 1,
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "img_size": 32,
    "grad_clip": 1.,
    "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
    "training_load_weight": None,
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_0_.pt",
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8
}
correlation_matrix=np.load("../dataset/I1563026_73.npy")
source_age = 73
target_age = 90
Heatmap(correlation_matrix,source_age)
for i in range(1,6):
    target_age = source_age+i*1
    out_matrix = eval(modelConfig,correlation_matrix, source_age,target_age,RL=False)
    out_matrix=torch.squeeze(out_matrix,0)
    one = ones(out_matrix)
    one = one.detach().cpu().numpy()
    out = out_matrix.detach().cpu().numpy()  # ndarry数据
    print("out:",out[0,:,:])
    print("one:",one[0,:,:])
    # print(out.shape)
    # print(out.shape)
    # 绘制热力图
    Heatmap(out[0,:,:],target_age)
    # Heatmap(one[0,:,:], target_age)
    # conection(out,atlas,target_age)
