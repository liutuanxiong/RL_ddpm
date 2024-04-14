import os
import torch
import math
from torch.utils.data import DataLoader
from utils.RLdataloader import CustomDataset
from ddpm_diffusion.Condition_Model import UNet1,Agecnn
from ddpm_diffusion.Diffusion_Model import UNet
from ddpm_diffusion.Diffusion import GaussianDiffusionSampler,GaussianDiffusionTrainer
import sys
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F

sys.path.append(".")
def ones(matrix):
    # 计算最大值
    max_val = torch.max(matrix)
    # 计算最小值
    min_val = torch.min(matrix)
    normalized_matrix =2* (matrix - min_val) / (max_val - min_val)-1
    return normalized_matrix

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
def calculate_psnr(original_images, predicted_images):
    """
    计算两批图像之间的平均PSNR值。

    参数:
        original_images (numpy.ndarray): 原始图像的数组，形状为(N, C, H, W)。
        predicted_images (numpy.ndarray): 预测图像的数组，形状为(N, C, H, W)。

    返回:
        float: 所有图像对的平均PSNR值。
    """
    # 确保输入是numpy数组
    original_images = np.asarray(original_images)
    predicted_images = np.asarray(predicted_images)

    # 初始化一个列表来存储每对图像的PSNR值
    psnr_scores = []

    # 遍历每对图像
    for i in range(original_images.shape[0]):
        # 计算MSE
        mse = np.mean((original_images[i] - predicted_images[i]) ** 2)
        if mse == 0:
            # 如果MSE为0，则两图像完全相同，PSNR有无穷大的情况
            return float('inf')

        # 计算最大像素值
        max_pixel = 1

        # 计算PSNR
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        # 将计算得到的PSNR值添加到列表中
        psnr_scores.append(psnr)

    # 计算平均PSNR值
    # average_psnr = np.mean(psnr_scores)
    psnr_scores = np.array(psnr_scores)
    return psnr_scores

def calculate_ssim(original_images, predicted_images):
    """
    计算两批图像之间的平均SSIM分数。

    参数:
        original_images (numpy.ndarray): 原始图像的数组，形状为(N, C, H, W)。
        predicted_images (numpy.ndarray): 预测图像的数组，形状为(N, C, H, W)。

    返回:
        float: 所有图像对的平均SSIM分数。
    """
    # 确保输入是numpy数组
    original_images = np.asarray(original_images)
    predicted_images = np.asarray(predicted_images)

    # 初始化一个列表来存储每对图像的SSIM分数
    ssim_scores = []

    # 遍历每对图像
    for i in range(original_images.shape[0]):
        # 计算当前对图像的SSIM分数
        # 注意，由于我们的图像只有一个通道，我们需要使用squeeze方法去除单通道维度
        score = ssim(original_images[i].squeeze(), predicted_images[i].squeeze(),
                     data_range=predicted_images[i].max() - predicted_images[i].min())

        # 将计算得到的SSIM分数添加到列表中
        ssim_scores.append(score)

    # 计算平均SSIM分数
    # average_ssim = np.mean(ssim_scores)
    ssim_scores = np.array(ssim_scores)
    return ssim_scores

def normalize(array):
    mean_val = np.mean(array)
    std_val = np.std(array)
    standardized_array = (array - mean_val) / std_val
    return standardized_array


def RL_scoring(pred,origin,target_age,device):
    global Agecnn
    Agecnn.to("cuda")
    pred = ones(pred)
    source_age_channel = torch.full_like(pred[:,:1, :, :], 0).to("cuda")
    target_age_channel = torch.full_like(pred[:,:1, :, :], 0).to("cuda")
    source_image = torch.cat([pred, source_age_channel, target_age_channel], dim=1)
    pred_age = Agecnn(source_image)
    target_age = target_age.unsqueeze(1)
    reward = F.mse_loss(pred_age,target_age,reduction='none')
    Agecnn.to("cpu")
    # origin = ones(origin)
    # pred_np = pred.detach().cpu().numpy()
    # origin_np = origin.detach().cpu().numpy()
    # psnr = calculate_psnr(pred_np,origin_np)
    # ssim = calculate_ssim(pred_np,origin_np)
    # reward = psnr*0.05 + ssim
    # reward = normalize(reward)
    # #可能加上一个标准化
    # return torch.tensor(reward,dtype=torch.float32).to(device)
    return reward



def calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t):
    std_dev_t = torch.clip(std_dev_t, 1e-6)
    log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
    return log_probs

def predict_xt_prev_mean_from_eps(x_t, t, eps):
    global coeff1,coeff2
    assert x_t.shape == eps.shape
    return (
    extract(coeff1, t, x_t.shape) * x_t -
    extract(coeff2, t, x_t.shape) * eps)


def p_mean_variance(laten, x_t, t):
    global posterior_var,model
    # device = x_t.device
    var = posterior_var
    var = extract(var, t, x_t.shape)
    eps = model(x_t, t,laten) #重点
    xt_prev_mean = predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
    return xt_prev_mean, var


def sd_sample(x_T, x,eta):#x:对应的提示图片 #重点内存优化
    global Unet
    Unet.to("cuda")
    x_t = x_T
    laten = Unet(x)
    Unet.to("cpu")

    all_step_preds, log_probs = [x_t], []

    # for i, time_step in enumerate(range(sampler.T-1,-1,-1)):
    for i, time_step in enumerate(range(10, -1, -1)):
        # print(time_step)
        # print_cuda()
        x_t = all_step_preds[-1].to("cuda")
        t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
        # print_cuda()#不变
        prev_sample_mean, var = p_mean_variance(laten, x_t=x_t, t=t)  # 重点
        # print_cuda()  # 减少3亿个
        std_dev_t = eta * var ** (0.5)
        # print_cuda()#不变
        if time_step > 0:
            noise_d = torch.randn_like(x_t)
            noise = (noise_d + noise_d.transpose(2, 3)) / math.sqrt(2)
            # print_cuda()#不变
        else:
            noise = 0

        x_t = prev_sample_mean + torch.sqrt(var) * noise
        # print_cuda()#不变
        log_probs.append(
            calculate_log_probs(x_t, prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim))))
        # print_cuda()#不变
        # mean与std有根据公式
        all_step_preds.append(x_t)
        # print_cuda()#不变
        del prev_sample_mean, x_t
        torch.cuda.empty_cache()

    return all_step_preds[-1],torch.stack(all_step_preds), torch.stack(log_probs)


def compute_loss(x_T, x, original_log_probs, advantages, clip_advantages, clip_ratio, eta):
    global Unet
    Unet.to("cuda")
    x_t=x_T[0]
    laten = Unet(x)
    # scheduler = pipe.scheduler
    # unet = pipe.unet
    # text_embeddings = pipe._encode_prompt(prompts, device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()
    # scheduler.set_timesteps(num_inference_steps, device=device)
    loss_value = 0.
    # print_cuda("out")
    for i, time_step in enumerate(range(10, -1, -1)):
        # print_cuda("p2 " + str(i) + " 1")
        clipped_advantages = torch.clip(advantages, -clip_advantages, clip_advantages).detach()
        # print_cuda()
        # print(time_step)
        # print_cuda("p2 " + str(i) + " 2")
        t = x_t.new_ones([x_T.shape[1], ], dtype=torch.long) * time_step
        # print_cuda("pin2 " + str(i) + " 3")
        prev_sample_mean, var = p_mean_variance(laten, x_t=x_T[i].detach(), t=t) #这里增加
        # print_cuda("p2 " + str(i) + " 4")
        std_dev_t = eta * var ** (0.5)
        # print_cuda("p2 " + str(i) + " 5")
        if time_step > 0:
            noise_d = torch.randn_like(x_t)
            noise = (noise_d + noise_d.transpose(2, 3)) / math.sqrt(2)
        else:
            noise = 0
        # print_cuda("p2 " + str(i) + " 6")
        x_t = prev_sample_mean + torch.sqrt(var) * noise
        # print_cuda("p2 " + str(i) + " 7")
        current_log_probs = calculate_log_probs(x_T[i+1].detach(), prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim)))
        # print_cuda("p2 " + str(i) + " 8")
        ratio = torch.exp(current_log_probs - original_log_probs[i].detach())  # this is the importance ratio of the new policy to the old policy
        # print_cuda("p2 " + str(i) + " 9")
        unclipped_loss = -clipped_advantages * ratio  # this is the surrogate loss
        # print_cuda("p2 " + str(i) + " 10")
        clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - clip_ratio,
                                                        1. + clip_ratio)  # this is the surrogate loss, but with artificially clipped ratios
        # print_cuda("p2 " + str(i) + " 11")
        loss = torch.max(unclipped_loss,clipped_loss).mean()  # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
        # print_cuda("p2 " + str(i) + " 12")
        loss.backward(retain_graph=True)  # 这里增加perform backward here, gets accumulated for all the timesteps
        # print_cuda("p2 " + str(i) + " 13")
        loss_value += loss.item()
        # print_cuda("pin " + str(i) + " end")
    return loss_value

def sample_and_calculate_rewards(x_T, x,eta,origin_t,target_age,device):
    preds,all_step_preds, log_probs = sd_sample(x_T, x,eta)
    imgs = ones(preds)
    rewards = RL_scoring(preds,origin_t, target_age,device)
    return imgs, rewards, all_step_preds, log_probs

def print_cuda(i):
    # 显示当前GPU的总内存和可用内存
    q = str(i)+":"
    print(q)
    print(torch.cuda.get_device_properties(0).total_memory)
    print(torch.cuda.memory_allocated(0))
    print(torch.cuda.memory_reserved(0))
    print(torch.cuda.get_device_properties(0).total_memory-torch.cuda.memory_reserved(0))
    print()
    # 打印更详细的内存使用报告
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))


modelConfig = {
    "state": "test",  # or eval
    "epoch":2,
    "num_inner_epochs": 2,
    "batch_size": 2,
    "num_samples_per_epoch":4,
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
    "clip_advantages": 2,
    "clip_ratio":1e-4,
    "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
    "training_load_weight": None,
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_0_.pt",
    "sampled_dir": "./SampledImgs/",
    "nrow": 8,
    "root_dir":r'../dataset',
}



device = modelConfig["device"]
beta_1 = modelConfig["beta_1"]
beta_T = modelConfig["beta_T"]
T = modelConfig["T"]
ckpt0 = torch.load("../model/best_unet_model_MCI(600)_2.pth", map_location=device)
Unet = UNet1().to(device)
Unet.load_state_dict(ckpt0)

Agecnn = Agecnn(UNet1())
ckpt_age = torch.load("../model/best_age_model_MCI(600).pth")
Agecnn.load_state_dict(ckpt_age)

model0 = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
             attn=modelConfig["attn"],
             num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
trainer = GaussianDiffusionTrainer(
    model0, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"])
ckpt = torch.load("../model/best_ddpm_model_MCI(600)_2.pth")
trainer.load_state_dict(ckpt)
print("model load weight done.")
model = trainer.model.to(device)
# sampler = GaussianDiffusionSampler(
#     Unet, trainer.model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
# #改改改改

#系数提出来
betas = torch.linspace(beta_1,beta_T,T).double()
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
coeff1 = torch.sqrt(1. / alphas)
coeff2 = coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar)
posterior_var = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)

coeff1 = coeff1.to(device)
coeff2 = coeff2.to(device)
posterior_var = posterior_var.to(device)

dataset = CustomDataset(root_dir=modelConfig["root_dir"])
train_dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True)
# print_cuda()
# for name, param in sampler.UNet1.named_parameters():
#     # print(name)
#     param.requires_grad = False
# # for name, param in sampler.named_parameters():
# #     print(name, param.requires_grad)

optimizer = torch.optim.Adam(model.parameters(), lr=modelConfig["lr"])
loss_get=[]

for epoch in range(modelConfig["epoch"]):
    with torch.no_grad():
        all_step_preds, log_probs, advantages, all_source_images = [], [], [], []
        # sampling `num_samples_per_epoch` images and calculating rewards
        for i, batch in enumerate(train_dataloader):
            print(i)
            source_images, target_images, origin_s, origin_t,target_age = batch
            source_images = source_images.to(device)
            target_age = target_age.to(device)
            # origin_t
            x_T = torch.randn(size=[source_images.shape[0], 1, 39, 39], device=device)
            x_T = (x_T + x_T.transpose(2, 3)) / math.sqrt(2)
            img, batch_advantages, batch_all_step_preds, batch_log_probs = sample_and_calculate_rewards(x_T,source_images,1, origin_t,target_age,device)
            all_step_preds.append(batch_all_step_preds)
            log_probs.append(batch_log_probs)
            advantages.append(batch_advantages)
            all_source_images.append(source_images)
            # print_cuda("p0 " + str(i))
        all_step_preds = torch.cat(all_step_preds, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        advantages = torch.cat(advantages)
        all_source_images = torch.cat(all_source_images, dim=0)
        all_step_preds = torch.chunk(all_step_preds, modelConfig["num_samples_per_epoch"] // modelConfig["batch_size"],
                                     dim=1)
        log_probs = torch.chunk(log_probs, modelConfig["num_samples_per_epoch"] // modelConfig["batch_size"], dim=1)
        advantages = torch.chunk(advantages, modelConfig["num_samples_per_epoch"] // modelConfig["batch_size"], dim=0)
        all_source_images = torch.chunk(all_source_images,
                                        modelConfig["num_samples_per_epoch"] // modelConfig["batch_size"], dim=0)
    # print_cuda("p1")
    # inner loop

    for inner_epoch in range(modelConfig["num_inner_epochs"]):
        model.train()
        with tqdm(range(len(all_step_preds))) as tqdmloader:
            for i in tqdmloader:
                # print_cuda("p2 " + str(i) + " 1")
                optimizer.zero_grad()
                # print_cuda("p2 " + str(i) + " 2")
                loss = compute_loss(all_step_preds[i], all_source_images[i],
                                    log_probs[i], advantages[i], modelConfig["clip_advantages"],
                                    modelConfig["clip_ratio"], 1)  # loss.backward happens inside
                # print_cuda("p2 " + str(i) + " 3")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
                # print_cuda("p2 " + str(i) + " 4")
                optimizer.step()
                show_epoch=epoch*modelConfig["num_inner_epochs"]+inner_epoch+1
                # print_cuda("p2 " + str(i) + " 5")
                # print(f"epoch:batch:RL_loss:{inner_epoch, i, loss}")
                tqdmloader.set_postfix(ordered_dict={
                    "epoch" : show_epoch,
                    "loss: ": loss,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                # print_cuda("p2 " + str(i) + " 6")
                loss_get.append(loss)
                # print_cuda("p2 "+str(i))
    del all_step_preds, all_source_images, log_probs
    torch.cuda.empty_cache()
    print_cuda("end")
    # optimizer.zero_grad()
    if epoch % 1 == 0:
        loss_get0 = [np.array(l) for l in loss_get]
        loss_get0 = np.array(loss_get0)
        np.save('RL_loss_get', loss_get0)
    torch.save(model.state_dict(),"rLmodel")