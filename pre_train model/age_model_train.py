import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import  random_split
import sys
sys.path.append(".")
from utils.age_dataloader import AgeDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.Scheduler import GradualWarmupScheduler
from ddpm_diffusion.Condition_Model import UNet1,Agecnn
import torch.nn.functional as F

def ones(matrix):
    batch = matrix.size(0)
    for i in range(batch):
        # 计算最大值
        max_val = torch.max(matrix[i,:,:])
        # 计算最小值
        min_val = torch.min(matrix[i,:,:])
        matrix[i,:,:] = 2 * (matrix[i,:,:] - min_val) / (max_val - min_val) - 1
    return matrix
def train_model(modelConfig):
    loss_get=[]
    val_loss_get=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")
    print(f"device: {device}")
    if torch.cuda.device_count() > 0:
        print(f"{torch.cuda.device_count()} GPU(s)")
        if torch.cuda.device_count() > 1:
            print("multi-GPU training is currently not supported.")

    # Create instances of the dataset and split into scripts and validation sets
    '''修改'''
    dataset = AgeDataset(root_dir=modelConfig["root_dir"]) #返回两个数

    # Assuming you want to use 80% of the data for scripts and 20% for validation 分开训练集与预测集
    train_size = 600 #奇怪点
    val_size = 19 #奇怪点
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for scripts and validation为脚本和验证创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=modelConfig["batch_size"], shuffle=True)#, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=modelConfig["batch_size"], shuffle=False)#( num_workers=num_workers)
    ckpt = torch.load("../model/best_unet_model_MCI(600)_2.pth")
    Unet = UNet1()
    Unet.load_state_dict(ckpt)
    trainer = Agecnn(Unet).to(device)


    optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["num_epochs"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["num_epochs"] // 20,
        after_scheduler=cosineScheduler)
    best_val_loss = float('inf')
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['train loss'])
    # writer = SummaryWriter('record')

    for epoch in range(modelConfig["start_epoch"] - 1, modelConfig["num_epochs"]):
        trainer.train()
        batch_idx = 0
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for i,batch in enumerate(tqdmDataLoader):
                batch_idx += 1
                source_images,age = batch
                source_images = source_images.to(device)
                age = age.to(device)
                optimizer.zero_grad()
                laten = trainer(source_images).to(torch.float64)
                age=age.unsqueeze(1)
                loss = F.mse_loss(laten, age,reduction='none').mean()
                loss.backward()
                loss_get.append(loss)
                torch.nn.utils.clip_grad_norm_(
                    trainer.parameters(), modelConfig["grad_clip"])
                optimizer.step()

                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": epoch,
                    "loss: ": loss.item(),
                    "img shape: ": source_images.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })

            warmUpScheduler.step()
            torch.save(trainer.state_dict(), '../model/recent_age_model_MCI(600).pth')
            # Validation
            if epoch % modelConfig["val_freq"] == 0:
                trainer.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_source_images, val_age = val_batch

                        val_source_images = val_source_images.to(device)
                        val_age = val_age.to(device)
                        val_laten = trainer(val_source_images).to(torch.float64)
                        val_age = val_age.unsqueeze(1)
                        loss = F.mse_loss(val_laten, val_age).mean()
                        total_val_loss += loss

                average_val_loss = total_val_loss / len(val_dataloader)
                val_loss_get.append(average_val_loss)
                print(f'Validation Epoch [{epoch + 1}/{modelConfig["num_epochs"]}], Average Loss: {average_val_loss}')
                if average_val_loss < best_val_loss:
                    best_val_loss = average_val_loss
                    torch.save(trainer.state_dict(), '../model/best_age_model_MCI(600).pth')
            if epoch % 20 == 0:
                loss_get0 = [l.cpu().detach().numpy() for l in loss_get]
                val_loss_get0 = [l.cpu().detach().numpy() for l in val_loss_get]
                loss_get0 = np.array(loss_get0)
                val_loss_get0 = np.array(val_loss_get0)
                np.save('../loss_data/age_loss_get', loss_get0)
                np.save('../loss_data/age_val_loss_get', val_loss_get0)



modelConfig = {
    "start_epoch": 2,
    "num_epochs": 1000,
    "val_freq": 10,
    "batch_size": 10,
    "channel": 128,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-5,
    "multiplier": 10,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "img_size": 32,
    "grad_clip": 1.,
    "training_load_weight": None,
    "root_dir":r'../dataset',
}
train_model(modelConfig)

