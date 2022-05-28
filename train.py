from torch.utils.data import  DataLoader
import torch
from torch.autograd import Variable
from utils.dataloader import SingleClassDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.Net import CoarseNet, FineNet,Vgg16
from utils.Func_Correct import  Correct_function
from loss.loss_functions2 import First_order_gradient
import os
import torch
import torch.nn as nn
from utils.ssim import SSIM

vgg = Vgg16()
vgg.cuda()
Tensor = torch.cuda.FloatTensor
device1 = torch.device('cuda')
coarse_net = CoarseNet().to(device1)
refine_net = FineNet().to(device1)
trainDataSet = SingleClassDataset(file_path="./LOLdataset/train")
trainDataSet_loader = DataLoader(dataset=trainDataSet, batch_size=4, shuffle=True)
coarse_optim = torch.optim.Adam(coarse_net.parameters(), lr=1e-3)
refine_optim = torch.optim.Adam(refine_net.parameters(), lr=1e-4)

def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    vgg.cuda()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))

    return vgg


def compute_vgg_loss(enhanced_result, input_high):
    instance_norm = nn.InstanceNorm2d(512, affine=False)
    vgg = load_vgg16("./model/")
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    img_fea = vgg(enhanced_result)
    target_fea = vgg(input_high)

    # loss = torch.mean((instance_norm(img_fea) - instance_norm(target_fea)) ** 2)
    loss = torch.mean(torch.abs(instance_norm(img_fea) - instance_norm(target_fea)))
    return loss

for epoch in range(100):
    running_loss = 0.0
    sum_loss = 0.0
    for i, (low_im,high_im) in enumerate(trainDataSet_loader, 0):
        low_im, high_im = low_im.cuda(), high_im.cuda()
        res_img, illumination, noise  = coarse_net(low_im)
        loss_v = 1 * Correct_function(low_im, res_img)
        max_rgb = torch.max(low_im,dim=1)[0].unsqueeze(1)
        delta_illux, delta_illuy = First_order_gradient(illumination)
        loss_illu =  0.01* torch.norm(delta_illux, 1) +  0.01* torch.norm(delta_illuy, 1)
        loss_noise = 500* torch.mean(noise**2)
        loss = loss_v + loss_illu + loss_noise

        coarse_optim.zero_grad()
        loss.backward()
        coarse_optim.step()
        sum_loss += loss
    print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss ))
    if (epoch + 1) % 20 == 0:
        torch.save(coarse_net.state_dict(), './checkpoint' + '/decom_' + str(epoch+1) + '.pth')

torch.save(coarse_net.state_dict(), './checkpoint' + '/decom_final.pth')

for epoch in range(200):
    running_loss = 0.0
    sum_loss = 0.0
    for i, (low_im, high_im) in enumerate(trainDataSet_loader, 0):
        low_im, high_im = low_im.cuda(), high_im.cuda()
        res_1,_,_ = coarse_net(low_im)
        res_img= refine_net(low_im,res_1.detach())
        loss_l1 = torch.mean(torch.abs(res_img - high_im))
        loss_ssim = 1-SSIM(res_img,high_im)
        loss_refl = 1*compute_vgg_loss(res_img, high_im)
        loss = 2*loss_l1+2*loss_ssim+loss_refl
        refine_optim.zero_grad()
        loss.backward()
        refine_optim.step()
        sum_loss += loss
    print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss ))
    if (epoch + 1) % 20 == 0:
        torch.save(refine_net.state_dict(), './checkpoint' + '/refine_' + str(epoch+1) + '.pth')
torch.save(refine_net.state_dict(), './checkpoint' + '/refine_final.pth')
