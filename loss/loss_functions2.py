import torch
import torch.nn.functional as F
import conf
import numpy as np
from torch.autograd import Variable


def reconstruction_loss(image, illumination, reflectance, noise):
    reconstructed_image = illumination*reflectance+noise
    return torch.norm(image-reconstructed_image, 1)


def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w
    # return gradient2_h,  gradient2_w
def Sobel_gradient(im):
    sobel_kernel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')  #
    sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
    weight_x = Variable(torch.from_numpy(sobel_kernel_x))
    weight_x = weight_x.to(conf.device)
    edge_detect_x = F.conv2d(Variable(im), weight_x)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    weight = weight.to(conf.device)
    edge_detect_y = F.conv2d(Variable(im), weight)
    edge_detect_x = F.pad(edge_detect_x , [1, 1, 1, 1], 'replicate')
    edge_detect_y  = F.pad(edge_detect_y , [1, 1, 1, 1], 'replicate')
    # edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect_x,edge_detect_y
def Sobel_conv2dy(im):
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    weight = weight.to(conf.device)
    edge_detect = F.conv2d(Variable(im), weight)
    # edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def First_order_gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,1:,:]-img[:,:,:height-1,:])
    gradient_w = (img[:, :, :, 1:] - img[:, :, :, :width-1])
    gradient_h = F.pad(gradient_h, [0, 0, 0, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 0, 0, 0], 'replicate')
    # gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    # gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    # gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    # gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h, gradient_w


def normalize01(img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)


def gaussianblur3(input):
    slice1 = F.conv2d(input[:,0,:,:].unsqueeze(1), weight=conf.gaussian_kernel, padding=conf.g_padding)
    slice2 = F.conv2d(input[:,1,:,:].unsqueeze(1), weight=conf.gaussian_kernel, padding=conf.g_padding)
    slice3 = F.conv2d(input[:,2,:,:].unsqueeze(1), weight=conf.gaussian_kernel, padding=conf.g_padding)
    x = torch.cat([slice1,slice2, slice3], dim=1)
    return x

def color_loss2(true_reflect,res_reflect):
    b, c, h, w = true_reflect.shape
    true_reflect_view = true_reflect.view(b, c, h * w).permute(0, 2, 1)
    pred_reflect_view = res_reflect.view(b, c, h * w).permute(0, 2, 1)  # 16 x (512x512) x 3
    true_reflect_norm = torch.nn.functional.normalize(true_reflect_view, dim=-1)
    pred_reflect_norm = torch.nn.functional.normalize(pred_reflect_view, dim=-1)
    cose_value = true_reflect_norm * pred_reflect_norm
    cose_value = torch.sum(cose_value, dim=-1)  # 16 x (512x512)  #
    # print(cose_value.min(), cose_value.max())
    color_loss = torch.mean(1 - cose_value)
    # print(color_loss)
    return color_loss