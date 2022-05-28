
import torch
from utils.Myloss import L_exp

def f_function(x,a,b):
    fx = torch.exp(b*(1-x.pow(a)))
    return fx

def g_function(img,k,a,b):

    fk = f_function(k,a,b)
    gx = fk*(img.pow(k.pow(a)))
    return gx

# def y_fx(x,mean_x):
#     if mean_x<0.1:
#         y = 2*x
#     elif mean_x<0.8:
#         y = 1.1616* (x ** 0.4 ) -0.2624
#     else:
#         y = x
#     return y
# def y_fx(x,mean_x):
#     if mean_x<0.3:
#         y = 2*x
#     elif mean_x<0.8:
#         y = 0.6738* (x ** 0.4 )+0.1837
#     else:
#         y = x
#     return y
def y_fx(x,mean_x):
    if mean_x<0.1012:
        y1 = torch.clamp(x,min=0.00001)
        y = y1**0.4
        # y = 0.4
    elif mean_x<0.2277:
    # elif mean_x<0.278855:
        y = 1.5811*x+0.24
        # y = x**0.4
    else:
        y = 0.6
    return y
# def y_fx2(x,mean_x):
#     y = 2.5*((mean_x-0.6)**2)+0.6
#     return y
#
# def y_fx3(x,mean_x):
#     if mean_x < 0.36:
#         y1 = torch.clamp(x, min=0.00001)
#         y = y1 ** 0.5
#         # y = 0.4
#     else:
#         y = 0.6
#     return y
def y_fx2(x,mean_x):
    y = 2.5*((mean_x-0.6)**2)+0.6
    return y

def y_fx3(x,mean_x):
    if mean_x < 0.278855:
        y1 = torch.clamp(x, min=0.00001)
        y = y1 ** 0.4
        # y = 0.4
    else:
        y = 0.6
    return y
def Correct_function(img_tensor,img_res):
    # [a, b]=[ -0.3293 ,1.1258]
    # image_hsv = rgb_to_hsv(img_tensor.squeeze(0)).unsqueeze(0)
    # image_v = image_hsv[:, 2, :, :].unsqueeze(1)
    image_v = torch.max(img_tensor,dim=1)[0].unsqueeze(0)
    V_mean = torch.mean(image_v)
    if V_mean<0.6:
        J = y_fx3(V_mean,V_mean)
    else:
        J = y_fx2(V_mean,V_mean)
    # J = 1/(1+torch.exp(-V_mean))
    # J = V_mean**(0.4)
    # J = y_fx3(V_mean, V_mean)
    Lexp = L_exp(16,0.6)
    J_exp = 2*Lexp(img_res)

    return J_exp
# def Correct_function(V):
#     [a, b]=[ -0.3293 ,1.1258]
#     V_mean = torch.mean(V)
#     kRatio =torch.clamp(V_mean,min=1/7)
#     J = g_function(V.squeeze(1),kRatio.pow(1/1.5),a,b)
#     J = J.unsqueeze(1)
#
#     return J

# def Correct_function(T,R,img,Noise):
#     [a, b]=[ -0.3293 ,1.1258]
#     # [a, b] = [-0.25, 1.5]
#     img_re = img-Noise
#
#     # T = T.squeeze(0)
#     # T,_ = torch.max(T,dim=1)
#
#     T_hsv = rgb_to_hsv(T.squeeze(0)).unsqueeze(0)
#     T1 = T_hsv[:,2,:,:]
#     # T1 = T[:,0,:,:]
#     T_s = T1.reshape(-1)
#     # print(T_s)
#     T_sort,_ = torch.sort(T_s,dim=-1)
#     # print(T_sort)
#     shape_T = T_sort.shape[0]
#     index_1 = math.floor(0.1*shape_T)
#
#     index_2 = math.floor(0.9*shape_T)
#     ratioMin = T_sort[index_1].data
#     ratioMax = T_sort[index_2].data
#     # hsv_T=rgb2hsv(T, conf.half_size,conf.img_size)
#     # T=hsv_T[:,2,:,:]
#     # [channels,height,width] = img.size()
#
#     # kRatio = T1,min=1/7)
#     kRatio =torch.clamp(1 / (T1 + 0.0001),min=1/ratioMax,max=1/ratioMin)
#     hsv = rgb_to_hsv(img_re.squeeze(0)).unsqueeze(0)
#     H = hsv[:,0,:,:].unsqueeze(1)
#     S = hsv[:,1,:,:].unsqueeze(1)
#     V = hsv[:,2,:,:]
#     J = g_function(V,kRatio.pow(1/1.5),a,b)
#     J = J.unsqueeze(1)
#     res_hsv = torch.cat([H,S,J],dim=1)
#     res_img = hsv_to_rgb(res_hsv.squeeze(0)).unsqueeze(0)
#     # res_img = g_function(img_re,kRatio.pow(1/1.5),a,b)
#     return res_img,H,S,J,V.unsqueeze(1)

