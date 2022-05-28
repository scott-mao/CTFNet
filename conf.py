
import numpy as np

device = 'cuda'  # 'cpu' or 'cuda'
test_path =r'./testdata/test1.jpg'
# test_path =r'./LOLdataset/test/low/1.png'
result_path =r'./result/result1.jpg'
coarse_model_path = './checkpoint/coarse_final.pth'
fine_model_path = './checkpoint/refine_final.pth'
# vgg_factor = 1
# detail_factor =2
# channel = 16
# channel_d = 32
# illu_channel = 1
# rgb_channel = 3
eps = 1e-6
contact_channel=6
