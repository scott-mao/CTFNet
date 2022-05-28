
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import torch
from glob import glob


batch_size = 8


class SingleClassDataset(Dataset):

    def __init__(self, file_path):
        # 保证输入的是正确的路径
        if not os.path.isdir(file_path):
            raise ValueError("input file_path is not a dir")
        # self.file_path = r'./LOLdataset2/train/'
        self.file_path = file_path
        self.file_path_high = os.path.join(self.file_path, 'high/')
        self.file_path_low = os.path.join(self.file_path, 'low/')
        self.image_list_high = os.listdir(self.file_path_high)
        self.image_list_low = os.listdir(self.file_path_low)

        # 将PIL的Image转为Tensor
        self.transforms = T.ToTensor()

    def __getitem__(self, index):
        # 根据index获取图片完整路径
        image_path_high = os.path.join(self.file_path_high, self.image_list_high[index])
        # # 都图片并转为Tensor

        image_path_low = os.path.join(self.file_path_low, self.image_list_low[index])

        # train_low_data_names = glob('D:/Newcode/R2RNet-main/data_dir' + '/Huawei/low/*.jpg') + \
        #                        glob('D:/Newcode/R2RNet-main/data_dir'  + '/Nikon/low/*.jpg')
        # # train_low_data_names.sort()
        # train_high_data_names = glob('D:/Newcode/R2RNet-main/data_dir'  + '/Huawei/high/*.jpg') + \
        #                         glob('D:/Newcode/R2RNet-main/data_dir'  + '/Nikon/high/*.jpg')
        # train_high_data_names.sort()
        # 都图片并转为Tensor
        image_low = self._read_convert_image(image_path_low)
        image_high = self._read_convert_image(image_path_high)
        # image_low = self._read_convert_image(train_low_data_names)
        # image_high = self._read_convert_image(train_high_data_names)
        # image_res = self._read_estimate_image(image_path_low)
        return (image_low,image_high)

    def _read_convert_image(self, image_name):


        image = Image.open(image_name)
        tf_resize = transforms.Compose([transforms.Resize((128,128))])
        image = tf_resize(image)
        image2 = T.ToTensor()(image)
        del image
        return image2



    def __len__(self):
        return len(self.image_list_low)
