import torch
from model.Net import CoarseNet,FineNet
from torchvision import transforms
from PIL import Image
import conf
# cuda = True if torch.cuda.is_available() else False
import matplotlib.pyplot as plt
import numpy as np
def Test_demo():
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # .to(conf.device)
    coarse_net = CoarseNet()
    coarse_net.load_state_dict(torch.load(conf.coarse_model_path,map_location='cpu'))
    coarse_net.eval()
    # .to(conf.device)
    fine_net = FineNet()
    fine_net.load_state_dict(torch.load(conf.fine_model_path,map_location='cpu'))
    fine_net.eval()
    img = Image.open(conf.test_path)
    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize([256,256])]
    # )
    img_tensor = transforms.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)
    coarse, _, _ = coarse_net(img_tensor)
    res = fine_net(img_tensor,coarse.detach())
    res = transforms.ToPILImage()(res.cpu().squeeze(0))
    res.save(conf.result_path)
    print(np.asarray(res))
    plt.imshow(res)
    plt.show()


if __name__ == '__main__':
    Test_demo()