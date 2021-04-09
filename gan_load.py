import argparse
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image

from model import CartoonGAN

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="test_img")
parser.add_argument("--load_size", type=int, default=450)
parser.add_argument("--model_path", type=str, default="./pretrained_model")
parser.add_argument("--style", type=str, default="Hayao")
parser.add_argument("--output_dir", type=str, default="test_output")
opt = parser.parse_args()


os.makedirs(opt.output_dir, exist_ok=True)


model = CartoonGAN()
model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.style + "_net_G_float.pth")))
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
else:
    model = model.float()

image_path = "my_images/lion.jpg"

# load image
input_image = Image.open(image_path).convert("RGB")

h = input_image.size[0]
w = input_image.size[1]

ratio = h * 1.0 / w

if ratio > 1:
    h = opt.load_size
    w = int(h * 1.0 / ratio)
else:
    w = opt.load_size
    h = int(w * ratio)


input_image = input_image.resize((h, w), Image.BICUBIC)
input_image = np.asarray(input_image)

# RGB -> BGR
input_image = input_image[:, :, [2, 1, 0]]
input_image = transforms.ToTensor()(input_image).unsqueeze(0)
input_image = -1 + 2 * input_image

if torch.cuda.is_available():
    input_image = Variable(input_image, volatile=True).cuda()
else:
    input_image = Variable(input_image, volatile=True).float()

# forward
output_image = model(input_image)
output_image = output_image[0]

# BGR -> RGB
output_image = output_image[[2, 1, 0], :, :]
output_image = output_image.data.cpu().float() * 0.5 + 0.5

save_image(output_image, os.path.join(opt.output_dir, image_path.split("/")[-1][:-4] + "_" + opt.style + ".jpg"))

