"""
    功能说明：单独将 Lim 模块拎出来，可视化操作
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义模型
class LIM(nn.Module):
    def __init__(self, in_channels):
        super(LIM, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        suppressed_output = self.conv(x)
        output = x * torch.sigmoid(suppressed_output)
        return output, suppressed_output

# 读取图片并进行预处理
image_path = r"C:\Users\liujunjun\Desktop\img_003_SRF_2_LR.png"  # 替换成你的图片路径
image = Image.open(image_path)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image = transform(image).unsqueeze(0)

# 创建模型并进行前向传播
in_channels = 3
model = LIM(in_channels)
model.eval()

with torch.no_grad():
    output, suppressed_output = model(image)
    sigmoid_output = torch.sigmoid(suppressed_output)

# 转换为可视化的图像
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    image = transforms.ToPILImage()(tensor.cpu())
    return image

# 可视化结果
output_image = tensor_to_image(output)
suppressed_output_image = tensor_to_image(suppressed_output)
sigmoid_output_image = tensor_to_image(sigmoid_output)

# 保存结果
output_image.save('output_image.jpg')
suppressed_output_image.save('suppressed_output_image.jpg')
sigmoid_output_image.save('sigmoid_output_image.jpg')

print("Images saved successfully!")