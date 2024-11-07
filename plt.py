import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体样式为 Times New Roman
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 17  # 增大字体大小，可根据需求调整数字

models = {
    "CARN": (1592, 32.13),
    "CFSRCNN": (1200, 32.06),
    "HGSRCNN": (2178, 32.13),
    "SREFBN": (683, 32.01),
    "EMASRN": (546, 32.17),
    "IRN": (524, 32.21),
    "SPAN-S": (498, 32.20),
    "RBSRNet": (1847, 32.26)
}

params = [param[0] for param in models.values()]
psnr = [param[1] for param in models.values()]

plt.figure(figsize=(10, 6))
for i, model_name in enumerate(models.keys()):
    if model_name == "RBSRNet":
        plt.scatter(params[i], psnr[i], color='red', label=model_name)
    else:
        plt.scatter(params[i], psnr[i], color='blue', label=model_name)

    # 根据模型名称单独设置标注文本
    if model_name == "CARN":
        plt.text(params[i], psnr[i] + 0.007, "CARN", fontsize=17, ha='center')
    elif model_name == "CFSRCNN":
        plt.text(params[i], psnr[i] + 0.007, "CFSRCNN", fontsize=17, ha='center')
    elif model_name == "HGSRCNN":
        plt.text(params[i], psnr[i] - 0.017, "HGSRCNN", fontsize=17, ha='center')
    elif model_name == "SREFBN":
        plt.text(params[i], psnr[i] + 0.007, "SREFBN", fontsize=17, ha='center')
    elif model_name == "EMASRN":
        plt.text(params[i], psnr[i] - 0.017, "EMASRN", fontsize=17, ha='center')
    elif model_name == "IRN":
        plt.text(params[i], psnr[i] + 0.007, "IRN", fontsize=17, ha='center')
    elif model_name == "SPAN-S":
        plt.text(params[i], psnr[i] - 0.017, "SPAN-S", fontsize=17, ha='center')
    elif model_name == "RBSRNet":
        plt.text(params[i], psnr[i] - 0.017, "RBSRNet", fontsize=17, ha='center')

plt.xticks(np.arange(0, 2600, 250))
plt.xlabel('Params (K)', fontsize=17)
plt.ylabel('PSNR (dB)', fontsize=17)

# 设置网格线为虚线
plt.grid(True, linestyle='--')

plt.savefig('network_performance.png', dpi=300, bbox_inches='tight')
plt.show()