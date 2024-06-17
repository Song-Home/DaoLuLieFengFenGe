import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
# from src import UNet
from src import VGG16UNet

def time_synchronized():                                                           # 这个函数保证在异步CUDA操作的情况下，以确保得到的时间测量是准确和可靠的
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def adjust_probabilities_and_fill(tensor):
    # Step 1: 调整概率
    adjusted_tensor = tensor.clone()
    adjusted_tensor[:, 1, :, :] = (adjusted_tensor[:, 1, :, :] >= 0.5).float()

    # Step 2: 处理每个图像
    for idx in range(tensor.size(0)):
        image = adjusted_tensor[idx, 1, :, :].cpu().numpy()

        # Step 3: 找到连通域和凸包
        contours, _ = cv2.findContours((image * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hulls = []
        for contour in contours:
            # 检查连通域面积
            if cv2.contourArea(contour) < 100:
                cv2.drawContours(image, [contour], -1, 0, -1)  # 填充面积小于100的连通域为0
            else:
                hulls.append(cv2.convexHull(contour))

        # Step 4: 连接近距离的凸包
        for i in range(len(hulls)):
            for j in range(i + 1, len(hulls)):
                for point1 in hulls[i]:
                    for point2 in hulls[j]:
                        if np.linalg.norm(point1 - point2) < 20:
                            cv2.line(image, tuple(point1[0]), tuple(point2[0]), 1, thickness=1)

        # Step 5: 对所有连通域进行去孔操作
        kernel = np.ones((9,9), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        # 回填图像到张量
        adjusted_tensor[idx, 1, :, :] = torch.from_numpy(image).to(tensor.device)

    # Step 6: 设置前景和背景
    adjusted_tensor[:, 0, :, :] = 1 - adjusted_tensor[:, 1, :, :]  # 背景为相反值

    return adjusted_tensor


def main():
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.pth"
    img_path = "./DRIVE/predict/images/20160328_153813_641_721.tif"
    roi_mask_path = "./DRIVE/predict/mask/20160328_153813_641_721_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."      # 下面3行代码用于检查weights_path、img_path、roi_mask_path指定的文件是否存在，如果不存在，将输出一条错误信息。
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=classes+1, base_c=32)
    model = VGG16UNet(num_classes=classes+1, pretrain_backbone=True)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])   # 这行代码加载之前保存的模型权重。
    model.to(device)                                                               # 将模型移动指定的设备（GPU或CPu）

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])  # 定义一个变换序列data_transform，将图像转换为张量，并对其进行标准化处理，使用给定的均值mean和标准差std
    img = data_transform(original_img)                                               # 将之前加载的原始图像original_img应用上述变换
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)                                                # 通过增加一个批次维度，将图像张量转换为适合模型输入的格式。

    model.eval()  # 进入验证模式，这在进行推断时很重要，因为它会禁用某些特定于训练阶段的行为
    with torch.no_grad(): # 在这个块内部，禁用梯度计算，这有助于减少内存使用并加速推断。
        # init model
        img_height, img_width = img.shape[-2:]                                       # 初始化模型并进行一次前向传递，以确保模型处于正确状态
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()                                                # 获取当前时间，作为推断开始的时间点
        output = model(img.to(device))                                               # 将处理后的图像输入模型进行推断，并将图像移动到指定的设备上
        t_end = time_synchronized()                                                  # 获取推断结束的时间点
        print("inference time: {}".format(t_end - t_start))

        adjusted_tensor = output['out'].clone()
        adjusted_tensor = adjust_probabilities_and_fill(output['out'])
        prediction = adjusted_tensor.argmax(1).squeeze(0)                              
        prediction = prediction.to("cpu").numpy().astype(np.uint8)                   # 提取模型的输出，并将其转换为Numpy数组，准备进行后续处理
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255                                            # 对预测结果进行处理，将前景像素值设置为255（白色），不感兴趣的区域设置为0（黑色）
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0                                                 
        mask = Image.fromarray(prediction)                                           # 将处理后的预测结果转换为图像并保存
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
