from Network import MyNetwork
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 创建一个转换对象，它会将PIL图像转换为张量，并重塑它
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为张量
    transforms.Lambda(lambda x: x.view(1, 28, 28)) # 将张量重塑为[1, 28, 28]
])

def convert_image(image_path):
    img = Image.open(image_path)
    img = img.resize((28, 28))
    img = img.convert('L')
    return img

# 显示图像和预测结果
def visualize_prediction(img_path, sorted_classes):
    img = Image.open(img_path)
    
    plt.figure(figsize=(10, 10))  # 创建一个新的图像，并设置图像的大小
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted class: {sorted_classes[0]}', fontsize=42)  # 添加一个标题，并设置字体大小
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

model = MyNetwork(input_dim=1*28*28, hidden_dim=128, output_dim=10)
model = model.to(device)  # 将模型移动到设备上

model.load_state_dict(torch.load('drills/model.pth'))

def predict_and_visualize(img_path):
    # 使用转换对象处理图像
    img = transform(convert_image(img_path)).unsqueeze(0).to(device)  # 将数据移动到设备上

    pred = model(img)

    # 对模型的输出进行排序
    _, indices = torch.sort(pred, descending=True)

    # 获取排序后的类别
    sorted_classes = indices.squeeze().tolist()

    print(f'排序后的类别: {sorted_classes}')

    visualize_prediction(img_path, sorted_classes)

    # 返回得分最高的类别
    return sorted_classes[0]
