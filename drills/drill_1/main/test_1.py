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
    plt.imshow(img, cmap='gray')
    plt.title(f'Predicted classes: {sorted_classes}')
    plt.show()

img_path_1 = 'drills/drill_1/main/my_image/61.png'
img_path_2 = 'drills/drill_1/main/my_image/62.png'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

model = MyNetwork(input_dim=1*28*28, hidden_dim=128, output_dim=10)
model = model.to(device)  # 将模型移动到设备上

model.load_state_dict(torch.load('drills/model.pth'))

# 使用转换对象处理图像
img_1 = transform(convert_image(img_path_1)).unsqueeze(0).to(device)  # 将数据移动到设备上
img_2 = transform(convert_image(img_path_2)).unsqueeze(0).to(device)  # 将数据移动到设备上

pred_1 = model(img_1)
pred_2 = model(img_2)

# 对模型的输出进行排序
_, indices_1 = torch.sort(pred_1, descending=True)
_, indices_2 = torch.sort(pred_2, descending=True)

# 获取排序后的类别
sorted_classes_1 = indices_1.squeeze().tolist()
sorted_classes_2 = indices_2.squeeze().tolist()

print(f'排序后的类别（歪溜）: {sorted_classes_1}')
print(f'排序后的类别（正溜）: {sorted_classes_2}')

visualize_prediction(img_path_1, sorted_classes_1)
visualize_prediction(img_path_2, sorted_classes_2)