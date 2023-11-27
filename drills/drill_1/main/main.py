from Network import MyNetwork
import torch
import datasets
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

if __name__ == '__main__':
    
    writer = SummaryWriter('logs') # 创建tensorboard对象
    
    train_loader = datasets.train_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备

    net = MyNetwork(input_dim=1*28*28, hidden_dim=128, output_dim=10)
    net = net.to(device)  # 将模型移动到设备上
    
    # 检查模型是否在GPU上
    if next(net.parameters()).is_cuda:
        print("The model is running on GPU.")
    else:
        print("The model is running on CPU.")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    loss_func = nn.CrossEntropyLoss()

    total_loss = []  # 记录损失
    for epoch in range(10000):  # 增加最大epoch数以防止模型过早停止
        epoch_loss = 0  # 回合损失
        for i, data in enumerate(train_loader, 0):
            
            image, label = data
            image = image.to(device)  # 将数据移动到设备上
            label = label.to(device)  # 将标签移动到设备上
            optimizer.zero_grad()
            pred = net(image)
            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() # 累计回合损失
            # 将图像、预测值和真实值添加到tensorboard
            writer.add_image('images', torchvision.utils.make_grid(image.cpu()), epoch * len(train_loader) + i)
            writer.add_scalar('Predicted value', pred.argmax(dim=1)[0].item(), epoch * len(train_loader) + i)
            writer.add_scalar('Real value', label[0].item(), epoch * len(train_loader) + i)
            
        epoch_loss /= len(train_loader)
        total_loss.append(epoch_loss) # 记录回合损失
        print(f'epoch : {epoch+1}, training loss: {epoch_loss}') # 打印信息

        # 如果损失小于0.1，停止训练
        if epoch_loss < 0.1:
            break

    torch.save(net.state_dict(), 'model.pth') # 保存模型参数