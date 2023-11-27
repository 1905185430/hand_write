import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader

trans = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    '/main/dataset',
    train=True,
    transform=trans,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)

if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)
