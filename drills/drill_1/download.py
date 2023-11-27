import torchvision
from torchvision import transforms

trans = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('drills/drill_1/dataset', train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST('drills/drill_1/dataset', train=False, transform=trans, download=True)
