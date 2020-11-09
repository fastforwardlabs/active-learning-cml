import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#LayerFunc = Callable[[nn.Module],None]

resnet18 = models.resnet18(pretrained=True)


def get_net(name):
    if name == 'MNIST':
        return Net1
    elif name == 'CIFAR10':
        return Net3
    elif name == 'CALTECH':
        # return Resnet18
        # return Net4
        # return resnet18_transfer
        return resnet18_extractor
    elif name == 'QUICKDRAW':
        return Net5


class resnet18_transfer(nn.Module):
    def __init__(self, n_classes=10):
        super(resnet18_transfer, self).__init__()
        image_modules = list(resnet18.children())[:-1]
        self.model = nn.Sequential(*image_modules)
        num_ftrs = resnet18.fc.in_features
        self.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        e1 = x
        x = self.fc(x)
        return x, e1

    def get_embedding_dim(self):
        return 512

class resnet18_extractor(nn.Module):
    def __init__(self, n_classes=10):
        super(resnet18_extractor, self).__init__()
        image_modules = list(resnet18.children())[:-1]
        self.model = nn.Sequential(*image_modules)
        for param in self.model.parameters():
            param.requires_grad = False
        # newly constructed modules have requires_grad=True by default
        num_ftrs = resnet18.fc.in_features
        self.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        e1 = x
        # Adam, p=0.7 works with lr=0.0001
        # Adam, p=0.5, betas=(0.9,0.99), lr=0.00005, 20 or 30 epochs
        # Adam, p=0.2, betas=(0.9,0.99), lr=0.00005, 20 or 30 epochs
        #x = F.dropout(x, p=0.2, training=self.training)
        # SGD p=0.5 works better
        # SGD p=0.2 lr=0.01, bs=25, 
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, e1

    def get_embedding_dim(self):
        return 512


class Net1(nn.Module):
    def __init__(self, n_classes=10):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class Net3(nn.Module):
    def __init__(self, n_classes=10):
        super(Net3, self).__init__()
        # cifar is 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # 4x4x64 = 1024
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class Net4(nn.Module):
    # fix for caltech input data
    def __init__(self, n_classes):
        super(Net4, self).__init__()
        # cifar is 32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=10, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=10, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=10, stride=2)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2,))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # 4x4x64 = 1024
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class Net5(nn.Module):
    def __init__(self, n_classes=10):
        super(Net5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Resnet18(nn.Module):
    # hard code resnet params for block and layers
    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # input size = 224x224
    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], n_classes=10):
        super(Resnet18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # move self.fc 
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # added this param
        self.out_size = 512 * block.expansion

        if n_classes > 0:  # add fully connected layer
            self.use_classification = True
            self.fc = nn.Linear(512 * block.expansion, n_classes)
            self.out_size = n_classes
        else:
            self.use_classification = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x is embedding
        e1 = x
        if self.use_classification:
            x = self.fc(x)
        return x, e1

    def get_embedding_dim(self):
        return 512
