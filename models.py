import torch
from torchvision.models import googlenet, vgg16, vgg16_bn, vgg19, vgg19_bn, resnet101, mobilenet_v2, squeezenet1_0, densenet201, densenet121
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18
from torch.nn.functional import interpolate
from torch.nn import Sequential, Conv3d, InstanceNorm3d, BatchNorm3d, ReLU, MaxPool3d, AvgPool3d, Linear, AdaptiveAvgPool3d

class VGG2D(torch.nn.Module):
    def __init__(self):
        super(VGG2D, self).__init__()

        self.convs = torch.nn.Sequential(torch.nn.Conv2d(1, 64, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2),
                                         torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2),
                                         torch.nn.Conv2d(128, 256, 5, padding=0), torch.nn.ReLU(),
                                         torch.nn.Conv2d(256, 512, 1, padding=0), torch.nn.ReLU(),
                                         torch.nn.Conv2d(512, 5, 1, padding=0)
                                         )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), 5)
        return x


class VGG3D(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(VGG3D, self).__init__()

        self.conv1 = Sequential(Conv3d(in_channels, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                InstanceNorm3d(64, True), ReLU(True),
                                Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                InstanceNorm3d(64, True), ReLU(True),
                                Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                InstanceNorm3d(64, True), ReLU(True),
                                MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        self.conv2 = Sequential(Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                InstanceNorm3d(128, True), ReLU(True),
                                Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                InstanceNorm3d(128, True), ReLU(True),
                                Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                InstanceNorm3d(128, True), ReLU(True),
                                MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.conv3 = Sequential(Conv3d(128, 256, kernel_size=3, padding=1),
                                InstanceNorm3d(256, True), ReLU(True),
                                Conv3d(256, 256, kernel_size=3, padding=1),
                                InstanceNorm3d(256, True), ReLU(True),
                                Conv3d(256, 256, kernel_size=3, padding=1),
                                InstanceNorm3d(256, True), ReLU(True),
                                MaxPool3d(2, stride=2))


        self.avgpool = AdaptiveAvgPool3d(1, 1, 1)
        self.fc = Linear(256, out_channels)

    def forward(self, x, seg=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class VGG_mod(torch.nn.Module):
    def __init__(self, num_channels=3):
        super(VGG_mod, self).__init__()

        self.fusion = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.vgg = vgg19_bn(pretrained=True)
        self.vgg.classifier[6] = torch.nn.Linear(4096, num_channels)

    def forward(self, x, get_feat=False):
        x = self.fusion(x)
        if not get_feat:
            x = self.vgg(x)
        else:
            feat = self.vgg.features(x)
            x = self.vgg.avgpool(feat)
            x = torch.flatten(x, 1)
            x = self.vgg.classifier(x)
            return x, feat
        return x


class resnet3d(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(resnet3d, self).__init__()
        self.r3d = r3d_18(pretrained=True)
        self.r3d.fc = Linear(512, num_classes)

    def forward(self, x, seg=None):
        x = self.r3d(x)
        return x


class resnet_mod(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(resnet_mod, self).__init__()

        self.fusion = torch.nn.Conv2d(in_channels, 3, 1, bias=False)
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = torch.nn.Linear(2048, num_classes)

    def forward(self, x, seg=None, get_feat=False):
        x = self.fusion(x)

        if not get_feat:
            if seg is None:
                x = self.resnet(x)
            else:
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)

                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)

                # seg = interpolate(seg, size=7, mode='bilinear', align_corners=False)
                # x = x * seg
                x = self.resnet.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.resnet.fc(x)

        else:
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            feat = self.resnet.layer4(x)

            seg = interpolate(seg, size=7, mode='bilinear', align_corners=False)
            feat = feat * seg

            x = self.resnet.avgpool(feat)
            x = torch.flatten(x, 1)
            x = self.resnet.fc(x)
            return x, feat

        return x


class denseNet_mod(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(denseNet_mod, self).__init__()

        self.fusion = torch.nn.Conv2d(in_channels, 3, 1, bias=False)
        self.densenet = densenet201(pretrained=True)
        self.densenet.classifier = torch.nn.Linear(1920, num_classes)

    def forward(self, x, seg=None, get_feat=False):
        x = self.fusion(x)

        if not get_feat:
            if seg is None:
                x = self.densenet(x)
            else:
                x = self.resnet.conv1(x)
                x = self.resnet.bn1(x)
                x = self.resnet.relu(x)
                x = self.resnet.maxpool(x)

                x = self.resnet.layer1(x)
                x = self.resnet.layer2(x)
                x = self.resnet.layer3(x)
                x = self.resnet.layer4(x)

                # seg = interpolate(seg, size=7, mode='bilinear', align_corners=False)
                # x = x * seg
                x = self.resnet.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.resnet.fc(x)

        else:
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            feat = self.resnet.layer4(x)

            seg = interpolate(seg, size=7, mode='bilinear', align_corners=False)
            feat = feat * seg

            x = self.resnet.avgpool(feat)
            x = torch.flatten(x, 1)
            x = self.resnet.fc(x)
            return x, feat

        return x


class GoogleNet_mod(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(GoogleNet_mod, self).__init__()

        self.fusion = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.googlenet = googlenet(pretrained=True, transform_input=False)
        '''
        for param in self.googlenet.parameters():
            param.requires_grad = False
        '''
        self.googlenet.fc = torch.nn.Linear(1024, num_classes)

    def forward(self, x, get_feat=False):
        x = self.fusion(x)
        if not get_feat:
            x = self.googlenet(x)
        else:
            # N x 3 x 224 x 224
            x = self.googlenet.conv1(x)
            # N x 64 x 112 x 112
            x = self.googlenet.maxpool1(x)
            # N x 64 x 56 x 56
            x = self.googlenet.conv2(x)
            # N x 64 x 56 x 56
            x = self.googlenet.conv3(x)
            # N x 192 x 56 x 56
            x = self.googlenet.maxpool2(x)

            # N x 192 x 28 x 28
            x = self.googlenet.inception3a(x)
            # N x 256 x 28 x 28
            x = self.googlenet.inception3b(x)
            # N x 480 x 28 x 28
            x = self.googlenet.maxpool3(x)
            # N x 480 x 14 x 14
            x = self.googlenet.inception4a(x)
            # N x 512 x 14 x 14
            if self.googlenet.training and self.googlenet.aux_logits:
                aux1 = self.googlenet.aux1(x)

            x = self.googlenet.inception4b(x)
            # N x 512 x 14 x 14
            x = self.googlenet.inception4c(x)
            # N x 512 x 14 x 14
            x = self.googlenet.inception4d(x)
            # N x 528 x 14 x 14
            if self.googlenet.training and self.googlenet.aux_logits:
                aux2 = self.googlenet.aux2(x)

            x = self.googlenet.inception4e(x)
            # N x 832 x 14 x 14
            x = self.googlenet.maxpool4(x)
            # N x 832 x 7 x 7
            x = self.googlenet.inception5a(x)
            # N x 832 x 7 x 7
            feat = self.googlenet.inception5b(x)
            # N x 1024 x 7 x 7

            x = self.googlenet.avgpool(feat)
            # N x 1024 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 1024
            x = self.googlenet.dropout(x)
            x = self.googlenet.fc(x)
            # N x 1000 (num_classes)
            return x, feat

        return x

class mobilenet_mod(torch.nn.Module):
    def __init__(self, num_classes=3):
        super(mobilenet_mod, self).__init__()

        self.fusion = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.mobilenet = mobilenet_v2(pretrained=True)
        '''
        for param in self.googlenet.parameters():
            param.requires_grad = False
        '''
        self.mobilenet.classifier[1] = torch.nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.fusion(x)
        x = self.mobilenet(x)
        return x