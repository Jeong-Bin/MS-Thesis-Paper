import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out

class CNN4Backbone(nn.Module):
    def __init__(self, in_channels, hidden_dim, padding):
        super(CNN4Backbone, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, hidden_dim, padding)
        self.conv_block2 = ConvBlock(hidden_dim, hidden_dim, padding)
        self.conv_block3 = ConvBlock(hidden_dim, hidden_dim, padding)
        self.conv_block4 = ConvBlock(hidden_dim, hidden_dim, padding)

    def forward(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        return out1, out2, out3, out4

class CNN4(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, padding=1, spatial_size=5, num_classes=5):
        super(CNN4, self).__init__()
        self.features = CNN4Backbone(in_channels, hidden_dim, padding)
        self.fc = nn.Linear(hidden_dim*spatial_size*spatial_size, num_classes)

    def forward(self, x):
        _, _, _, out4 = self.features(x)
        output = out4.view(out4.size(0), -1)
        output = self.fc(output)
        return output
    
    def forward_features(self, x):
        _, _, _, out4 = self.features(x)
        return out4

    def all_features(self, x):
        out1, out2, out3, out4 = self.features(x)
        return [out1, out2, out3, out4]
    
    def feature_mix(self, x, teacher_feature=None, alpha=0.5):
        _, _, _, out4 = self.features(x)
        if teacher_feature != None :
            out4 = (1-alpha)*out4 + alpha*teacher_feature
        output = out4.view(out4.size(0), -1)
        output = self.fc(output)
        return output
    
# -------------------------------------------------------------------------------------------

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class CNN2Backbone(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN2Backbone, self).__init__()
        self.conv_block1 = ConvBlock2(in_channel, out_channel)
        self.conv_block2 = ConvBlock2(out_channel, out_channel)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x

class CNN2(nn.Module):
    def __init__(self, channel_size, num_classes):
        super(CNN2, self).__init__()
        # if channel_size != 1280 :
        #     nn.Linear(channel_size, 1280)
        self.backbone = CNN2Backbone(channel_size, channel_size)
        self.fc = nn.Linear(channel_size*1*1, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
    
    
# -------------------------------------------------------------------------------------------  

class CNN1(nn.Module):
    def __init__(self, channel_size, kernel_size, num_classes):
        super(CNN1, self).__init__()
        self.conv = nn.Conv2d(channel_size, channel_size, kernel_size, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(channel_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channel_size*1*1, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output




