import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TumorSegmentationModel(nn.Module):
    def __init__(self, 
                 resnet_type='resnet50', 
                 input_channels=3, 
                 num_classes=1, 
                 weights=True):
        super(TumorSegmentationModel, self).__init__()
        
        resnet_dict = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
            'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
            'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
            'resnet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2)
        }
        
        if resnet_type not in resnet_dict:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}. Choose from {list(resnet_dict.keys())}")
        
        model_constructor, weight_class = resnet_dict[resnet_type]
        if weights is True:
            weights = weight_class.DEFAULT
        elif weights is False:
            weights = None
        
        base_model = model_constructor(weights=weights)
        
        if input_channels != 3:
            base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.encoder0 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4
        
        encoder_channels = self._get_encoder_channels(resnet_type)
        
        self.decoder4_reduce = nn.Conv2d(encoder_channels[4], encoder_channels[3], kernel_size=1)
        
        self.decoder4 = self._make_decoder_block(encoder_channels[3], encoder_channels[2])
        self.decoder3 = self._make_decoder_block(encoder_channels[2], encoder_channels[1])
        self.decoder2 = self._make_decoder_block(encoder_channels[1], 128)
        self.decoder1 = self._make_decoder_block(128, 64)
        self.decoder0 = self._make_decoder_block(64, 32)
        
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def _get_encoder_channels(self, resnet_type):
        """
        Returns output channels for each encoder stage based on ResNet type
        """
        encoder_channels = {
            'resnet18':  {1: 64, 2: 128, 3: 256, 4: 512, 0: 64},
            'resnet34':  {1: 64, 2: 128, 3: 256, 4: 512, 0: 64},
            'resnet50':  {1: 256, 2: 512, 3: 1024, 4: 2048, 0: 64},
            'resnet101': {1: 256, 2: 512, 3: 1024, 4: 2048, 0: 64},
            'resnet152': {1: 256, 2: 512, 3: 1024, 4: 2048, 0: 64}
        }
        return encoder_channels[resnet_type]
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        x4_reduced = self.decoder4_reduce(x4)
        
        d4 = self.decoder4(x4_reduced)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        d0 = self.decoder0(d1)
        
        out = self.final_conv(d0)
        
        return out