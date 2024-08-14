import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence
from torchvision.models.resnet import resnet50
import lightning as L
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchmetrics import JaccardIndex, F1Score
from collections import OrderedDict
import torchvision



@torch.no_grad()
def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False


class DeepLabV3Model(L.LightningModule):
    def __init__(self, backbone=None, pred_head=None, num_classes=6, learning_rate=0.001, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone if backbone else DeepLabV3Backbone()
        self.pred_head = pred_head if pred_head else DeepLabV3PredictionHead(num_classes=num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        if freeze_backbone:
            deactivate_requires_grad(self.backbone)
            
        self.IoU = JaccardIndex(num_classes=num_classes, task='multiclass')
        self.F1 = F1Score(num_classes=num_classes, task='multiclass')

    # def forward(self, x):
    #     input_shape = x.shape[-2:]
    #     h = self.backbone(x)
    #     z = self.pred_head(h)
    #     # Upscaling
    #     return F.interpolate(z, size=input_shape, mode="bilinear", align_corners=False)

    def forward(self, x):
        input_shape = x.shape[-2:]  # Save the original input shape
        features = self.backbone(x)
        if isinstance(features, OrderedDict):
            features = features['out']
        x = self.pred_head(features)
        return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X.float())
        loss = self.loss_fn(logits, y.squeeze(1).to(torch.long))
        
        pred = torch.argmax(logits, dim=1, keepdim=True)

        f1 = self.F1(pred, y)
        IoU = self.IoU(pred, y)
        
        self.log("train_loss", loss)
        self.log("train_F1", f1)
        self.log("train_IoU", IoU)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.forward(X.float())
        val_loss = self.loss_fn(logits, y.squeeze(1).to(torch.long))
        
        pred = torch.argmax(logits, dim=1, keepdim=True)

        val_f1 = self.F1(pred, y)
        val_IoU = self.IoU(pred, y)
        
        self.log("val_loss", val_loss)
        self.log("val_F1", val_f1)
        self.log("val_IoU", val_IoU)
        return val_loss   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        return optimizer
    
class DeepLabV3Backbone(nn.Module):
    def __init__(self, num_classes=6, pretrain=''):
        super().__init__()
        if pretrain == '':
            print('********** Backbone from scratch carregado **********')
            self.RN50model = resnet50(replace_stride_with_dilation=[False, True, True])
        elif pretrain == 'imagenet':
            print('********** Backbone IMAGENET carregado **********')
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1            
            self.RN50model = resnet50(replace_stride_with_dilation=[False, True, True],
                                      weights=weights)
    
    def freeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = True

    def forward(self, x):
            x = self.RN50model.conv1(x)
            x = self.RN50model.bn1(x)
            x = self.RN50model.relu(x)
            x = self.RN50model.maxpool(x)
            x = self.RN50model.layer1(x)
            x = self.RN50model.layer2(x)
            x = self.RN50model.layer3(x)
            x = self.RN50model.layer4(x)
            #x = self.RN50model.avgpool(x)      # These should be removed for deeplabv3
            #x = torch.RN50model.flatten(x, 1)  # These should be removed for deeplabv3
            #x = self.RN50model.fc(x)           # These should be removed for deeplabv3
            return x
    
class DeepLabV3PredictionHead(nn.Sequential):
    def __init__(self, 
                    in_channels: int = 2048, 
                    num_classes: int = 6, 
                    atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )