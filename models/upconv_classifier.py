import torch
import lightning as L
import torchvision

from torch.nn import functional as F
from torch import nn
from models.deeplabv3 import DeepLabV3Backbone, DeepLabV3PredictionHead
from torchmetrics import JaccardIndex, F1Score
from collections import OrderedDict



# --- Utilities ---------------------------------------------------------

@torch.no_grad()
def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False

# --- Model Parts ---------------------------------------------------------

class PredictionHead(nn.Module):
    
    """Predction Head for downstream task.
    Upsamples the features from the backbone in a series of ConvTranspose2d layers
    They add up to 16x the original size of the input.
    Projects the number of in_channels to the number of classes.
    """
    def __init__(self, num_classes=6, in_channels=2048):
        super().__init__()
        self.upsample1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(512, num_classes, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        return x

# --- Class implementation ----------------------------------------------------------

class SegmentationModel(L.LightningModule):
        
        """Segmentation Model for downstream task.
        Combines the backbone and the prediction head.
        """
        
        def __init__(self, num_classes=6, 
                    backbone=None,
                    head=None,
                    loss_fn=None,
                    learning_rate=0.001, 
                    freeze_backbone=False,
                    ):
            
            super().__init__()
            self.backbone = backbone if backbone else DeepLabV3Backbone()
            self.prediction_head = head if head else DeepLabV3PredictionHead(num_classes=num_classes)
            
            self.loss_fn = loss_fn if loss_fn else torch.nn.CrossEntropyLoss()
            self.lr = learning_rate
            
            self.IoU = JaccardIndex(num_classes=num_classes, task='multiclass')
            self.F1 = F1Score(num_classes=num_classes, task='multiclass')
            
            self.freeze_backbone = freeze_backbone
            if self.freeze_backbone:
                deactivate_requires_grad(self.backbone)
            
        def forward(self, x):
            input_shape = x.shape[-2:]  # Save the original input shape
            features = self.backbone(x)
            if isinstance(features, OrderedDict):
                features = features['out']
            x = self.prediction_head(features)
            return F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            
        def training_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self.forward(X)
            
            # Compute the loss
            y = y.squeeze(1).to(torch.long) 
            loss = self.loss_fn(y_hat, y)
            IoU = self.IoU(y_hat, y)
            F1 = self.F1(y_hat, y)
            
            self.log("train_loss", loss)
            self.log("train_IoU", IoU)
            self.log("tain_F1", F1)
            return loss
        
        def validation_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self.forward(X)

            y = y.squeeze(1).to(torch.long)
            val_loss = self.loss_fn(y_hat, y)
            val_IoU = self.IoU(y_hat, y)
            val_F1 = self.F1(y_hat.argmax(dim=1), y)

            self.log("val_loss", val_loss)
            self.log("val_IoU", val_IoU)
            self.log("val_F1", val_F1)
            return val_loss
        
        def test_step(self, batch, batch_idx):
            X, y = batch
            y_hat = self.forward(X)

            y = y.squeeze(1).to(torch.long)
            test_loss = self.loss_fn(y_hat, y)
            test_IoU = self.IoU(y_hat, y)
            test_F1 = self.F1(y_hat.argmax(dim=1), y)

            self.log("test_loss", test_loss)
            self.log("test_IoU", test_IoU)
            self.log("test_F1", test_F1)
            return test_loss
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=0.0005)
            return optimizer