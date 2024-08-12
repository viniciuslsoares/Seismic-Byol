import torch
from torchmetrics import JaccardIndex, F1Score
import lightning as L
import torchvision.models.segmentation as models

import torch.nn as nn
import torch.optim as optim


class DeepLabV3Module(L.LightningModule):
    def __init__(self, freeze_backbone=False, lr=0.001, num_classes=6):
        super().__init__()
        self.lr = lr
        self.model = models.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        
        self.num_classes = num_classes
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.IoU = JaccardIndex(num_classes=num_classes, task='multiclass')
        self.F1 = F1Score(num_classes=num_classes, task='multiclass')
        
    def forward(self, x):
        return self.model(x)
    
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
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer