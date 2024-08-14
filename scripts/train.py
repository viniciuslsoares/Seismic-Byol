import sys
sys.path.append('../')

import torch
import lightning as L

import models.deeplabv3 as dlv3
from lightning.pytorch.callbacks import EarlyStopping
from models.upconv_classifier import SegmentationModel, PredictionHead
from data_modules.seismic import F3SeismicDataModule, ParihakaSeismicDataModule
from pytorch_lightning.loggers import CSVLogger

import torchvision.models.segmentation as models

### utilities 

def num_files(path):
    import os
    list = os.listdir(path)
    return len(list)

### ---------- PreTrain  ------------------------------------------------------------

# This function should load the backbone weights
def load_pretrained_backbone(pretrained_backbone_checkpoint_filename, mode:str='byol'):

    # backbone direto do pytorch

    backbone = models.deeplabv3_resnet50().backbone
    if mode == 'byol':
        print('***** Backbone carregado *****')
        backbone.load_state_dict(torch.load(pretrained_backbone_checkpoint_filename))    
    
    elif mode == 'supervised':
        print('***** Backbone from scratch *****')
        # Nenhum peso carregado no backbone
    
    elif mode == 'coco':
        print('***** Backbone COCO carregado *****')
        backbone = models.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1').backbone
    
    elif mode == 'imagenet':
        print('***** Backbone IMAGENET carregado *****')
        backbone = dlv3.DeepLabV3Backbone(num_classes=6, pretrain='imagenet')
    
    return backbone

### ---------- DataModule -----------------------------------------------------------

# This function must instantiate and configure the datamodule for the downstream task.

def build_downstream_datamodule(batch_size, cap, data) -> L.LightningDataModule:

    if data == 'parihaka':
        print("Parihaka datas being used")
        return ParihakaSeismicDataModule(root_dir="../data/", batch_size=batch_size, cap=cap)

    elif data == 'f3':
        print("F3 datas being used")
        return F3SeismicDataModule(root_dir="../data/", batch_size=batch_size, cap=cap)

    else:
        raise ValueError('Data not found. Choose between parihaka and f3')
    
### --------------- LightningModule --------------------------------------------------

# This function must instantiate and configure the downstream model

def build_downstream_model(backbone, freeze) -> L.LightningModule:
    
    pred_head = dlv3.DeepLabV3PredictionHead(num_classes=6)
    
    return SegmentationModel(num_classes=6,
                            backbone=backbone,
                            head=pred_head,
                            loss_fn=torch.nn.CrossEntropyLoss(),
                            learning_rate=0.001,
                            freeze_backbone=freeze)
    

### --------------- Trainer -------------------------------------------------------------

# This function must instantiate and configure the lightning trainer

def build_lightning_trainer(SSL_technique_prefix, save_name:str, supervised:bool, epocas) -> L.Trainer:
    from lightning.pytorch.callbacks import ModelCheckpoint
    
    # Configure the ModelCheckpoint object to save the best model 
    # according to validation loss
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_IoU',
        dirpath=f'../saves/models/',
        filename=f'{SSL_technique_prefix}_{save_name}',
        save_top_k=1,
        mode='max',
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_IoU',
        patience=10,
        mode='max'
    )
    
    return L.Trainer(
        accelerator="gpu",
        max_epochs=epocas,
        logger=CSVLogger("logs", name="Supervised" if supervised else "Pretrained", version=save_name),
        # callbacks=[checkpoint_callback, early_stopping_callback],
        callbacks=[checkpoint_callback],
        # strategy='ddp_find_unused_parameters_true',
        # devices=[1]
        )
    
### --------------- Main -----------------------------------------------------------------

def train_func(epocas:int, 
               batch_size:int, 
               cap:float, 
               import_name:str, 
               save_name:str,
               supervised:bool = False, 
               freeze:bool = False, 
               downstream_data:str = 'f3',
               SSL_technique_prefix:str = "Byol",
               mode:str = 'byol'
               ):
    
    EPOCAS = epocas
    BATCH_SIZE = batch_size
    CAP = cap
    SUPERVISED = supervised
    FREEZE = freeze
    
    # EPOCAS = 50
    # BATCH_SIZE = 8
    # CAP = 1
    # SUPERVISED = False 
    # FREEZE = False
    # import_name = 'E300_B32_S256_f3'
    # save_name = f'E{EPOCAS}_B{BATCH_SIZE}_{CAP*100}%_LR0.005'
    # save_name = 'pretreino_COCO_seam_ai_100%'

    # Load the pretrained backbone
    pretrained_backbone_checkpoint_filename = f"../saves/backbones/{SSL_technique_prefix}_{import_name}.pth"
    backbone = load_pretrained_backbone(pretrained_backbone_checkpoint_filename, mode=mode)

    # Build the downstream model, the downstream datamodule, and the trainer
    downstream_model = build_downstream_model(backbone, FREEZE)
    downstream_datamodule = build_downstream_datamodule(BATCH_SIZE, CAP, downstream_data)
    lightning_trainer = build_lightning_trainer(SSL_technique_prefix, save_name, SUPERVISED, EPOCAS)

    lightning_trainer.fit(downstream_model, downstream_datamodule)


# if __name__ == "__main__":
    # train_func()
