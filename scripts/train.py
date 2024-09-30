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
        print('**********Backbone carregado **********')
        backbone.load_state_dict(torch.load(pretrained_backbone_checkpoint_filename))    
    
    elif mode == 'supervised':
        print('********** Backbone from scratch **********')
        # Nenhum peso carregado no backbone
    
    elif mode == 'coco':
        print('********** Backbone COCO carregado **********')
        backbone = models.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1').backbone
    
    elif mode == 'imagenet':
        print('********** Backbone IMAGENET carregado **********')
        backbone = dlv3.DeepLabV3Backbone(num_classes=6, pretrain='imagenet')
    
    elif mode == 'seg':
        print('********** Backbone Segmentation **********')
        pred_head = dlv3.DeepLabV3PredictionHead(num_classes=6)
        downstream_model = SegmentationModel.load_from_checkpoint(pretrained_backbone_checkpoint_filename,
                                                                num_classes=6,                                                             
                                                                backbone=backbone,
                                                                head=pred_head,
                                                                loss_fn=torch.nn.CrossEntropyLoss(),
                                                                learning_rate=0.001,
                                                                freeze_backbone=False,
                                                                map_location='cuda')
        backbone = downstream_model.backbone

    return backbone

### ---------- DataModule -----------------------------------------------------------

# This function must instantiate and configure the datamodule for the downstream task.

def build_downstream_datamodule(root_dir, batch_size, cap, data, seed) -> L.LightningDataModule:
# build_downstream_datamodule(root_dir=root_dir, batch_size=batch_size, cap=cap, data=downstream_data, seed=seed)

    
    path = f'{root_dir}/images'
    num_of_files = num_files(f'{path}/train')
    print("Number of files in dataset: ", num_of_files)

    assert data in ['seam_ai', 'f3'], f"Datamodule {data} not found. Must be one of 'seam_ai' or 'f3'"

    if data == 'seam_ai':
        print(f'******* Path: {path} *******')
        print("Parihaka datas being used")
        return ParihakaSeismicDataModule(root_dir=root_dir, batch_size=batch_size, cap=cap, seed=seed)

    elif data == 'f3':
        print(f'******* Path: {path} *******')
        print("F3 datas being used")
        return F3SeismicDataModule(root_dir=root_dir, batch_size=batch_size, cap=cap, seed=seed)

    else:
        raise ValueError(f"Unknown dataset: {data}")
    
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

def build_lightning_trainer(save_name:str, supervised:bool, epocas, reps) -> L.Trainer:
    from lightning.pytorch.callbacks import ModelCheckpoint
    
    # Configure the ModelCheckpoint object to save the best model 
    # according to validation loss
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_IoU',
        # dirpath=f'../saves/models/{reps}/',
        dirpath=f'../saves/models/V_0.01/',
        filename=f'{save_name}',
        save_top_k=1,
        mode='max',
    )
    
    # early_stopping_callback = EarlyStopping(
    #     monitor='val_IoU',
    #     patience=10,
    #     mode='max'
    # )
    
    return L.Trainer(
        accelerator="gpu",
        max_epochs=epocas,
        logger=CSVLogger("logs", name="Supervised" if supervised else "Pretrained", version=save_name),
        callbacks=[checkpoint_callback],
        # strategy='ddp_find_unused_parameters_true',
        devices=[1]
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
               mode:str = 'byol',
               repetition:str = 'Vx',
               seed:int = 42,
               root_dir:str = '../data/f3/'
               ):

    # Load the pretrained backbone
    if mode == 'seg':
        pretrained_backbone_checkpoint_filename = f"../saves/models/{repetition}/{import_name}.ckpt"
    else:
        pretrained_backbone_checkpoint_filename = f"../saves/backbones/{repetition}/{import_name}.pth"
    print(f'Loading pretrained backbone from {pretrained_backbone_checkpoint_filename}')
    backbone = load_pretrained_backbone(pretrained_backbone_checkpoint_filename, mode=mode)

    # Build the downstream model, the downstream datamodule, and the trainer
    downstream_model = build_downstream_model(backbone, freeze)
    downstream_datamodule = build_downstream_datamodule(root_dir=root_dir, batch_size=batch_size, cap=cap, data=downstream_data, seed=seed)
    lightning_trainer = build_lightning_trainer(save_name, supervised, epocas, reps=repetition)

    lightning_trainer.fit(downstream_model, downstream_datamodule)


if __name__ == "__main__":
    train_func(epocas=50,
               batch_size=8,
               cap=1.0,
               import_name='V1_E300_B32_S256_f3',
               save_name='window_1',
               supervised=True,
               freeze=False,
               downstream_data='f3',
               mode='supervised',
               repetition='Vx',
               seed=42,
               root_dir='../../shared_data/seismic_vinicius/f3_segmentation_dataset_window/'
            #    root_dir='../data/f3/'
               )
