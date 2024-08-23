import sys
sys.path.append('..')

from torch import nn
import torch
import lightning as L
import pytorch_lightning as pl
import torchvision

from pathlib import Path
import models.deeplabv3 as dlv3
import models.byol as byol_module
import torchvision.models as models
from transforms.byol import BYOLTransform
from data_modules.Pretrain_dataset import PretrainDataModule as ByolDataModule
from pytorch_lightning.loggers import CSVLogger

# This function must save the weights of the pretrained model
def pretext_save_backbone_weights(pretext_model, checkpoint_filename):
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.backbone.state_dict(), checkpoint_filename)

def num_files(path):
    import os
    list = os.listdir(path)
    return len(list)

### ---------- DataModule -----------------------------------------------------------

# This function must instantiate and configure the datamodule for the pretext task

def build_pretext_datamodule(batch, input_size, data:str = 'both') -> L.LightningDataModule:
    # Build the transform object
    transform = BYOLTransform(input_size=input_size,
                            min_scale=0,
                            degrees=5,
                            r_prob=0.0,
                            h_prob=0.5,
                            v_prob=0.0,
                            collor_jitter_prob=0,
                            grayscale_prob=0,
                            gaussian_blur_prob=0,
                            solarize_prob=0.0
                            )
    
    assert data in ['both', 'seam_ai', 'f3'], f"Data {data} not found. Must be one of 'both', 'seam_ai' or 'f3'"
    
    num_of_files = num_files(f"../data/{data}/images/train/")
    
    # Create the datamodule
    print("Number of files in the pretext dataset: ", num_of_files)
    
    path = f'../data/{data}/images/'
    # The selection of path/train is inside of the datamodule
    
    print(f'******* Data Loaded: {data} *******')
    print(f'******* Path: {path} *******')
    
    return ByolDataModule(root_dir=path,
                                batch_size=batch,
                                transform=transform), num_of_files

### --------------- LightningModule --------------------------------------------------

# This function must instantiate and configure the pretext model

def build_pretext_model(schedule:int=9000 ) -> L.LightningModule:
    # Build the backbone
    
    backbone = models.segmentation.deeplabv3_resnet50().backbone
        
    return byol_module.BYOLModel(backbone=backbone,
                                learning_rate=0.1,
                                schedule=schedule,
                                )
    
### --------------- Trainer -------------------------------------------------------------

# This function must instantiate and configure the lightning trainer

def build_lightning_trainer(save_name:str, epocas:int) -> L.Trainer:
    return L.Trainer(
        accelerator="gpu",
        max_epochs=epocas,
        enable_checkpointing=False, 
        logger=CSVLogger("logs", name="Byol", version=save_name),
        # strategy='ddp_find_unused_parameters_true',
        devices=[0],
        )
    
### --------------- Main -----------------------------------------------------------------

def pretrain_func(epocas:int = 300,
                 batch_size:int = 32,
                 input_size:int = 256,
                 repetition:str = 'V1',
                 save_name:str = 'byol',
                 data:str = 'both'
                 ):
        
    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_datamodule, num_of_files = build_pretext_datamodule(batch_size, input_size, data)
    
    schedule = int((num_of_files // batch_size) * epocas) 
    # Used to determine the cossine schedule in the pretext model
    # Numero de batches por epoca: num_of_files // batch_size pela quantidade de épocas
    
    pretext_model = build_pretext_model(schedule=schedule)
    lightning_trainer = build_lightning_trainer(save_name, epocas)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)

    # Save the backbone weights
    output_filename = f"../saves/backbones/{repetition}/{save_name}.pth"
    pretext_save_backbone_weights(pretext_model, output_filename)




# if __name__ == "__main__":
#     pretrain_func('Byol')
