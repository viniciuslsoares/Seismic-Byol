import sys
sys.path.append('..')

from torch import nn
import torch
import lightning as L
import pytorch_lightning as pl

from pathlib import Path
import models.deeplabv3 as dlv3
import models.byol as byol_module
from transforms.byol import BYOLTransform
from data_modules.Parihaka_dataset import ParihakaDataModule as ByolDataModule
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
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning DataModule.

def build_pretext_datamodule(batch, input_size) -> L.LightningDataModule:
    # Build the transform object
    transform = BYOLTransform(input_size=input_size,
                            min_scale=0,
                            degrees=5,
                            r_prob=0.0,
                            h_prob=0.0,
                            v_prob=0.0,
                            collor_jitter_prob=0,
                            grayscale_prob=0,
                            gaussian_blur_prob=0,
                            solarize_prob=0.0
                            )
    # Create the datamodule
    # print("Number of files in the pretext dataset: ", num_files("../data/pretext/images/pretrain/"))
    return ByolDataModule(root_dir="../data/f3/images/",
                                batch_size=batch,
                                transform=transform)

### --------------- LightningModule --------------------------------------------------

# This function must instantiate and configure the pretext model
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning model.

def build_pretext_model() -> L.LightningModule:
    # Build the backbone
    backbone = dlv3.DeepLabV3Backbone()
    # Loss function and projection head already inside LightningModule
    # Build the pretext model
    return byol_module.BYOLModel(backbone=backbone,
                                learning_rate=0.1)
    
### --------------- Trainer -------------------------------------------------------------

# This function must instantiate and configure the lightning trainer
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure you return a Lightning trainer.

def build_lightning_trainer(save_name:str, epocas:int) -> L.Trainer:
    return L.Trainer(
        accelerator="gpu",
        max_epochs=epocas,
        # max_steps=10500,
        enable_checkpointing=False, 
        logger=CSVLogger("logs", name="Byol", version=save_name),
        strategy='ddp_find_unused_parameters_true'
        )
    
### --------------- Main -----------------------------------------------------------------

def main(SSL_technique_prefix):
    
    # numero de imagens: aprox 2780
    
    EPOCAS = 300
    BATCH_SIZE = 32
    INPUT_SIZE = 256
    
    save_name = f'E{EPOCAS}_B{BATCH_SIZE}_S{INPUT_SIZE}_f3'

    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_model = build_pretext_model()
    pretext_datamodule = build_pretext_datamodule(BATCH_SIZE, INPUT_SIZE)
    lightning_trainer = build_lightning_trainer(save_name, EPOCAS)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)

    # Save the backbone weights
    output_filename = f"../saves/backbones/{SSL_technique_prefix}_{save_name}.pth"
    pretext_save_backbone_weights(pretext_model, output_filename)

if __name__ == "__main__":
    main('Byol')
