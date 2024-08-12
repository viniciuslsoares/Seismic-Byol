import sys
sys.path.append('../')

import torch
import lightning as L

import models.deeplabv3 as dlv3
import models.deeplabv3_2 as dlv3_2
from data_modules.seismic import F3SeismicDataModule, ParihakaSeismicDataModule
from models.upconv_classifier import SegmentationModel, PredictionHead
import torchvision.models.segmentation as models


### - Extra Code --------------------------------------------------------------------
from torchmetrics import JaccardIndex
from torchmetrics import F1Score

def evaluate_model(model, dataset_dl):
    # Inicialize JaccardIndex metric
    jaccard = JaccardIndex(task="multiclass", num_classes=6)
    f1 = F1Score(num_classes=6, task="multiclass")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For each batch, compute the predictions and compare with the labels.
    for X, y in dataset_dl:
        # Move the model, data and metric to the GPU if available
        model.eval()
        model.to(device)
        X = X.to(device)
        y = y.to(device)
        jaccard.to(device)
        f1.to(device)
        with torch.no_grad():
            logits = model(X.float())
            predictions = torch.argmax(logits, dim=1, keepdim=True)
            jaccard(predictions, y)
            f1_score = f1(predictions, y)
    # Return a tuple with the number of correct predictions and the total number of predictions
    return (float(jaccard.compute().to("cpu"))), (float(f1_score.to("cpu")))

def report_IoU(model, dataset_dl, prefix=""):
    iou, f1 = evaluate_model(model, dataset_dl)
    print(prefix + " IoU = {:0.4f}".format(iou))
    print(prefix + " F1 = {:0.4f}".format(f1))

### ---------- DataModule -----------------------------------------------------------

# This function must instantiate and configure the datamodule for the downstream task.
def build_downstream_datamodule() -> L.LightningDataModule:
    return ParihakaSeismicDataModule(root_dir="../data/", batch_size=8, cap=1)

### ------------- Pretrained Model --------------------------------------------------

# This function must instantiate the downstream model and load its weights
# from checkpoint_filename.
# Weights saved by the *_train.py script.

def load_downstream_model(checkpoint_filename) -> L.LightningModule:
    
    # head = PredictionHead(num_classes=6, in_channels=2048)
    
    head = dlv3.DeepLabV3PredictionHead(num_classes=6)    
    
    backbone = dlv3.DeepLabV3Backbone()
    # backbone = models.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1').backbone
    
    downstream_model = SegmentationModel.load_from_checkpoint(checkpoint_filename,
                                                                backbone=backbone,
                                                                head=head)
    
    
    # downstream_model = dlv3.DeepLabV3Model.load_from_checkpoint(checkpoint_filename,
    #                                                             backbone=backbone,
    #                                                             head=head)
    
    return downstream_model

### --------------- Main -------------------------------------------------------------

# This function must not be changed. 
def main(SSL_technique_prefix): 
    import_name = 'pretreino_COCO_seam_ai_1s%'
    # import_name = 'supervised_f3_100%'
    
    # Load the pretrained model
    downstream_model = load_downstream_model(f'../saves/models/{SSL_technique_prefix}_{import_name}.ckpt')

    # Retrieve the train, validation and test sets.
    downstream_datamodule = build_downstream_datamodule()
    train_dl = downstream_datamodule.train_dataloader()
    val_dl   = downstream_datamodule.val_dataloader()
    test_dl  = downstream_datamodule.test_dataloader()    
    print(f'Data loaded: {import_name}')

    # Compute and report the mIoU metric for each subset
    # print(len(iter(train_dl)))
    report_IoU(downstream_model, train_dl, prefix="   Training dataset")
    # print(len(iter(val_dl)))
    report_IoU(downstream_model, val_dl,   prefix=" Validation dataset")
    # print(len(iter(test_dl)))
    report_IoU(downstream_model, test_dl,  prefix="       Test dataset")

if __name__ == "__main__":
    SSL_technique_prefix = "Byol"
    main(SSL_technique_prefix)
