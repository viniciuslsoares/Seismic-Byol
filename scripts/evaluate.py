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
    return iou, f1

### ---------- DataModule -----------------------------------------------------------

# This function must instantiate and configure the datamodule for the downstream task.
def build_downstream_datamodule(data) -> L.LightningDataModule:
    
    if data == 'f3':
        print("F3 datas being used")
        return F3SeismicDataModule(root_dir="../data/", batch_size=8, cap=1)

    elif data == 'seam_ai':
        print("Parihaka datas being used")
        return ParihakaSeismicDataModule(root_dir="../data/", batch_size=8, cap=1)
        
    else:
        raise ValueError(f"Unknown dataset: {data}")

### ------------- Pretrained Model --------------------------------------------------

# This function must instantiate the downstream model and load its weights
# from checkpoint_filename.

def load_downstream_model(checkpoint_filename, mode:str = 'byol') -> L.LightningModule:
    
    backbone = models.deeplabv3_resnet50().backbone

    if mode == 'byol' or mode == 'supervised':
        print('***** Backbone carregado *****')    
    
    elif mode == 'coco':
        print('***** Backbone COCO carregado *****')
        backbone = models.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1').backbone
        # Just to ensure that the model is the same as the one used in the training
    
    elif mode == 'imagenet':
        print('***** Backbone IMAGENET carregado *****')
        backbone = dlv3.DeepLabV3Backbone(num_classes=6, pretrain='imagenet')
        # This backbone is slightly different from the others, so it is necessary to load it again

    pred_head = dlv3.DeepLabV3PredictionHead(num_classes=6)    
        
    downstream_model = SegmentationModel.load_from_checkpoint(checkpoint_filename,
                                                                num_classes=6,                                                             
                                                                backbone=backbone,
                                                                head=pred_head,
                                                                loss_fn=torch.nn.CrossEntropyLoss(),
                                                                learning_rate=0.001,
                                                                freeze_backbone=False)
    
    return downstream_model

### --------------- Main -------------------------------------------------------------

# This function must not be changed. 
def eval_func(import_name:str,
              mode:str = 'byol',
              dataset:str = 'f3',
              repetition:str = 'Vx',
              ): 


    # import_name = 'pretreino_COCO_seam_ai_1s%'
    
    # Load the pretrained model
    downstream_model = load_downstream_model(f'../saves/models/{repetition}/{import_name}.ckpt', mode=mode)

    # Retrieve the train, validation and test sets.
    downstream_datamodule = build_downstream_datamodule(data=dataset)
    train_dl = downstream_datamodule.train_dataloader()
    val_dl   = downstream_datamodule.val_dataloader()
    test_dl  = downstream_datamodule.test_dataloader()    
    print(f' ----------- Data loaded: {import_name}')


    # Compute and report the mIoU metric for each subset
    # train_iou, train_f1 = report_IoU(downstream_model, train_dl, prefix="   Training dataset")
    # val_iou, val_f1 = report_IoU(downstream_model, val_dl,   prefix=" Validation dataset")
    test_iou, test_f1 = report_IoU(downstream_model, test_dl,  prefix="       Test dataset")

    train_iou, val_iou, train_f1, val_f1 = 0, 0, 0, 0

    return (train_iou, val_iou, test_iou), (train_f1, val_f1, test_f1)



# if __name__ == "__main__":
#     eval_func()
