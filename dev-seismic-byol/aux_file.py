
import numpy as np
from pathlib import Path

from functions import *

from minerva.transforms.transform import *
from minerva.transforms.random_transform import *

from minerva.data.readers import TiffReader, PartialPatchedZarrReader, NumpyArrayReader
from minerva.data.datasets import SimpleDataset
from minerva.data.data_modules import MinervaDataModule

from minerva.models.ssl.byol import BYOL
from minerva.models.nets.image.deeplabv3 import DeepLabV3Backbone

from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer

from a700 import A700DataModule


# data_path = '/home/vinicius.soares/asml/datasets/tiff_data/f3_segmentation/images'

data_path = '/home/vinicius.soares/asml/datasets/seismic-attributes-calculation/raw/S0.n/K1/data'

input_size = (256, 256)

model_name = 'aux'

# Transforms
random_flip = RandomFlip(possible_axis=[1])
random_crop = RandomCrop(crop_size=input_size)
random_rotation = RandomRotation(degrees=25, prob=0.2)
transpose_to_HWC = Transpose([1, 2, 0])
transpose_to_CHW = Transpose([2, 0, 1])
cast_to_tensor = CastTo(dtype=np.float32)
repeat = Repeat(axis=2, n_repetitions=3)

byol_transform_pipeline = TransformPipeline([
    transpose_to_HWC,
    repeat,
    random_crop,
    random_flip,
    random_rotation,
    transpose_to_CHW,
])

# byol_transform_pipeline = TransformPipeline([
#     random_crop,
#     random_flip,
#     random_rotation,
#     transpose_to_CHW,
#     cast_to_tensor,
# ])

constrastive_transform = ContrastiveTransform(byol_transform_pipeline)

# Dataset

train_img_reader = PartialPatchedZarrReader(
    path=data_path,
    data_shape=(1, 6625, 2001),
    stride=(1, 6625,  2001),
    pad_width=None,
    index_bounds=[(2000, 0, 0), (2100, 6625, 2001)],
    )
    
# train_img_reader = TiffReader(path=data_path)    
    

pretrain_dataset = SimpleDataset(
    readers=train_img_reader,
    transforms=constrastive_transform,
    return_single=True
)

data_module = MinervaDataModule(
    train_dataset=pretrain_dataset,
    batch_size=8,
    drop_last=True,
    shuffle_train=True,
    # name=dataset_name
)

loader = data_module.train_dataloader()
data_iter = iter(loader)
item = next(data_iter)


print(item[0][0])

print(item[0].shape)

print(type(item[0]))

