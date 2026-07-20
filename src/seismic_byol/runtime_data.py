"""Minerva-native data builders used by executable experiment runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from minerva.data.data_modules import MinervaDataModule
from minerva.data.datasets import SimpleDataset
from minerva.data.datasets.binary_tree_subset import BinaryTreeSubset
from minerva.data.readers import PNGReader, TiffReader
from minerva.transforms.random_transform import (
    RandomCrop,
    RandomFlip,
    RandomRotation,
)
from minerva.transforms.transform import (
    CastTo,
    ContrastiveTransform,
    Padding,
    Repeat,
    TransformPipeline,
    Transpose,
    Unsqueeze,
)

from seismic_byol.environments import DatasetDefinition, ResolvedDataset
from seismic_byol.experiments import ConfigError


def default_num_workers() -> int:
    """Choose a conservative worker count for workstations and HPC nodes."""

    return min(8, os.cpu_count() or 1)


class A700Dataset(Dataset):
    """A700 NPY sections with deterministic 90/10 split and sample z-score."""

    def __init__(
        self,
        root: str | Path,
        partition: Literal["train", "val"],
        transform: Callable[[np.ndarray], Any],
    ):
        root = Path(root)
        files: list[Path] = []
        for orientation in ("iline", "xline"):
            orientation_files = sorted((root / orientation).rglob("*.npy"))
            files.extend(
                path
                for index, path in enumerate(orientation_files)
                if ((index % 100) < 90) == (partition == "train")
            )
        if not files:
            raise ConfigError(
                f"A700 {partition!r} partition contains no NPY files under {root}."
            )
        self.files = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = np.load(self.files[index]).astype(np.float32, copy=False)
        standard_deviation = float(image.std())
        if standard_deviation == 0:
            raise ValueError(f"Cannot z-score constant A700 sample {self.files[index]}.")
        normalized = (image - float(image.mean())) / standard_deviation
        return self.transform(normalized)


def _a700_transform(input_size: tuple[int, int]) -> ContrastiveTransform:
    return ContrastiveTransform(
        TransformPipeline(
            [
                RandomCrop(crop_size=input_size),
                RandomFlip(possible_axis=1, prob=0.5),
                RandomRotation(degrees=5, prob=1.0),
                Unsqueeze(axis=0),
                Repeat(axis=0, n_repetitions=3),
                CastTo(dtype="float32"),
            ]
        )
    )


def build_pretrain_data_module(
    dataset: ResolvedDataset,
    *,
    transform: Callable[[np.ndarray], Any],
    input_size: tuple[int, int],
    batch_size: int,
    num_workers: int | None = None,
) -> MinervaDataModule:
    """Build a Minerva data module for TIFF or A700 BYOL pretraining."""

    if dataset.input_path is None:
        raise ConfigError(f"Pretrain dataset {dataset.name!r} has no input path.")
    workers = default_num_workers() if num_workers is None else num_workers

    if dataset.kind == "a700":
        a700_transform = _a700_transform(input_size)
        train_dataset = A700Dataset(dataset.input_path, "train", a700_transform)
        val_dataset = A700Dataset(dataset.input_path, "val", a700_transform)
    else:
        reader = TiffReader(path=dataset.input_path)
        if len(reader) == 0:
            raise ConfigError(f"No TIFF files found under {dataset.input_path}.")
        train_dataset = SimpleDataset(
            readers=reader,
            transforms=transform,
            return_single=True,
        )
        val_dataset = None

    return MinervaDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        drop_last=True,
        shuffle_train=True,
        name=dataset.name,
        additional_train_dataloader_kwargs={"pin_memory": True},
        additional_val_dataloader_kwargs={"pin_memory": True},
    )


def _paired_dataset(
    root: Path,
    partition: str,
    image_transform: TransformPipeline,
    mask_transform: TransformPipeline,
    filters: str | None = None,
) -> SimpleDataset:
    image_reader = TiffReader(root / "images" / partition, filters=filters)
    mask_reader = PNGReader(root / "annotations" / partition, filters=filters)
    if len(image_reader) == 0:
        suffix = f" matching {filters!r}" if filters else ""
        raise ConfigError(
            f"No TIFF images found in {root / 'images' / partition}{suffix}."
        )
    if len(image_reader) != len(mask_reader):
        raise ConfigError(
            f"Image/mask count mismatch in {partition}: "
            f"{len(image_reader)} images and {len(mask_reader)} masks."
        )
    image_stems = [path.stem for path in image_reader.files]
    mask_stems = [path.stem for path in mask_reader.files]
    if image_stems != mask_stems:
        raise ConfigError(f"Image/mask filenames do not match in {partition}.")
    return SimpleDataset(
        readers=[image_reader, mask_reader],
        transforms=[image_transform, mask_transform],
    )


def _balanced_subset(
    root: Path,
    cap: int,
    image_transform: TransformPipeline,
    mask_transform: TransformPipeline,
) -> Dataset:
    inline = _paired_dataset(
        root, "train", image_transform, mask_transform, filters=r"il.*"
    )
    crossline = _paired_dataset(
        root, "train", image_transform, mask_transform, filters=r"xl.*"
    )
    available = len(inline) + len(crossline)
    if cap > available:
        raise ConfigError(f"Requested cap {cap}, but dataset contains {available} samples.")

    inline_size = min((cap + 1) // 2, len(inline))
    crossline_size = min(cap - inline_size, len(crossline))
    remaining = cap - inline_size - crossline_size
    if remaining:
        inline_size += min(remaining, len(inline) - inline_size)
        remaining = cap - inline_size - crossline_size
    if remaining:
        crossline_size += min(remaining, len(crossline) - crossline_size)

    subsets: list[Dataset] = []
    if inline_size:
        subsets.append(BinaryTreeSubset(inline, inline_size))
    if crossline_size:
        subsets.append(BinaryTreeSubset(crossline, crossline_size))
    return subsets[0] if len(subsets) == 1 else ConcatDataset(subsets)


def build_finetune_data_module(
    dataset: ResolvedDataset,
    definition: DatasetDefinition,
    *,
    cap: int | str,
    batch_size: int,
    num_workers: int | None = None,
) -> MinervaDataModule:
    """Build the supervised Minerva data module used by downstream runs."""

    if dataset.root is None:
        raise ConfigError(f"Finetune dataset {dataset.name!r} has no root path.")
    if definition.padding is None:
        raise ConfigError(f"Dataset {dataset.name!r} has no padding shape.")
    workers = default_num_workers() if num_workers is None else num_workers
    height, width = definition.padding
    image_transform = TransformPipeline(
        [
            Padding(height, width),
            Transpose((2, 0, 1)),
            CastTo(dtype="float32"),
        ]
    )
    mask_transform = TransformPipeline(
        [
            Padding(height, width),
            Transpose((2, 0, 1)),
            CastTo(dtype="int64"),
        ]
    )

    if cap == "full":
        train_dataset: Dataset = _paired_dataset(
            dataset.root, "train", image_transform, mask_transform
        )
    elif isinstance(cap, int) and cap > 0:
        train_dataset = _balanced_subset(
            dataset.root, cap, image_transform, mask_transform
        )
    else:
        raise ConfigError(f"Unsupported downstream cap {cap!r}.")

    val_dataset = _paired_dataset(
        dataset.root, "val", image_transform, mask_transform
    )
    test_dataset = _paired_dataset(
        dataset.root, "test", image_transform, mask_transform
    )
    return MinervaDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_split="test",
        batch_size=batch_size,
        num_workers=workers,
        drop_last=False,
        shuffle_train=True,
        name=dataset.name,
        additional_train_dataloader_kwargs={"pin_memory": True, "drop_last": True},
        additional_val_dataloader_kwargs={"pin_memory": True},
        additional_test_dataloader_kwargs={"pin_memory": True},
    )
