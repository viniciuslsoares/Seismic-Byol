"""Instantiate materialized runs as executable Minerva pipelines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from minerva.models.loaders import FromPretrained
from minerva.models.nets.image.deeplabv3 import DeepLabV3, DeepLabV3Backbone
from minerva.models.ssl.byol import BYOL
from minerva.pipelines.experiment import (
    Experiment,
    ModelConfig,
    ModelInformation,
    ModelInstantiator,
)
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.transforms.transform import _Transform
from minerva.utils.instantiators import instantiate_cls
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, JaccardIndex
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from seismic_byol.environments import DatasetRegistry, ResolvedDataset
from seismic_byol.experiments import ConfigError
from seismic_byol.materialize import MaterializedRun
from seismic_byol.runtime_data import (
    build_finetune_data_module,
    build_pretrain_data_module,
)


@dataclass(frozen=True)
class RuntimeBuild:
    """Concrete objects required to execute one run."""

    materialized: MaterializedRun
    pipeline: Any
    model: L.LightningModule | None
    data_module: L.LightningDataModule
    task: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.materialized.run.run_id,
            "stage": self.materialized.run.stage,
            "task": self.task,
            "pipeline": _qualified_name(self.pipeline),
            "model": (
                _qualified_name(self.model) if self.model is not None else "lazy"
            ),
            "data_module": _qualified_name(self.data_module),
            "log_dir": str(self.pipeline.log_dir),
            "dataset": self.data_module.dataset_name,
            "train_samples": _dataset_length(self.data_module, "train_dataset"),
            "val_samples": _dataset_length(self.data_module, "val_dataset"),
            "test_samples": _dataset_length(self.data_module, "test_dataset"),
        }

    def execute(self, *, debug: bool = False):
        """Execute the configured Minerva task."""

        if isinstance(self.pipeline, Experiment):
            return self.pipeline.run(task=self.task, debug=debug)
        if debug:
            raise ConfigError(
                "Debug execution is currently supported only for downstream "
                "Minerva Experiment runs."
            )
        return self.pipeline.run(self.data_module, task=self.task)


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _dataset_length(data_module: Any, attribute: str) -> int | None:
    dataset = getattr(data_module, attribute, None)
    return len(dataset) if dataset is not None else None


def _component(specification: Mapping[str, Any], expected_type: type):
    class_path = specification.get("class_path")
    init_args = specification.get("init_args", {})
    if not isinstance(class_path, str) or not isinstance(init_args, Mapping):
        raise ConfigError("Minerva component requires class_path and mapping init_args.")
    configured_type = _import_class(class_path)
    if not issubclass(configured_type, expected_type):
        raise ConfigError(
            f"{class_path} is not a subclass of {_qualified_name(expected_type)}."
        )
    return instantiate_cls(configured_type, dict(init_args))


def _import_class(class_path: str) -> type:
    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ConfigError(f"Invalid class_path {class_path!r}.")
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"Cannot import Minerva component {class_path!r}.") from exc
    if not isinstance(cls, type):
        raise ConfigError(f"Configured component {class_path!r} is not a class.")
    return cls


def _minerva_parameters(materialized: MaterializedRun) -> Mapping[str, Any]:
    parameters = materialized.run.parameters.get("minerva")
    if not isinstance(parameters, Mapping):
        raise ConfigError(f"Run {materialized.run.run_id} has no Minerva parameters.")
    return parameters


class ReproducibleExperiment(Experiment):
    """Minerva Experiment with deterministic trainer settings."""

    def _trainer_parameters(self, enable_logging: bool = True, debug: bool = False):
        parameters = super()._trainer_parameters(enable_logging, debug)
        parameters["deterministic"] = True
        parameters["benchmark"] = False
        return parameters


class SeismicModelInstantiator(ModelInstantiator):
    """Create Minerva DeepLabV3 models for every configured backbone source."""

    def __init__(self, materialized: MaterializedRun):
        self.materialized = materialized
        values = materialized.run.values
        self.source = str(values["pretrain"])
        self.linear = values["head"] == "linear"
        self.freeze = values["backbone_mode"] == "frozen"
        self.num_classes = int(materialized.run.parameters["num_classes"])
        self.learning_rate = float(materialized.run.parameters["learning_rate"])

    def _random_or_builtin_backbone(self) -> DeepLabV3Backbone:
        if self.source == "scratch":
            return DeepLabV3Backbone(num_classes=self.num_classes)
        if self.source == "imagenet":
            return DeepLabV3Backbone(
                num_classes=self.num_classes,
                pretrained=True,
            )
        if self.source == "coco":
            backbone = DeepLabV3Backbone(num_classes=self.num_classes)
            coco = deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
            ).backbone
            missing, unexpected = backbone.RN50model.load_state_dict(
                coco.state_dict(), strict=False
            )
            if unexpected:
                raise ConfigError(f"Unexpected COCO backbone keys: {unexpected}")
            allowed_missing = {"fc.weight", "fc.bias"}
            if set(missing) - allowed_missing:
                raise ConfigError(f"Missing COCO backbone keys: {missing}")
            return backbone
        raise ConfigError(
            f"Backbone source {self.source!r} requires a pretrain checkpoint."
        )

    def _backbone_from_byol(self, checkpoint_path: str | Path) -> nn.Module:
        byol = BYOL(
            backbone=DeepLabV3Backbone(num_classes=self.num_classes),
            learning_rate=self.learning_rate,
        )
        loaded = FromPretrained(
            model=byol,
            ckpt_path=checkpoint_path,
            strict=False,
            error_on_missing_keys=False,
        )
        return loaded.backbone

    def _model(self, backbone: nn.Module) -> DeepLabV3:
        prediction_head = (
            nn.Conv2d(2048, self.num_classes, kernel_size=1)
            if self.linear
            else None
        )
        return DeepLabV3(
            backbone=backbone,
            pred_head=prediction_head,
            learning_rate=self.learning_rate,
            num_classes=self.num_classes,
            freeze_backbone=self.freeze,
            optimizer_kwargs={"weight_decay": 0.0},
        )

    def create_model_randomly_initialized(self) -> L.LightningModule:
        return self._model(self._random_or_builtin_backbone())

    def create_model_and_load_backbone(
        self, backbone_checkpoint_path: str | Path
    ) -> L.LightningModule:
        return self._model(self._backbone_from_byol(backbone_checkpoint_path))

    def load_model_from_checkpoint(
        self, checkpoint_path: str | Path
    ) -> L.LightningModule:
        model = self._model(DeepLabV3Backbone(num_classes=self.num_classes))
        return FromPretrained(model=model, ckpt_path=checkpoint_path)


def _build_pretrain(
    materialized: MaterializedRun,
    *,
    accelerator: str | None,
    num_workers: int | None,
) -> RuntimeBuild:
    parameters = materialized.run.parameters
    minerva_parameters = _minerva_parameters(materialized)
    dataset = materialized.inputs["dataset"]
    if not isinstance(dataset, ResolvedDataset):
        raise ConfigError("Pretrain runtime input is not a resolved dataset.")

    transform = _component(minerva_parameters["transforms"], _Transform)
    input_size = tuple(parameters["input_size"])
    data_module = build_pretrain_data_module(
        dataset,
        transform=transform,
        input_size=input_size,
        batch_size=int(parameters["batch_size"]),
        num_workers=num_workers,
    )
    model = _component(minerva_parameters["model"], BYOL)

    trainer_parameters = deepcopy(minerva_parameters["trainer"]["init_args"])
    if accelerator is not None:
        trainer_parameters["accelerator"] = accelerator
        trainer_parameters["devices"] = 1
    trainer_parameters.update(
        {
            "logger": CSVLogger(
                save_dir=materialized.outputs["logs"],
                name="",
                version="",
            ),
            "callbacks": [
                ModelCheckpoint(
                    dirpath=materialized.outputs["checkpoints"],
                    filename="{step:06d}",
                    every_n_train_steps=max(
                        1,
                        int(parameters["max_steps"])
                        // int(parameters["checkpoints"]),
                    ),
                    save_top_k=-1,
                    save_last=True,
                )
            ],
            "deterministic": True,
            "benchmark": False,
            "max_epochs": -1,
        }
    )
    trainer = L.Trainer(**trainer_parameters)
    pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=materialized.outputs["run_root"],
        save_run_status=True,
        seed=int(materialized.run.values["seed"]),
    )
    return RuntimeBuild(
        materialized=materialized,
        pipeline=pipeline,
        model=model,
        data_module=data_module,
        task="fit",
    )


def _constant_step_epochs(
    data_module: L.LightningDataModule,
    *,
    full_train_samples: int,
    batch_size: int,
    full_epochs: int,
) -> int:
    target_steps = max(1, full_train_samples // batch_size) * full_epochs
    selected_samples = len(data_module.train_dataset)  # type: ignore[arg-type]
    selected_steps = max(1, selected_samples // batch_size)
    return math.ceil(target_steps / selected_steps)


def _build_finetune(
    materialized: MaterializedRun,
    registry: DatasetRegistry,
    *,
    accelerator: str | None,
    num_workers: int | None,
) -> RuntimeBuild:
    run = materialized.run
    parameters = run.parameters
    minerva_parameters = _minerva_parameters(materialized)
    dataset = materialized.inputs["dataset"]
    if not isinstance(dataset, ResolvedDataset):
        raise ConfigError("Finetune runtime input is not a resolved dataset.")
    definition = registry.datasets[dataset.name]
    cap = run.values["cap"]
    batch_size = int(parameters["batch_size"])
    data_module = build_finetune_data_module(
        dataset,
        definition,
        cap=cap,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    full_train_samples = len(
        build_finetune_data_module(
            dataset,
            definition,
            cap="full",
            batch_size=batch_size,
            num_workers=0,
        ).train_dataset
    )
    max_epochs = _constant_step_epochs(
        data_module,
        full_train_samples=full_train_samples,
        batch_size=batch_size,
        full_epochs=int(parameters["full_data_epochs"]),
    )

    model_instantiator = SeismicModelInstantiator(materialized)
    model_name = (
        f"deeplabv3-{run.values['head']}-"
        f"{run.values['backbone_mode']}-{run.values['pretrain']}"
    )
    model_config = ModelConfig(
        instantiator=model_instantiator,
        information=ModelInformation(
            name=model_name,
            backbone_name="resnet50",
            task_type="segmentation",
            input_shape=(3, *definition.padding),  # type: ignore[misc]
            output_shape=(int(parameters["num_classes"]), *definition.padding),  # type: ignore[misc]
            num_classes=int(parameters["num_classes"]),
            return_logits=True,
        ),
    )
    pipeline_parameters = deepcopy(minerva_parameters["pipeline"]["init_args"])
    pipeline_parameters.update(
        {
            "experiment_name": run.experiment_name,
            "model_config": model_config,
            "data_module": data_module,
            "root_log_dir": materialized.environment.output_root,
            "execution_id": run.run_id,
            "max_epochs": max_epochs,
            "seed": int(run.values["seed"]),
            "_run_id": run.run_id,
            "evaluation_metrics": {
                "mIoU": JaccardIndex(
                    task="multiclass",
                    num_classes=int(parameters["num_classes"]),
                    average="macro",
                ),
                "accuracy": Accuracy(
                    task="multiclass",
                    num_classes=int(parameters["num_classes"]),
                ),
                "f1_weighted": F1Score(
                    task="multiclass",
                    num_classes=int(parameters["num_classes"]),
                    average="weighted",
                ),
            },
        }
    )
    if accelerator is not None:
        pipeline_parameters["accelerator"] = accelerator
        pipeline_parameters["devices"] = 1
    backbone = materialized.inputs["backbone"]
    if backbone["kind"] == "minerva_checkpoint":
        pipeline_parameters["pretrained_backbone_ckpt_path"] = backbone["checkpoint"]

    pipeline = ReproducibleExperiment(**pipeline_parameters)
    return RuntimeBuild(
        materialized=materialized,
        pipeline=pipeline,
        model=None,
        data_module=data_module,
        task=str(parameters["task"]),
    )


def build_runtime(
    materialized: MaterializedRun,
    registry: DatasetRegistry,
    *,
    accelerator: str | None = None,
    num_workers: int | None = None,
) -> RuntimeBuild:
    """Build all Minerva objects for one materialized run."""

    if materialized.issues:
        raise ConfigError(
            f"Cannot build {materialized.run.run_id}: dataset path validation failed."
        )
    if materialized.run.stage == "pretrain":
        return _build_pretrain(
            materialized,
            accelerator=accelerator,
            num_workers=num_workers,
        )
    if materialized.run.stage == "finetune":
        return _build_finetune(
            materialized,
            registry,
            accelerator=accelerator,
            num_workers=num_workers,
        )
    raise ConfigError(f"Unsupported runtime stage {materialized.run.stage!r}.")
