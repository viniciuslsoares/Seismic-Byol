"""Explicit dataset and output resolution for execution environments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml

from seismic_byol.experiments import ConfigError


SUPPORTED_SCHEMA_VERSION = 1
ENVIRONMENT_VARIABLE = "SEISMIC_ENV"
DATA_ROOT_VARIABLE = "SEISMIC_DATA_ROOT"
OUTPUT_ROOT_VARIABLE = "SEISMIC_OUTPUT_ROOT"
CONFIG_ROOT_VARIABLE = "SEISMIC_CONFIG_ROOT"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class DatasetDefinition:
    """Filesystem-independent description of one logical dataset."""

    name: str
    kind: str
    roles: tuple[str, ...]
    pretrain_path: Path | None
    required: Mapping[str, tuple[Path, ...]]


@dataclass(frozen=True)
class DatasetRegistry:
    """Validated logical dataset registry."""

    datasets: Mapping[str, DatasetDefinition]
    source: Path


@dataclass(frozen=True)
class EnvironmentProfile:
    """Resolved roots and dataset locations for one execution environment."""

    name: str
    data_root: Path
    output_root: Path
    dataset_paths: Mapping[str, Path]
    source: Path


@dataclass(frozen=True)
class ResolvedDataset:
    """A logical dataset resolved for a role in one environment."""

    name: str
    kind: str
    role: str
    root: Path | None
    input_path: Path | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "root": str(self.root) if self.root is not None else None,
            "input_path": str(self.input_path) if self.input_path is not None else None,
        }


@dataclass(frozen=True)
class PathIssue:
    """A missing or invalid filesystem path."""

    dataset: str
    role: str
    path: Path
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "role": self.role,
            "path": str(self.path),
            "message": self.message,
        }


def _load_yaml(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigError(f"{label} file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ConfigError(f"{label} file must contain a mapping: {path}")
    if document.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"{label} schema_version must be {SUPPORTED_SCHEMA_VERSION}: {path}"
        )
    return document


def _require_name(value: Any, location: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ConfigError(
            f"{location} must contain lowercase letters, numbers, '-' or '_'."
        )
    return value


def _require_string_list(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{location} must be a non-empty list.")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{location} must contain non-empty strings.")
    if len(set(value)) != len(value):
        raise ConfigError(f"{location} contains duplicate values.")
    return tuple(value)


def load_dataset_registry(config_root: str | Path) -> DatasetRegistry:
    """Load the logical dataset registry from ``configs/datasets.yaml``."""

    source = Path(config_root) / "datasets.yaml"
    document = _load_yaml(source, "Dataset registry")
    raw_datasets = document.get("datasets")
    if not isinstance(raw_datasets, Mapping) or not raw_datasets:
        raise ConfigError("datasets must be a non-empty mapping.")

    datasets: dict[str, DatasetDefinition] = {}
    for raw_name, raw_definition in raw_datasets.items():
        name = _require_name(raw_name, "dataset name")
        if not isinstance(raw_definition, Mapping):
            raise ConfigError(f"datasets.{name} must be a mapping.")
        kind = _require_name(raw_definition.get("kind"), f"datasets.{name}.kind")
        roles = _require_string_list(
            raw_definition.get("roles"), f"datasets.{name}.roles"
        )
        raw_pretrain_path = raw_definition.get("pretrain_path")
        if raw_pretrain_path is not None and not isinstance(raw_pretrain_path, str):
            raise ConfigError(f"datasets.{name}.pretrain_path must be a string.")
        pretrain_path = (
            Path(raw_pretrain_path) if raw_pretrain_path is not None else None
        )

        raw_required = raw_definition.get("required", {})
        if not isinstance(raw_required, Mapping):
            raise ConfigError(f"datasets.{name}.required must be a mapping.")
        required: dict[str, tuple[Path, ...]] = {}
        for role, raw_paths in raw_required.items():
            if role not in roles:
                raise ConfigError(
                    f"datasets.{name}.required defines undeclared role {role!r}."
                )
            required[role] = tuple(
                Path(path)
                for path in _require_string_list(
                    raw_paths, f"datasets.{name}.required.{role}"
                )
            )

        datasets[name] = DatasetDefinition(
            name=name,
            kind=kind,
            roles=roles,
            pretrain_path=pretrain_path,
            required=required,
        )

    return DatasetRegistry(datasets=datasets, source=source.resolve())


def resolve_environment_name(
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve environment with precedence CLI argument over environment variable."""

    variables = os.environ if environ is None else environ
    environment = explicit or variables.get(ENVIRONMENT_VARIABLE)
    if not environment:
        raise ConfigError(
            f"Execution environment is required; pass --environment or set "
            f"{ENVIRONMENT_VARIABLE}."
        )
    return _require_name(environment, "environment name")


def _dataset_override_variable(dataset_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]", "_", dataset_name).upper()
    return f"SEISMIC_DATASET_{normalized}_ROOT"


def load_environment(
    name: str,
    config_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> EnvironmentProfile:
    """Load an environment profile and apply filesystem overrides."""

    checked_name = _require_name(name, "environment name")
    source = Path(config_root) / "environments" / f"{checked_name}.yaml"
    document = _load_yaml(source, "Environment")
    raw_environment = document.get("environment")
    if not isinstance(raw_environment, Mapping):
        raise ConfigError("environment must be a mapping.")
    declared_name = _require_name(raw_environment.get("name"), "environment.name")
    if declared_name != checked_name:
        raise ConfigError(
            f"Environment file {source} declares name {declared_name!r}, "
            f"expected {checked_name!r}."
        )

    variables = os.environ if environ is None else environ
    raw_data_root = variables.get(DATA_ROOT_VARIABLE, raw_environment.get("data_root"))
    raw_output_root = variables.get(
        OUTPUT_ROOT_VARIABLE, raw_environment.get("output_root")
    )
    if not isinstance(raw_data_root, str) or not raw_data_root:
        raise ConfigError("environment.data_root must be a non-empty string.")
    if not isinstance(raw_output_root, str) or not raw_output_root:
        raise ConfigError("environment.output_root must be a non-empty string.")
    data_root = Path(os.path.expandvars(os.path.expanduser(raw_data_root)))
    output_root = Path(os.path.expandvars(os.path.expanduser(raw_output_root)))

    raw_dataset_paths = raw_environment.get("datasets")
    if not isinstance(raw_dataset_paths, Mapping):
        raise ConfigError("environment.datasets must be a mapping.")
    dataset_paths: dict[str, Path] = {}
    for raw_dataset_name, raw_path in raw_dataset_paths.items():
        dataset_name = _require_name(raw_dataset_name, "environment dataset name")
        if not isinstance(raw_path, str) or not raw_path:
            raise ConfigError(
                f"environment.datasets.{dataset_name} must be a non-empty string."
            )
        override = variables.get(_dataset_override_variable(dataset_name))
        path = Path(
            os.path.expandvars(os.path.expanduser(override or raw_path))
        )
        dataset_paths[dataset_name] = path if path.is_absolute() else data_root / path

    return EnvironmentProfile(
        name=declared_name,
        data_root=data_root,
        output_root=output_root,
        dataset_paths=dataset_paths,
        source=source.resolve(),
    )


def resolve_dataset(
    name: str,
    role: str,
    registry: DatasetRegistry,
    environment: EnvironmentProfile,
) -> ResolvedDataset:
    """Resolve one dataset and enforce its declared role."""

    try:
        definition = registry.datasets[name]
    except KeyError as exc:
        raise ConfigError(f"Dataset {name!r} is not registered.") from exc
    if role not in definition.roles:
        raise ConfigError(f"Dataset {name!r} does not support role {role!r}.")

    root = environment.dataset_paths.get(name)
    if definition.kind in {"builtin_backbone", "random_initialization"}:
        return ResolvedDataset(
            name=name,
            kind=definition.kind,
            role=role,
            root=None,
            input_path=None,
        )
    if root is None:
        raise ConfigError(
            f"Dataset {name!r} has no path in environment {environment.name!r}."
        )

    input_path = root
    if role == "pretrain" and definition.pretrain_path is not None:
        input_path = root / definition.pretrain_path
    return ResolvedDataset(
        name=name,
        kind=definition.kind,
        role=role,
        root=root,
        input_path=input_path,
    )


def validate_dataset_paths(
    dataset: ResolvedDataset,
    registry: DatasetRegistry,
) -> tuple[PathIssue, ...]:
    """Validate the directories required by a resolved dataset role."""

    if dataset.root is None:
        return ()
    definition = registry.datasets[dataset.name]
    issues: list[PathIssue] = []
    for relative_path in definition.required.get(dataset.role, ()):
        path = dataset.root / relative_path
        if not path.is_dir():
            issues.append(
                PathIssue(
                    dataset=dataset.name,
                    role=dataset.role,
                    path=path,
                    message="required directory does not exist",
                )
            )
    return tuple(issues)


def with_output_root(
    environment: EnvironmentProfile,
    output_root: str | Path,
) -> EnvironmentProfile:
    """Return an environment with an explicit output override."""

    return replace(environment, output_root=Path(output_root))
