"""Load, validate, and expand declarative experiment matrices."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml


SUPPORTED_SCHEMA_VERSION = 1
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class MatrixDefinition:
    """One Cartesian experiment matrix within a research workflow."""

    name: str
    stage: str
    axes: Mapping[str, tuple[Any, ...]]
    parameters: Mapping[str, Any]
    exclude: tuple[Mapping[str, Any], ...] = ()
    include: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration."""

    schema_version: int
    name: str
    description: str
    metadata: Mapping[str, Any]
    matrices: tuple[MatrixDefinition, ...]
    source: Path


@dataclass(frozen=True)
class ResolvedRun:
    """An immutable, fully resolved member of an experiment matrix."""

    schema_version: int
    experiment_name: str
    matrix_name: str
    stage: str
    values: Mapping[str, Any]
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]
    run_id: str

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable run manifest."""

        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment_name,
            "run": {
                "id": self.run_id,
                "matrix": self.matrix_name,
                "stage": self.stage,
                "values": deepcopy(dict(self.values)),
            },
            "metadata": deepcopy(dict(self.metadata)),
            "parameters": deepcopy(dict(self.parameters)),
        }


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping.")
    return value


def _require_name(value: Any, location: str) -> str:
    if not isinstance(value, str) or not _NAME_PATTERN.fullmatch(value):
        raise ConfigError(
            f"{location} must contain only lowercase letters, numbers, '-' or '_', "
            "and must start with a letter or number."
        )
    return value


def _canonical_json(value: Any, location: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{location} must contain only JSON-compatible values.") from exc


def _validate_selectors(
    raw_selectors: Any,
    *,
    axes: Mapping[str, tuple[Any, ...]],
    matrix_location: str,
    selector_name: str,
    require_all_axes: bool,
) -> tuple[Mapping[str, Any], ...]:
    if raw_selectors is None:
        return ()
    if not isinstance(raw_selectors, list):
        raise ConfigError(f"{matrix_location}.{selector_name} must be a list.")

    selectors: list[Mapping[str, Any]] = []
    axis_names = set(axes)
    for index, raw_selector in enumerate(raw_selectors):
        location = f"{matrix_location}.{selector_name}[{index}]"
        selector = dict(_require_mapping(raw_selector, location))
        selector_names = set(selector)
        unknown = selector_names - axis_names
        if unknown:
            raise ConfigError(
                f"{location} contains unknown axes: {', '.join(sorted(unknown))}."
            )
        if not selector:
            raise ConfigError(f"{location} cannot be empty.")
        if require_all_axes and selector_names != axis_names:
            missing = axis_names - selector_names
            raise ConfigError(
                f"{location} must define every axis; missing: "
                f"{', '.join(sorted(missing))}."
            )
        _canonical_json(selector, location)
        selectors.append(selector)
    return tuple(selectors)


def _parse_matrix(raw: Any, index: int) -> MatrixDefinition:
    location = f"matrices[{index}]"
    matrix = _require_mapping(raw, location)
    name = _require_name(matrix.get("name"), f"{location}.name")
    stage = _require_name(matrix.get("stage"), f"{location}.stage")

    raw_axes = _require_mapping(matrix.get("axes"), f"{location}.axes")
    if not raw_axes:
        raise ConfigError(f"{location}.axes cannot be empty.")

    axes: dict[str, tuple[Any, ...]] = {}
    for raw_axis_name, raw_values in raw_axes.items():
        axis_name = _require_name(raw_axis_name, f"{location}.axes key")
        if not isinstance(raw_values, list) or not raw_values:
            raise ConfigError(f"{location}.axes.{axis_name} must be a non-empty list.")
        canonical_values = [
            _canonical_json(value, f"{location}.axes.{axis_name}") for value in raw_values
        ]
        if len(set(canonical_values)) != len(canonical_values):
            raise ConfigError(f"{location}.axes.{axis_name} contains duplicate values.")
        axes[axis_name] = tuple(raw_values)

    parameters = matrix.get("parameters", {})
    parameters = _require_mapping(parameters, f"{location}.parameters")
    _canonical_json(parameters, f"{location}.parameters")

    exclude = _validate_selectors(
        matrix.get("exclude"),
        axes=axes,
        matrix_location=location,
        selector_name="exclude",
        require_all_axes=False,
    )
    include = _validate_selectors(
        matrix.get("include"),
        axes=axes,
        matrix_location=location,
        selector_name="include",
        require_all_axes=True,
    )

    return MatrixDefinition(
        name=name,
        stage=stage,
        axes=axes,
        parameters=dict(parameters),
        exclude=exclude,
        include=include,
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")

    try:
        with source.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {source}: {exc}") from exc

    root = _require_mapping(document, "configuration")
    schema_version = root.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION}, got {schema_version!r}."
        )

    raw_experiment = _require_mapping(root.get("experiment"), "experiment")
    name = _require_name(raw_experiment.get("name"), "experiment.name")
    description = raw_experiment.get("description", "")
    if not isinstance(description, str):
        raise ConfigError("experiment.description must be a string.")
    metadata = _require_mapping(raw_experiment.get("metadata", {}), "experiment.metadata")
    _canonical_json(metadata, "experiment.metadata")

    raw_matrices = root.get("matrices")
    if not isinstance(raw_matrices, list) or not raw_matrices:
        raise ConfigError("matrices must be a non-empty list.")
    matrices = tuple(_parse_matrix(raw, index) for index, raw in enumerate(raw_matrices))
    matrix_names = [matrix.name for matrix in matrices]
    if len(set(matrix_names)) != len(matrix_names):
        raise ConfigError("Matrix names must be unique within an experiment.")

    return ExperimentConfig(
        schema_version=schema_version,
        name=name,
        description=description,
        metadata=dict(metadata),
        matrices=matrices,
        source=source.resolve(),
    )


def _matches(values: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    return all(values.get(key) == expected for key, expected in selector.items())


def _run_id(
    config: ExperimentConfig,
    matrix: MatrixDefinition,
    values: Mapping[str, Any],
) -> str:
    identity = {
        "schema_version": config.schema_version,
        "experiment": config.name,
        "matrix": matrix.name,
        "stage": matrix.stage,
        "values": values,
        "metadata": config.metadata,
        "parameters": matrix.parameters,
    }
    digest = sha256(_canonical_json(identity, "resolved run").encode()).hexdigest()[:12]
    return f"{config.name}-{matrix.name}-{digest}"


def _expand_matrix(
    config: ExperimentConfig,
    matrix: MatrixDefinition,
) -> Iterable[ResolvedRun]:
    axis_names = tuple(matrix.axes)
    combinations = (
        dict(zip(axis_names, selected_values))
        for selected_values in product(*(matrix.axes[name] for name in axis_names))
    )

    unique: dict[str, Mapping[str, Any]] = {}
    for values in combinations:
        if not any(_matches(values, selector) for selector in matrix.exclude):
            unique[_canonical_json(values, "matrix combination")] = values
    for values in matrix.include:
        unique[_canonical_json(values, "included combination")] = dict(values)

    for values in unique.values():
        yield ResolvedRun(
            schema_version=config.schema_version,
            experiment_name=config.name,
            matrix_name=matrix.name,
            stage=matrix.stage,
            values=values,
            parameters=deepcopy(dict(matrix.parameters)),
            metadata=deepcopy(dict(config.metadata)),
            run_id=_run_id(config, matrix, values),
        )


def expand_experiment(
    config: ExperimentConfig,
    only: Mapping[str, Any] | None = None,
) -> list[ResolvedRun]:
    """Expand every matrix and optionally retain matching runs."""

    filters = dict(only or {})
    known_axes = {axis for matrix in config.matrices for axis in matrix.axes}
    unknown_filters = set(filters) - known_axes - {"matrix", "stage"}
    if unknown_filters:
        raise ConfigError(
            f"Unknown filter axes: {', '.join(sorted(unknown_filters))}."
        )

    runs = [run for matrix in config.matrices for run in _expand_matrix(config, matrix)]
    if filters:
        runs = [
            run
            for run in runs
            if all(
                (run.matrix_name == value if key == "matrix" else
                 run.stage == value if key == "stage" else
                 run.values.get(key) == value)
                for key, value in filters.items()
            )
        ]
    return runs


def write_run_manifests(runs: Sequence[ResolvedRun], directory: str | Path) -> Path:
    """Write one immutable YAML manifest per resolved run."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        target = output_dir / f"{run.run_id}.yaml"
        with target.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(run.as_dict(), stream, sort_keys=False)
    return output_dir
