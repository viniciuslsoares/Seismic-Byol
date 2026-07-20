"""Materialize scientific runs with environment-specific paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from seismic_byol.environments import (
    DatasetRegistry,
    EnvironmentProfile,
    PathIssue,
    ResolvedDataset,
    resolve_dataset,
    validate_dataset_paths,
)
from seismic_byol.experiments import (
    ConfigError,
    ExperimentConfig,
    ResolvedRun,
    expand_experiment,
)


LOCAL_PRETRAIN_SOURCES = frozenset({"f3_N", "seam_ai_N", "both_N", "a700"})
BUILTIN_BACKBONES = frozenset({"imagenet", "coco", "scratch"})


@dataclass(frozen=True)
class MaterializedRun:
    """A scientific run bound to one execution environment."""

    run: ResolvedRun
    environment: EnvironmentProfile
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Path]
    dependencies: tuple[Mapping[str, Any], ...]
    issues: tuple[PathIssue, ...]
    paths_checked: bool

    def as_dict(self) -> dict[str, Any]:
        document = self.run.as_dict()
        document["runtime"] = {
            "environment": {
                "name": self.environment.name,
                "data_root": str(self.environment.data_root),
                "output_root": str(self.environment.output_root),
                "profile": str(self.environment.source),
            },
            "inputs": _serialize(self.inputs),
            "outputs": {name: str(path) for name, path in self.outputs.items()},
            "dependencies": [dict(dependency) for dependency in self.dependencies],
            "path_validation": {
                "checked": self.paths_checked,
                "issues": [issue.as_dict() for issue in self.issues],
            },
        }
        return document


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ResolvedDataset):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _outputs(run: ResolvedRun, environment: EnvironmentProfile) -> dict[str, Path]:
    run_root = environment.output_root / run.experiment_name / run.run_id
    return {
        "run_root": run_root,
        "logs": run_root / "logs",
        "checkpoints": run_root / "checkpoints",
        "last_checkpoint": run_root / "checkpoints" / "last.ckpt",
        "results": run_root / "results",
    }


def _pretrain_index(runs: Sequence[ResolvedRun]) -> dict[tuple[str, Any], ResolvedRun]:
    index: dict[tuple[str, Any], ResolvedRun] = {}
    for run in runs:
        if run.stage != "pretrain":
            continue
        key = (run.values.get("pretrain"), run.values.get("seed"))
        if key in index:
            raise ConfigError(
                f"More than one pretrain run provides source={key[0]!r}, seed={key[1]!r}."
            )
        index[key] = run
    return index


def _materialize_one(
    run: ResolvedRun,
    *,
    environment: EnvironmentProfile,
    registry: DatasetRegistry,
    pretrain_index: Mapping[tuple[str, Any], ResolvedRun],
    check_paths: bool,
) -> MaterializedRun:
    inputs: dict[str, Any] = {}
    dependencies: list[Mapping[str, Any]] = []
    issues: list[PathIssue] = []

    if run.stage == "pretrain":
        source = run.values.get("pretrain")
        if not isinstance(source, str):
            raise ConfigError(f"Run {run.run_id} has no string pretrain source.")
        dataset = resolve_dataset(source, "pretrain", registry, environment)
        inputs["dataset"] = dataset
        if check_paths:
            issues.extend(validate_dataset_paths(dataset, registry))

    elif run.stage == "finetune":
        downstream = run.values.get("finetune")
        source = run.values.get("pretrain")
        seed = run.values.get("seed")
        if not isinstance(downstream, str) or not isinstance(source, str):
            raise ConfigError(
                f"Run {run.run_id} must define string pretrain and finetune values."
            )

        downstream_dataset = resolve_dataset(
            downstream, "finetune", registry, environment
        )
        inputs["dataset"] = downstream_dataset
        if check_paths:
            issues.extend(validate_dataset_paths(downstream_dataset, registry))

        if source in LOCAL_PRETRAIN_SOURCES:
            dependency_key = (source, seed)
            try:
                dependency = pretrain_index[dependency_key]
            except KeyError as exc:
                raise ConfigError(
                    f"Run {run.run_id} requires a pretrain run for "
                    f"source={source!r}, seed={seed!r}."
                ) from exc
            checkpoint = _outputs(dependency, environment)["last_checkpoint"]
            inputs["backbone"] = {
                "source": source,
                "kind": "minerva_checkpoint",
                "checkpoint": checkpoint,
            }
            dependencies.append(
                {
                    "run_id": dependency.run_id,
                    "artifact": "last_checkpoint",
                    "path": str(checkpoint),
                }
            )
        elif source in BUILTIN_BACKBONES:
            backbone = resolve_dataset(source, "backbone", registry, environment)
            inputs["backbone"] = {
                "source": source,
                "kind": backbone.kind,
                "checkpoint": None,
            }
        else:
            raise ConfigError(
                f"Run {run.run_id} uses unsupported backbone source {source!r}."
            )
    else:
        raise ConfigError(f"Unsupported run stage {run.stage!r} in {run.run_id}.")

    return MaterializedRun(
        run=run,
        environment=environment,
        inputs=inputs,
        outputs=_outputs(run, environment),
        dependencies=tuple(dependencies),
        issues=tuple(issues),
        paths_checked=check_paths,
    )


def materialize_experiment(
    config: ExperimentConfig,
    environment: EnvironmentProfile,
    registry: DatasetRegistry,
    *,
    only: Mapping[str, Any] | None = None,
    check_paths: bool = False,
) -> list[MaterializedRun]:
    """Bind selected runs to an environment while retaining all dependencies."""

    all_runs = expand_experiment(config)
    selected_runs = expand_experiment(config, only=only)
    pretrain_index = _pretrain_index(all_runs)
    return [
        _materialize_one(
            run,
            environment=environment,
            registry=registry,
            pretrain_index=pretrain_index,
            check_paths=check_paths,
        )
        for run in selected_runs
    ]


def write_materialized_manifests(
    runs: Sequence[MaterializedRun],
    directory: str | Path,
) -> Path:
    """Write one environment-bound manifest for each run."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for materialized in runs:
        target = output_dir / f"{materialized.run.run_id}.yaml"
        with target.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(materialized.as_dict(), stream, sort_keys=False)
    return output_dir
