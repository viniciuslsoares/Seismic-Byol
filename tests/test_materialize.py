from pathlib import Path
import tempfile
import unittest

import yaml

from seismic_byol.environments import (
    EnvironmentProfile,
    load_dataset_registry,
    load_environment,
)
from seismic_byol.experiments import expand_experiment, load_experiment
from seismic_byol.materialize import (
    materialize_experiment,
    write_materialized_manifests,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
PAPER_CONFIG = CONFIG_ROOT / "experiments/paper_main.yaml"


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment(PAPER_CONFIG)
        self.registry = load_dataset_registry(CONFIG_ROOT)
        self.environment = load_environment("container", CONFIG_ROOT, environ={})

    def test_every_run_can_be_materialized_without_filesystem_checks(self):
        runs = materialize_experiment(
            self.config,
            self.environment,
            self.registry,
        )

        self.assertEqual(len(runs), 3660)
        self.assertTrue(all(not run.paths_checked for run in runs))

    def test_pretrain_run_resolves_minerva_dataset_input(self):
        runs = materialize_experiment(
            self.config,
            self.environment,
            self.registry,
            only={"matrix": "byol-pretrain", "pretrain": "both_N", "seed": 2},
        )

        self.assertEqual(len(runs), 1)
        dataset = runs[0].inputs["dataset"]
        self.assertEqual(dataset.input_path, Path("/workspaces/shared_data/seismic/both_N/images"))
        self.assertEqual(runs[0].dependencies, ())

    def test_finetune_run_links_to_deterministic_pretrain_checkpoint(self):
        runs = materialize_experiment(
            self.config,
            self.environment,
            self.registry,
            only={
                "matrix": "downstream",
                "pretrain": "both_N",
                "finetune": "f3_N",
                "head": "linear",
                "backbone_mode": "frozen",
                "cap": 32,
                "seed": 2,
            },
        )

        self.assertEqual(len(runs), 1)
        materialized = runs[0]
        dependency = materialized.dependencies[0]
        pretrain_run = expand_experiment(
            self.config,
            only={"matrix": "byol-pretrain", "pretrain": "both_N", "seed": 2},
        )[0]
        expected = (
            self.environment.output_root
            / self.config.name
            / pretrain_run.run_id
            / "checkpoints/last.ckpt"
        )
        self.assertEqual(dependency["run_id"], pretrain_run.run_id)
        self.assertEqual(materialized.inputs["backbone"]["checkpoint"], expected)

    def test_builtin_and_scratch_backbones_have_no_checkpoint_dependency(self):
        for source, expected_kind in (
            ("imagenet", "builtin_backbone"),
            ("coco", "builtin_backbone"),
            ("scratch", "random_initialization"),
        ):
            with self.subTest(source=source):
                run = materialize_experiment(
                    self.config,
                    self.environment,
                    self.registry,
                    only={
                        "matrix": "downstream",
                        "pretrain": source,
                        "finetune": "f3_N",
                        "head": "linear",
                        "backbone_mode": "unfrozen",
                        "cap": "full",
                        "seed": 0,
                    },
                )[0]
                self.assertEqual(run.inputs["backbone"]["kind"], expected_kind)
                self.assertIsNone(run.inputs["backbone"]["checkpoint"])
                self.assertEqual(run.dependencies, ())

    def test_materialization_does_not_change_scientific_run_id(self):
        resolved = expand_experiment(
            self.config,
            only={"matrix": "byol-pretrain", "pretrain": "f3_N", "seed": 0},
        )[0]
        materialized = materialize_experiment(
            self.config,
            self.environment,
            self.registry,
            only={"matrix": "byol-pretrain", "pretrain": "f3_N", "seed": 0},
        )[0]

        self.assertEqual(materialized.run.run_id, resolved.run_id)

    def test_checked_materialization_records_missing_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = EnvironmentProfile(
                name="test",
                data_root=root,
                output_root=root / "outputs",
                dataset_paths={
                    "f3_N": root / "f3_N",
                    "seam_ai_N": root / "seam_ai_N",
                    "both_N": root / "both_N",
                    "a700": root / "a700",
                },
                source=root / "test.yaml",
            )
            run = materialize_experiment(
                self.config,
                environment,
                self.registry,
                only={"matrix": "byol-pretrain", "pretrain": "both_N", "seed": 0},
                check_paths=True,
            )[0]

            self.assertTrue(run.paths_checked)
            self.assertEqual(len(run.issues), 2)

    def test_materialized_manifest_contains_environment_and_dependency(self):
        runs = materialize_experiment(
            self.config,
            self.environment,
            self.registry,
            only={
                "matrix": "downstream",
                "pretrain": "a700",
                "finetune": "seam_ai_N",
                "head": "aspp",
                "backbone_mode": "unfrozen",
                "cap": 16,
                "seed": 4,
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = write_materialized_manifests(runs, temporary_directory)
            manifest = yaml.safe_load(
                next(output.glob("*.yaml")).read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["runtime"]["environment"]["name"], "container")
        self.assertEqual(len(manifest["runtime"]["dependencies"]), 1)
        self.assertEqual(
            manifest["runtime"]["inputs"]["dataset"]["root"],
            "/workspaces/shared_data/seam_ai_datasets/seam_ai_N",
        )


if __name__ == "__main__":
    unittest.main()
