from pathlib import Path
import tempfile
import unittest

import yaml

from seismic_byol.experiments import (
    ConfigError,
    expand_experiment,
    load_experiment,
    write_run_manifests,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = REPOSITORY_ROOT / "configs/experiments/paper_main.yaml"


class PaperMatrixTests(unittest.TestCase):
    def test_paper_matrix_expands_every_expected_combination(self):
        config = load_experiment(PAPER_CONFIG)
        runs = expand_experiment(config)

        by_stage = {
            stage: [run for run in runs if run.stage == stage]
            for stage in {run.stage for run in runs}
        }
        self.assertEqual(len(by_stage["pretrain"]), 20)
        self.assertEqual(len(by_stage["finetune"]), 3640)
        self.assertEqual(len(runs), 3660)
        self.assertEqual(len({run.run_id for run in runs}), len(runs))

    def test_scratch_frozen_combinations_are_excluded(self):
        runs = expand_experiment(load_experiment(PAPER_CONFIG))

        invalid = [
            run
            for run in runs
            if run.values.get("pretrain") == "scratch"
            and run.values.get("backbone_mode") == "frozen"
        ]
        self.assertEqual(invalid, [])

    def test_filter_accepts_typed_values(self):
        config = load_experiment(PAPER_CONFIG)
        runs = expand_experiment(
            config,
            only={
                "matrix": "downstream",
                "pretrain": "both_N",
                "finetune": "f3_N",
                "cap": 32,
                "seed": 2,
            },
        )

        self.assertEqual(len(runs), 4)
        self.assertEqual(
            {(run.values["head"], run.values["backbone_mode"]) for run in runs},
            {
                ("linear", "frozen"),
                ("linear", "unfrozen"),
                ("aspp", "frozen"),
                ("aspp", "unfrozen"),
            },
        )

    def test_run_ids_are_stable_and_include_minerva_revision(self):
        first = expand_experiment(load_experiment(PAPER_CONFIG))
        second = expand_experiment(load_experiment(PAPER_CONFIG))

        self.assertEqual(
            [run.run_id for run in first],
            [run.run_id for run in second],
        )
        self.assertEqual(
            first[0].metadata["minerva"]["revision"],
            "f23918a58ea25ee4c016860a348d592e4ac5a04d",
        )

    def test_manifests_are_written_for_all_runs(self):
        config = load_experiment(PAPER_CONFIG)
        runs = expand_experiment(
            config,
            only={"matrix": "byol-pretrain", "pretrain": "both_N"},
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = write_run_manifests(runs, temporary_directory)
            manifests = sorted(output.glob("*.yaml"))

            self.assertEqual(len(manifests), 5)
            document = yaml.safe_load(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(document["run"]["stage"], "pretrain")
            self.assertEqual(document["run"]["values"]["pretrain"], "both_N")
            self.assertIn("minerva", document["metadata"])


class ValidationTests(unittest.TestCase):
    def _write_config(self, document):
        temporary_directory = tempfile.TemporaryDirectory()
        path = Path(temporary_directory.name) / "experiment.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        self.addCleanup(temporary_directory.cleanup)
        return path

    def test_include_adds_a_complete_combination(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "experiment": {"name": "tiny"},
                "matrices": [
                    {
                        "name": "train",
                        "stage": "finetune",
                        "axes": {"dataset": ["f3"], "seed": [0]},
                        "exclude": [{"dataset": "f3"}],
                        "include": [{"dataset": "parihaka", "seed": 1}],
                    }
                ],
            }
        )

        runs = expand_experiment(load_experiment(path))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].values, {"dataset": "parihaka", "seed": 1})

    def test_include_requires_every_axis(self):
        path = self._write_config(
            {
                "schema_version": 1,
                "experiment": {"name": "tiny"},
                "matrices": [
                    {
                        "name": "train",
                        "stage": "finetune",
                        "axes": {"dataset": ["f3"], "seed": [0]},
                        "include": [{"dataset": "parihaka"}],
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ConfigError, "missing: seed"):
            load_experiment(path)

    def test_unknown_filter_is_rejected(self):
        config = load_experiment(PAPER_CONFIG)
        with self.assertRaisesRegex(ConfigError, "Unknown filter axes"):
            expand_experiment(config, only={"unknown": "value"})


if __name__ == "__main__":
    unittest.main()
