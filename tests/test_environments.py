from pathlib import Path
import tempfile
import unittest

from seismic_byol.environments import (
    ConfigError,
    EnvironmentProfile,
    load_dataset_registry,
    load_environment,
    resolve_dataset,
    resolve_environment_name,
    validate_dataset_paths,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_dataset_registry(CONFIG_ROOT)

    def test_explicit_environment_wins_over_environment_variable(self):
        selected = resolve_environment_name(
            "container", environ={"SEISMIC_ENV": "sdumont"}
        )
        self.assertEqual(selected, "container")

    def test_environment_variable_is_used_without_explicit_value(self):
        selected = resolve_environment_name(
            environ={"SEISMIC_ENV": "sdumont"}
        )
        self.assertEqual(selected, "sdumont")

    def test_missing_environment_is_rejected(self):
        with self.assertRaisesRegex(ConfigError, "Execution environment is required"):
            resolve_environment_name(environ={})

    def test_all_profiles_resolve_every_local_dataset(self):
        for environment_name in ("local", "container", "sdumont"):
            with self.subTest(environment=environment_name):
                environment = load_environment(
                    environment_name, CONFIG_ROOT, environ={}
                )
                for dataset_name in ("f3_N", "seam_ai_N", "both_N", "a700"):
                    resolved = resolve_dataset(
                        dataset_name, "pretrain", self.registry, environment
                    )
                    self.assertIsNotNone(resolved.root)
                    self.assertIsNotNone(resolved.input_path)

    def test_container_separates_pretrain_images_from_finetune_root(self):
        environment = load_environment("container", CONFIG_ROOT, environ={})
        for dataset_name in ("f3_N", "seam_ai_N"):
            with self.subTest(dataset=dataset_name):
                pretrain = resolve_dataset(
                    dataset_name, "pretrain", self.registry, environment
                )
                finetune = resolve_dataset(
                    dataset_name, "finetune", self.registry, environment
                )
                self.assertEqual(pretrain.input_path, finetune.root / "images")
                self.assertEqual(finetune.input_path, finetune.root)

    def test_root_and_per_dataset_overrides_are_applied(self):
        environment = load_environment(
            "container",
            CONFIG_ROOT,
            environ={
                "SEISMIC_DATA_ROOT": "/tmp/shared",
                "SEISMIC_OUTPUT_ROOT": "/tmp/results",
                "SEISMIC_DATASET_A700_ROOT": "/private/a700",
            },
        )

        self.assertEqual(environment.data_root, Path("/tmp/shared"))
        self.assertEqual(environment.output_root, Path("/tmp/results"))
        self.assertEqual(
            environment.dataset_paths["f3_N"],
            Path("/tmp/shared/seismic/f3_segmentation_N"),
        )
        self.assertEqual(environment.dataset_paths["a700"], Path("/private/a700"))

    def test_path_validation_checks_role_specific_layout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for relative in (
                "images/train",
                "images/val",
                "images/test",
                "annotations/train",
                "annotations/val",
                "annotations/test",
            ):
                (root / relative).mkdir(parents=True)

            environment = EnvironmentProfile(
                name="test",
                data_root=root,
                output_root=root / "outputs",
                dataset_paths={"f3_N": root},
                source=root / "test.yaml",
            )
            dataset = resolve_dataset(
                "f3_N", "finetune", self.registry, environment
            )
            self.assertEqual(validate_dataset_paths(dataset, self.registry), ())

            (root / "annotations/test").rmdir()
            issues = validate_dataset_paths(dataset, self.registry)
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0].path, root / "annotations/test")


if __name__ == "__main__":
    unittest.main()
