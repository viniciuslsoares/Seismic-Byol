from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import tifffile
    from PIL import Image
    from minerva.models.ssl.byol import BYOL
    from minerva.pipelines.experiment import Experiment

    from seismic_byol.runtime import build_runtime
    from seismic_byol.runtime_data import A700Dataset

    RUNTIME_AVAILABLE = True
except ImportError:
    RUNTIME_AVAILABLE = False

from seismic_byol.environments import (
    EnvironmentProfile,
    load_dataset_registry,
)
from seismic_byol.experiments import load_experiment
from seismic_byol.materialize import materialize_experiment


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
PAPER_CONFIG = CONFIG_ROOT / "experiments/paper_main.yaml"


@unittest.skipUnless(RUNTIME_AVAILABLE, "Minerva runtime is not installed")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_experiment(PAPER_CONFIG)
        self.registry = load_dataset_registry(CONFIG_ROOT)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.addCleanup(self.temporary_directory.cleanup)

    def _environment(self, **datasets):
        return EnvironmentProfile(
            name="test",
            data_root=self.root,
            output_root=self.root / "outputs",
            dataset_paths=datasets,
            source=self.root / "test.yaml",
        )

    def _write_tiff(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            path,
            np.zeros((16, 16, 3), dtype=np.float32),
            photometric="rgb",
        )

    def _write_mask(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(path)

    def test_builds_real_minerva_pretrain_pipeline(self):
        dataset_root = self.root / "f3_N"
        self._write_tiff(dataset_root / "images/train/il_0.tiff")
        self._write_tiff(dataset_root / "images/val/il_1.tiff")
        self._write_tiff(dataset_root / "images/test/must_not_be_used.tiff")
        environment = self._environment(f3_N=dataset_root)
        materialized = materialize_experiment(
            self.config,
            environment,
            self.registry,
            only={
                "matrix": "byol-pretrain",
                "pretrain": "f3_N",
                "seed": 0,
            },
            check_paths=True,
        )[0]

        runtime = build_runtime(
            materialized,
            self.registry,
            accelerator="cpu",
            num_workers=0,
        )

        self.assertIsInstance(runtime.model, BYOL)
        self.assertEqual(runtime.task, "fit")
        self.assertEqual(len(runtime.data_module.train_dataset), 2)
        first_view, second_view = runtime.data_module.train_dataset[0]
        self.assertEqual(first_view.shape, (3, 256, 256))
        self.assertEqual(second_view.shape, (3, 256, 256))
        self.assertEqual(
            runtime.pipeline.log_dir,
            materialized.outputs["run_root"],
        )

    def test_builds_real_minerva_downstream_experiment(self):
        dataset_root = self.root / "f3_N"
        for stem in ("il_0", "xl_0"):
            self._write_tiff(dataset_root / f"images/train/{stem}.tiff")
            self._write_mask(dataset_root / f"annotations/train/{stem}.png")
        for partition in ("val", "test"):
            self._write_tiff(dataset_root / f"images/{partition}/il_0.tiff")
            self._write_mask(dataset_root / f"annotations/{partition}/il_0.png")

        environment = self._environment(f3_N=dataset_root)
        materialized = materialize_experiment(
            self.config,
            environment,
            self.registry,
            only={
                "matrix": "downstream",
                "pretrain": "scratch",
                "finetune": "f3_N",
                "head": "linear",
                "backbone_mode": "unfrozen",
                "cap": 2,
                "seed": 0,
            },
            check_paths=True,
        )[0]

        runtime = build_runtime(
            materialized,
            self.registry,
            accelerator="cpu",
            num_workers=0,
        )

        self.assertIsInstance(runtime.pipeline, Experiment)
        self.assertIsNone(runtime.model)
        self.assertEqual(runtime.task, "fit-evaluate")
        self.assertEqual(len(runtime.data_module.train_dataset), 2)
        self.assertEqual(len(runtime.data_module.val_dataset), 1)
        self.assertEqual(len(runtime.data_module.test_dataset), 1)
        image, mask = runtime.data_module.train_dataset[0]
        self.assertEqual(image.shape, (3, 256, 704))
        self.assertEqual(mask.shape, (1, 256, 704))
        self.assertEqual(runtime.pipeline.log_dir, materialized.outputs["run_root"])
        self.assertTrue(runtime.pipeline._trainer_parameters()["deterministic"])
        self.assertFalse(runtime.pipeline._trainer_parameters()["benchmark"])

    def test_a700_split_is_deterministic(self):
        dataset_root = self.root / "a700"
        for orientation in ("iline", "xline"):
            directory = dataset_root / orientation
            directory.mkdir(parents=True)
            for index in range(100):
                np.save(
                    directory / f"{index:03d}.npy",
                    np.arange(16, dtype=np.float32).reshape(4, 4),
                )

        train = A700Dataset(dataset_root, "train", lambda value: value)
        validation = A700Dataset(dataset_root, "val", lambda value: value)

        self.assertEqual(len(train), 180)
        self.assertEqual(len(validation), 20)
        np.testing.assert_allclose(float(train[0].mean()), 0.0, atol=1e-6)
        np.testing.assert_allclose(float(train[0].std()), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
