from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from seismic_byol.cli import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = REPOSITORY_ROOT / "configs/experiments/paper_main.yaml"


class CliTests(unittest.TestCase):
    def _run(self, *arguments):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(list(arguments))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_validate_reports_matrix_and_run_counts(self):
        status, stdout, stderr = self._run("validate", str(PAPER_CONFIG))

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("2 matrices, 3660 runs", stdout)

    def test_plan_filters_and_limits_table(self):
        status, stdout, stderr = self._run(
            "plan",
            str(PAPER_CONFIG),
            "--only",
            "matrix=downstream",
            "--only",
            "pretrain=both_N",
            "--only",
            "cap=32",
            "--limit",
            "3",
        )

        self.assertEqual(status, 0)
        self.assertIn("both_N", stdout)
        self.assertIn("Showing 3 of 40 resolved runs.", stderr)

    def test_plan_writes_all_filtered_manifests_despite_display_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            status, _, stderr = self._run(
                "plan",
                str(PAPER_CONFIG),
                "--only",
                "matrix=byol-pretrain",
                "--only",
                "pretrain=a700",
                "--limit",
                "1",
                "--write-manifests",
                temporary_directory,
            )

            self.assertEqual(status, 0)
            self.assertEqual(len(list(Path(temporary_directory).glob("*.yaml"))), 5)
            self.assertIn("Wrote 5 manifests", stderr)

    def test_plan_rejects_unknown_filters(self):
        status, _, stderr = self._run(
            "plan",
            str(PAPER_CONFIG),
            "--only",
            "does_not_exist=value",
        )

        self.assertEqual(status, 2)
        self.assertIn("Unknown filter axes", stderr)


if __name__ == "__main__":
    unittest.main()
