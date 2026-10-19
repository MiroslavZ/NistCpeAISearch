import contextlib
import csv
import io
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from cpe_matcher import cli
from cpe_matcher.llm import ServiceError


def matched():
    return dict(cpe_part="a", cpe_vendor="vendor", cpe_product="product",
                status="matched", conclusion="Подтверждено.")


class CliTests(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger('cpe_matcher')
        self.addCleanup(lambda: logger.handlers.clear())
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "input.csv"
        self.output = self.root / "output.csv"
        self.state = self.root / "state.sqlite3"
        self.env_file = self.root / "settings.env"
        self.source.write_text(
            "input_vendor,input_product,input_version,previous_mapping\n"
            "Vendor,First,001,wrong\nVendor,Second,2,wrong\nVendor,First,001,wrong\n",
            encoding="utf-8-sig",
        )
        self.env_file.write_text("# Test configuration\n", encoding="utf-8")
        self.env = patch.dict(os.environ, {
            "LLM_API_KEY": uuid.uuid4().hex,
            "LLM_BASE_URL": "https://example.invalid",
            "DEFAULT_MODEL": "offline-model",
            "NIST_API_KEY": uuid.uuid4().hex,
        }, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.arguments = [str(self.source), "--output", str(self.output),
                          "--state", str(self.state), "--env-file", str(self.env_file)]
        self.stderr = io.StringIO()

    def run_cli(self, extra=()):
        # These fake processors have no awaits. Drive them directly so the test
        # needs no Windows event-loop socketpair (blocked by some sandboxes).
        def run_offline(coroutine):
            try:
                coroutine.send(None)
            except StopIteration as completed:
                return completed.value
            finally:
                coroutine.close()
            raise AssertionError("The offline processor unexpectedly suspended")

        with contextlib.redirect_stderr(self.stderr), patch.object(cli.asyncio, "run", run_offline):
            return cli.main([*self.arguments, *extra])

    def output_rows(self):
        with self.output.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_service_failure_exports_progress_and_resume_skips_saved_rows(self):
        async def interrupted(rows, state, settings, verbose):
            state.save(rows[0], matched())
            raise ServiceError("offline failure")

        with patch.object(cli, "process", side_effect=interrupted):
            self.assertEqual(self.run_cli(), 1)
        partial = self.output_rows()
        self.assertEqual([row["status"] for row in partial],
                         ["matched", "search_incomplete", "matched"])
        processed = []

        async def resumed(rows, state, settings, verbose):
            for row in rows:
                if state.get(row) is None:
                    processed.append(row.input_product)
                    state.save(row, matched())

        with patch.object(cli, "process", side_effect=resumed):
            self.assertEqual(self.run_cli(["--resume"]), 0)
        self.assertEqual(processed, ["Second"])
        completed = self.output_rows()
        self.assertEqual([row["input_product"] for row in completed], ["First", "Second", "First"])
        self.assertTrue(all(row["status"] == "matched" for row in completed))
        self.assertNotIn("previous_mapping", completed[0])
        self.assertEqual(completed[0]["input_version"], "001")

    def test_resume_rejects_changed_settings_or_input_bytes_without_replacing_report(self):
        async def complete(rows, state, settings, verbose):
            for row in rows:
                state.save(row, matched())

        with patch.object(cli, "process", side_effect=complete):
            self.assertEqual(self.run_cli(), 0)
        original_output = self.output.read_bytes()
        with patch.object(cli, "process") as process:
            with patch.dict(os.environ, {"DEFAULT_MODEL": "different-model"}):
                self.assertEqual(self.run_cli(["--resume"]), 1)
            self.source.write_bytes(self.source.read_bytes() + b"\n")
            self.assertEqual(self.run_cli(["--resume"]), 1)
            process.assert_not_called()
        self.assertEqual(self.output.read_bytes(), original_output)

    def test_colliding_paths_preserve_original_files(self):
        for option, target in [("--output", self.source), ("--state", self.source),
                               ("--output", self.env_file), ("--state", self.env_file),
                               ("--state", self.output)]:
            with self.subTest(option=option, target=target):
                before_source = self.source.read_bytes()
                before_env = self.env_file.read_bytes()
                with patch.object(cli, "process") as process:
                    self.assertEqual(self.run_cli([option, str(target)]), 1)
                    process.assert_not_called()
                self.assertEqual(self.source.read_bytes(), before_source)
                self.assertEqual(self.env_file.read_bytes(), before_env)
                self.assertFalse(self.state.exists())
                self.assertFalse(self.output.exists())

    def test_retry_status_requires_resume(self):
        with patch.object(cli, "process") as process:
            self.assertEqual(self.run_cli(["--retry-status", "not_found"]), 1)
            process.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_keyboard_interrupt_exports_progress(self):
        async def interrupted(rows, state, settings, verbose):
            state.save(rows[0], matched())
            raise KeyboardInterrupt

        with patch.object(cli, "process", side_effect=interrupted):
            self.assertEqual(self.run_cli(), 130)
        self.assertEqual(self.output_rows()[1]["status"], "search_incomplete")


if __name__ == "__main__":
    unittest.main()
