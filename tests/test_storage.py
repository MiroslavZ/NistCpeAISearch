import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from cpe_matcher.storage import InputRow, RunLock, State, export_report, read_report


def result(status="matched"):
    return dict(cpe_part="a", cpe_vendor="vendor", cpe_product="product",
                status=status, conclusion="Подтверждено действующей записью.")


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.db = self.root / "state.sqlite"

    def test_input_preserves_strings_ignores_prior_mappings(self):
        source = self.root / "input.csv"
        source.write_text(
            'input_vendor,input_product,input_version,cpe_vendor\n'
            '  Производитель  ,"Продукт, ""редакция""",001.020,wrong\n'
            ',Продукт,,wrong\n', encoding="utf-8-sig"
        )
        self.assertEqual(read_report(source), [
            InputRow("  Производитель  ", 'Продукт, "редакция"', "001.020"),
            InputRow("", "Продукт", ""),
        ])

    def test_rejects_missing_duplicate_and_short_input(self):
        source = self.root / "invalid.csv"
        for content in (
            "input_vendor,input_product\nv,p\n",
            "input_vendor,input_product,input_version,input_vendor\nv,p,1,v\n",
            "input_vendor,input_product,input_version\nv,p\n",
        ):
            source.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_report(source)

    def test_resume_binds_both_fingerprints_and_retains_completed_results(self):
        row = InputRow("v", "p", "1")
        other_version = InputRow("v", "p", "2")
        with State(self.db, "bytes-sha256", "configuration-sha256") as state:
            state.save(row, result())
            state.save(other_version, result("search_incomplete"))
        with self.assertRaises(ValueError):
            with State(self.db, "bytes-sha256", "configuration-sha256"):
                pass
        for input_digest, config_digest in [("changed", "configuration-sha256"),
                                             ("bytes-sha256", "changed")]:
            with self.assertRaises(ValueError):
                with State(self.db, input_digest, config_digest, resume=True):
                    pass
        with State(self.db, "bytes-sha256", "configuration-sha256", resume=True) as state:
            state.reset_statuses({"search_incomplete"})
            self.assertEqual(state.get(row), result())
            self.assertIsNone(state.get(other_version))

    def test_7000_rows_interruption_resume_duplicate_order(self):
        rows = [InputRow("Производитель", f"Продукт {index % 3500}", "001.0")
                for index in range(7000)]
        digest = hashlib.sha256(repr(rows).encode()).hexdigest()
        output = self.root / "result.csv"
        with State(self.db, digest, "config") as state:
            for row in rows[:1000]:
                state.save(row, result())
            export_report(output, rows, state)
        with output.open(encoding="utf-8-sig", newline="") as stream:
            partial = list(csv.DictReader(stream))
        self.assertEqual(len(partial), 7000)
        self.assertEqual(partial[1000]["status"], "search_incomplete")
        self.assertEqual(partial[3500]["status"], "matched")
        processed = 0
        with State(self.db, digest, "config", resume=True) as state:
            for row in rows:
                if state.get(row) is None:
                    state.save(row, result())
                    processed += 1
            export_report(output, rows, state)
        self.assertEqual(processed, 2500)
        self.assertEqual(read_report(output), rows)
        with output.open(encoding="utf-8-sig", newline="") as stream:
            complete = list(csv.DictReader(stream))
        self.assertTrue(all(item["status"] == "matched" for item in complete))

    def test_export_failure_keeps_previous_file(self):
        output = self.root / "output.csv"
        output.write_bytes(b"previous report")
        with State(self.db, "input", "config") as state:
            with patch("cpe_matcher.storage.os.replace", side_effect=OSError("failed")):
                with self.assertRaises(OSError):
                    export_report(output, [InputRow("v", "p", "1")], state)
        self.assertEqual(output.read_bytes(), b"previous report")
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_process_lock_blocks_any_shared_key_and_releases_after_exit(self):
        key = uuid.uuid4().hex
        script = (
            "from cpe_matcher.storage import RunLock; import sys\n"
            "try:\n"
            " with RunLock([sys.argv[1]]): pass\n"
            "except RuntimeError:\n"
            " sys.exit(23)\n"
        )
        with RunLock([uuid.uuid4().hex, key]):
            blocked = subprocess.run([sys.executable, "-c", script, key],
                                     capture_output=True, timeout=15)
        self.assertEqual(blocked.returncode, 23, blocked.stderr.decode())
        unblocked = subprocess.run([sys.executable, "-c", script, key],
                                   capture_output=True, timeout=15)
        self.assertEqual(unblocked.returncode, 0, unblocked.stderr.decode())

    def test_failed_multikey_lock_releases_previously_acquired_keys(self):
        keys = sorted([uuid.uuid4().hex, uuid.uuid4().hex],
                      key=lambda key: hashlib.sha256(key.encode()).hexdigest())
        with RunLock([keys[1]]):
            with self.assertRaises(RuntimeError):
                with RunLock(keys):
                    pass
            with RunLock([keys[0]]):
                pass


if __name__ == "__main__":
    unittest.main()
