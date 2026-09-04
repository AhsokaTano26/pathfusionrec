import tempfile
import unittest
from pathlib import Path

from scripts.verify_baseline_delivery import (
    parse_actionpiece_log,
    parse_manifest,
    sample_stats,
    verify_manifest,
)


class BaselineVerificationTests(unittest.TestCase):
    def test_sample_stats_uses_sample_standard_deviation(self):
        result = sample_stats([1.0, 2.0, 3.0])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["sample_std"], 1.0)

    def test_parse_manifest_accepts_sha256sum_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SHA256SUMS.txt"
            path.write_text("a" * 64 + "  ./result.json\n", encoding="utf-8")
            self.assertEqual(parse_manifest(path), [("a" * 64, "result.json")])

    def test_verify_manifest_accepts_windows_checkout_line_endings(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "result.txt"
            payload.write_bytes(b"first\r\nsecond\r\n")
            expected = hashlib.sha256(b"first\nsecond\n").hexdigest()
            manifest = root / "SHA256SUMS.txt"
            manifest.write_text(f"{expected}  result.txt\n", encoding="utf-8")
            checks = []
            verify_manifest(checks, "test", manifest, root)
        self.assertEqual(checks[0]["status"], "PASS")
        self.assertIn("crlf_normalized=1", checks[0]["detail"])

    def test_parse_actionpiece_log_extracts_trace(self):
        content = "\n".join(
            [
                "INFO [Epoch 1] Train Loss: 2.5",
                "INFO [Epoch 1] Val Results: OrderedDict([('ndcg@10', 0.25), ('recall@10', 0.5)])",
                "INFO Best epoch: 1, Best val score: 0.25",
                "INFO Test Results: OrderedDict([('ndcg@10', 0.2), ('recall@10', 0.4)])",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(content, encoding="utf-8")
            result = parse_actionpiece_log(path)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs"][0]["train_loss"], 2.5)
        self.assertEqual(result["epochs"][0]["val_ndcg@10"], 0.25)
        self.assertEqual(result["test"]["recall@10"], 0.4)


if __name__ == "__main__":
    unittest.main()
