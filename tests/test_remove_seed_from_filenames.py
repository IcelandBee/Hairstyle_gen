import tempfile
import unittest
from pathlib import Path

from remove_seed_from_filenames import collect_seed_rename_ops, apply_rename_ops


class RemoveSeedFromFilenamesTest(unittest.TestCase):
    def test_collects_recursive_seed_suffix_renames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cand00 = root / "cand00"
            cand01 = root / "cand01"
            cand00.mkdir()
            cand01.mkdir()
            (cand00 / "portrait_seed12345.jpg").write_text("a", encoding="utf-8")
            (cand01 / "sample_seed67890.png").write_text("b", encoding="utf-8")
            (cand01 / "keep.jpg").write_text("c", encoding="utf-8")

            ops = collect_seed_rename_ops(root)
            targets = sorted(op.target.name for op in ops)

            self.assertEqual(targets, ["portrait.jpg", "sample.png"])

    def test_apply_renames_without_overwriting_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "portrait_seed12345.jpg").write_text("seed", encoding="utf-8")
            (root / "portrait.jpg").write_text("existing", encoding="utf-8")

            ops = collect_seed_rename_ops(root)
            results = apply_rename_ops(ops, apply=True)

            self.assertEqual(results.renamed, 0)
            self.assertEqual(results.skipped_existing, 1)
            self.assertTrue((root / "portrait_seed12345.jpg").exists())
            self.assertEqual((root / "portrait.jpg").read_text(encoding="utf-8"), "existing")

    def test_dry_run_does_not_rename_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "portrait_seed12345.jpg"
            target = root / "portrait.jpg"
            source.write_text("seed", encoding="utf-8")

            ops = collect_seed_rename_ops(root)
            results = apply_rename_ops(ops, apply=False)

            self.assertEqual(results.planned, 1)
            self.assertEqual(results.renamed, 0)
            self.assertTrue(source.exists())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
