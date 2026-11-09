"""
EPL442 Assignment 3: Kohonen Self-Organizing Map (SOM)
Author: <Name Surname>, Student ID: <AM>

Tests for make_zip.py (in the repository root): the archive has the layout
the assignment asks for (code/, README.txt, report.pdf), the author
placeholders are filled in only inside the zip, caches and per-run
experiment folders are left out, and a missing file is reported.
"""

import importlib.util
import os
import tempfile
import unittest
import zipfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("make_zip", os.path.join(REPO_DIR, "make_zip.py"))
make_zip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_zip)

HEADER = '"""\nAuthor: <Name Surname>, Student ID: <AM>\n"""\n'


def make_fake_repo(root, with_report=True):
    """Create every file make_zip needs, plus some it must leave out."""
    code = os.path.join(root, "code")
    for sub in ("tests", "plots", "experiments/seed_1", "__pycache__", "tests/__pycache__"):
        os.makedirs(os.path.join(code, sub), exist_ok=True)
    for name in make_zip.REQUIRED_CODE_FILES:
        with open(os.path.join(code, name), "w", encoding="utf-8") as f:
            f.write(HEADER if name.endswith(".py") else "data\n")
    files = {
        "tests/__init__.py": "", "tests/test_som.py": HEADER, "tests/test_make_zip.py": HEADER,
        "plots/error_curve.png": "png", "experiments/summary.txt": "table\n",
        "experiments/seed_1/results.txt": "1 0.3 0.3\n",
        "__pycache__/som.cpython-313.pyc": "x", "tests/__pycache__/t.pyc": "x",
        "baseline_run.log": "log\n",
    }
    for name, text in files.items():
        with open(os.path.join(code, name), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(root, "README.txt"), "w", encoding="utf-8") as f:
        f.write("readme\n")
    if with_report:
        with open(os.path.join(root, "report.pdf"), "wb") as f:
            f.write(b"%PDF-1.4\n")


class MakeZipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def build(self, **kwargs):
        path, warnings = make_zip.build_zip("123456", root=self.root, **kwargs)
        with zipfile.ZipFile(path) as zf:
            return path, warnings, zf.namelist(), {n: zf.read(n) for n in zf.namelist()}

    def test_layout_and_name(self):
        make_fake_repo(self.root)
        path, warnings, names, _ = self.build(name="Name Surname")
        self.assertEqual(os.path.basename(path), "EPL442_ex3_123456.zip")
        self.assertEqual(warnings, [])
        self.assertIn("README.txt", names)
        self.assertIn("report.pdf", names)
        for f in make_zip.REQUIRED_CODE_FILES:
            self.assertIn("code/" + f, names)
        self.assertIn("code/tests/test_som.py", names)
        self.assertNotIn("code/tests/test_make_zip.py", names)
        self.assertIn("code/plots/error_curve.png", names)
        self.assertIn("code/experiments/summary.txt", names)
        # Every entry is at the top level or under code/.
        self.assertTrue(all(n in ("README.txt", "report.pdf") or n.startswith("code/")
                            for n in names))

    def test_leaves_out_caches_logs_and_run_folders(self):
        make_fake_repo(self.root)
        _, _, names, _ = self.build(name="N S")
        self.assertFalse([n for n in names if "__pycache__" in n or n.endswith(".pyc")])
        self.assertNotIn("code/experiments/seed_1/results.txt", names)
        self.assertNotIn("code/baseline_run.log", names)

    def test_placeholders_filled_in_zip_only(self):
        make_fake_repo(self.root)
        _, _, _, content = self.build(name="Andreas Test")
        self.assertIn(b"Author: Andreas Test, Student ID: 123456", content["code/main.py"])
        with open(os.path.join(self.root, "code", "main.py"), encoding="utf-8") as f:
            self.assertIn("<Name Surname>", f.read())

    def test_warns_without_name(self):
        make_fake_repo(self.root)
        _, warnings, _, content = self.build()
        self.assertIn(b"Author: <Name Surname>, Student ID: 123456", content["code/main.py"])
        self.assertTrue(any("placeholders" in w for w in warnings))

    def test_warns_without_report(self):
        make_fake_repo(self.root, with_report=False)
        _, warnings, names, _ = self.build(name="N S")
        self.assertNotIn("report.pdf", names)
        self.assertTrue(any("report.pdf" in w for w in warnings))

    def test_missing_results_is_an_error(self):
        make_fake_repo(self.root)
        os.remove(os.path.join(self.root, "code", "results.txt"))
        with self.assertRaisesRegex(make_zip.PackageError, "results.txt"):
            make_zip.build_zip("1", root=self.root)

    def test_real_repository_is_complete(self):
        # The repository itself has every required file (results.txt and
        # clustering.txt of the baseline run are committed).
        files = make_zip.collect_files(REPO_DIR)
        self.assertIn("code/results.txt", [arc for _, arc in files])


if __name__ == "__main__":
    unittest.main()
