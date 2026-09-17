"""Repository hygiene: only text files with allowed names, size limits for the CSV model."""
import fnmatch
import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_SUFFIXES = {".py", ".csv", ".md", ".txt"}
ALLOWED_NAMES = {"LICENSE", "NOTICE", ".gitignore", ".gitattributes"}
MAX_FILE_BYTES = 50_000_000
MAX_MODEL_BYTES = 2_000_000_000


def ignore_patterns():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def is_ignored(relative, patterns):
    parts = relative.split("/")
    for pattern in patterns:
        if pattern.endswith("/"):  # a directory pattern matches any directory on the path
            if any(fnmatch.fnmatch(part, pattern[:-1]) for part in parts[:-1]):
                return True
        elif fnmatch.fnmatch(parts[-1], pattern):
            return True
    return False


def project_files():
    """Every file git would track: `git ls-files` in a git repository, else all files minus .gitignore matches."""
    if (ROOT / ".git").exists() and shutil.which("git"):
        listing = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                 cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8")
        return sorted(f for f in listing.split(chr(0)) if f and (ROOT / f).exists())
    patterns = ignore_patterns()
    files = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in names:
            relative = Path(folder, name).relative_to(ROOT).as_posix()
            if not is_ignored(relative, patterns):
                files.append(relative)
    return sorted(files)


class RepoHygiene(unittest.TestCase):
    def test_only_allowed_file_names(self):
        bad = [f for f in project_files()
               if Path(f).suffix not in ALLOWED_SUFFIXES and Path(f).name not in ALLOWED_NAMES]
        self.assertEqual(bad, [])

    def test_every_file_is_utf8_text(self):
        bad = []
        for f in project_files():
            data = (ROOT / f).read_bytes()
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                bad.append(f)
                continue
            if b"\x00" in data:
                bad.append(f)
        self.assertEqual(bad, [])

    def test_file_size_limits(self):
        sizes = {f: (ROOT / f).stat().st_size for f in project_files()}
        self.assertEqual([f for f, size in sizes.items() if size >= MAX_FILE_BYTES], [])
        model_total = sum(size for f, size in sizes.items() if f.split("/")[0].startswith("model_csv"))
        self.assertGreater(model_total, 0)
        self.assertLessEqual(model_total, MAX_MODEL_BYTES)

    def test_gitignore_keeps_binaries_out(self):
        patterns = ignore_patterns()
        for path in ["downloads/model.safetensors", ".venv-ref/Lib/x.py", "tests/__pycache__/a.pyc", "x.pyc",
                     "model_csv/weights/a.csv"]:
            self.assertTrue(is_ignored(path, patterns), path)


if __name__ == "__main__":
    unittest.main()
