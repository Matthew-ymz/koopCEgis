import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NOTEBOOK_SNIPPETS = (
    "data_generators/",
    "data_generators\\\\",
    "../data_generators/",
    "E:/code/pykoop",
    "e:/code/pykoop",
    "E:\\\\code\\\\pykoop",
    "e:\\\\code\\\\pykoop",
    "mypykoop\\\\pykoop",
)


def load_module(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem + "_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ImportRegressionTests(unittest.TestCase):
    def test_air_quality_sweep_imports(self):
        load_module("exp/air_quality/run_air_macro_sweep.py")

    def test_kuramoto_sweep_imports(self):
        load_module("exp/kuramoto/run_whitened_macro_sweep.py")

    def test_rulkov_sweep_imports(self):
        load_module("exp/rulkov_map/run_map_analysis_sweep.py")

    def test_notebooks_do_not_reference_old_paths(self):
        offenders = []
        for path in sorted((REPO_ROOT / "exp").rglob("*.ipynb")):
            text = path.read_text(encoding="utf-8")
            hits = [snippet for snippet in FORBIDDEN_NOTEBOOK_SNIPPETS if snippet in text]
            if hits:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {', '.join(hits)}")
        self.assertEqual([], offenders, "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
