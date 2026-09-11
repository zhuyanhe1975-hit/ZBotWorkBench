"""Lightweight dependency discovery: never import torch or initialize a GPU."""
import importlib.metadata
import importlib.util
import json
import ast
from pathlib import Path


def cuda_build_from_source(source, version):
    """Inspect the packaged version constant without importing/executing Torch."""
    try:
        for node in ast.parse(source).body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(target, ast.Name) and target.id == "cuda" for target in targets):
                value = ast.literal_eval(node.value)
                if value is None or isinstance(value, str):
                    return bool(value)
    except (SyntaxError, ValueError, TypeError):
        pass
    if "+cpu" in version:
        return False
    if "+cu" in version:
        return True
    return None


def probe():
    result = {"available": False}
    try:
        for package, key in (("mjlab", "mjlabVersion"), ("torch", "torchVersion"), ("warp-lang", "warpVersion")):
            result[key] = importlib.metadata.version(package)
        for module in ("mjlab", "torch", "warp", "mujoco", "rsl_rl"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"Missing runtime module: {module}")
        source = ""
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec.origin:
            try:
                source = Path(torch_spec.origin).with_name("version.py").read_text()
            except OSError:
                pass
        cuda_build = cuda_build_from_source(source, result["torchVersion"])
        if cuda_build is not None:
            result["cudaBuild"] = cuda_build
        if result["mjlabVersion"] != "1.3.0" or importlib.metadata.version("rsl-rl-lib") != "5.2.0":
            raise RuntimeError("Training requires mjlab==1.3.0 and rsl-rl-lib==5.2.0")
        result["available"] = True
    except (ImportError, RuntimeError) as error:
        result["error"] = str(error)
    return result


if __name__ == "__main__":
    print(json.dumps(probe()))
