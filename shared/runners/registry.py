"""Auto-discovers method adapters under `methods/Bx_xxx/adapter.py`.

`run_phm2010.py` never hardcodes a method list — it asks this module. Each
`methods/Bx_xxx/adapter.py` must expose a module-level `ADAPTER_CLASS`
(a `shared.runners.method_adapter.MethodAdapter` subclass) with a
`method_id` class attribute like `"B1"`. This lets each method's fork/PR be
purely additive — dropping in a new `methods/Bx_xxx/` directory is enough,
nothing else needs editing.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS_DIR = REPO_ROOT / "methods"


def discover_method_dirs() -> dict[str, Path]:
    """Returns {method_id: method_dir} for every methods/Bx_xxx/ that has an
    adapter.py, WITHOUT importing them (cheap, safe to call for listing)."""
    out: dict[str, Path] = {}
    if not METHODS_DIR.exists():
        return out
    for d in sorted(METHODS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        adapter_py = d / "adapter.py"
        if not adapter_py.exists():
            continue
        # method dir names are expected as "Bx_..." -- derive method_id from the
        # leading token, but the authoritative id still comes from the loaded
        # class's own `method_id` attribute (checked in load_adapter_class).
        method_id = d.name.split("_")[0]
        out[method_id] = d
    return out


_CLASS_CACHE: dict[str, type] = {}


def load_adapter_class(method_id: str):
    """Imports methods/Bx_xxx/adapter.py and returns its ADAPTER_CLASS.
    Raises KeyError if method_id is unknown, ImportError/AttributeError if the
    method's adapter.py is malformed."""
    if method_id in _CLASS_CACHE:
        return _CLASS_CACHE[method_id]

    dirs = discover_method_dirs()
    if method_id not in dirs:
        raise KeyError(
            f"Unknown method {method_id!r}. Available: {sorted(dirs.keys())}"
        )
    method_dir = dirs[method_id]
    adapter_py = method_dir / "adapter.py"

    # Every method's adapter.py commonly does `sys.path.insert(0, code_dir);
    # import model` / `import preprocessing` for its own methods/Bx/code/
    # submodules -- and those generic names (model, preprocessing, dataset,
    # ...) collide across methods once more than one has been imported in the
    # same process, because sys.modules caches by bare name: a second method's
    # `import model` silently returns the FIRST method's already-cached
    # `model` module instead of loading its own (confirmed empirically:
    # B6/B7/B8's own model classes were unreachable after B4 was imported
    # first in the same process). Isolate each load: after loading, remove
    # only the sys.modules entries this load added THAT LIVE UNDER
    # methods/ (i.e. this method's own or _internal_shared's local
    # submodules) -- the loaded adapter keeps working regardless, since
    # whatever it bound via `from model import Foo` is already captured in
    # its own namespace. Third-party/stdlib modules (numpy, torch, pandas,
    # ...) are deliberately NEVER purged: some C-extension modules raise
    # "cannot load module more than once per process" if re-imported after
    # being evicted from sys.modules, so purging must be scoped to exactly
    # the local-path modules causing the collision, not "everything new".
    mod_name = f"_wt_kuochong_method_adapter_{method_dir.name}"
    modules_before = set(sys.modules.keys())
    path_before = list(sys.path)

    def _is_local_methods_module(mod) -> bool:
        f = getattr(mod, "__file__", None)
        if not f:
            return False
        try:
            Path(f).resolve().relative_to(METHODS_DIR)
            return True
        except ValueError:
            return False
    try:
        spec = importlib.util.spec_from_file_location(mod_name, adapter_py)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        if not hasattr(module, "ADAPTER_CLASS"):
            raise AttributeError(f"{adapter_py} does not define module-level ADAPTER_CLASS")
        cls = module.ADAPTER_CLASS
        if getattr(cls, "method_id", None) != method_id:
            raise AttributeError(
                f"{adapter_py}: ADAPTER_CLASS.method_id={getattr(cls, 'method_id', None)!r} "
                f"does not match its directory-derived id {method_id!r}"
            )
        _CLASS_CACHE[method_id] = cls
        return cls
    finally:
        for name in set(sys.modules.keys()) - modules_before:
            if name == mod_name:  # keep the adapter module itself cached under its unique name
                continue
            mod = sys.modules.get(name)
            if mod is not None and _is_local_methods_module(mod):
                del sys.modules[name]
        sys.path[:] = path_before


def list_methods() -> list[dict]:
    """Returns [{method_id, method_name, method_dir}] for every discoverable
    method, best-effort (a method whose adapter.py fails to import is listed
    with method_name="<import error>" rather than raising, so `--method list`
    still shows every OTHER method)."""
    out = []
    for method_id, method_dir in discover_method_dirs().items():
        try:
            cls = load_adapter_class(method_id)
            name = getattr(cls, "method_name", method_id)
        except Exception as exc:  # noqa: BLE001
            name = f"<import error: {exc}>"
        out.append({"method_id": method_id, "method_name": name, "method_dir": str(method_dir)})
    return out
