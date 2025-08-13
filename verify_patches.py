import inspect
import polypesto  # triggers patches
from amici import sbml_import

print("amici:", getattr(__import__('amici'), '__version__', 'unknown'))
print("has SbmlImporter? ->", hasattr(sbml_import, "SbmlImporter"))
cls = getattr(sbml_import, "SbmlImporter", None)
print("class ->", cls)

wrapped = None
for n in dir(cls):
    fn = getattr(cls, n, None)
    if callable(fn):
        try:
            src = inspect.getsource(fn)
        except Exception:
            continue
        if "def _guard(" in src or fn.__module__ == "polypesto._patches":
            wrapped = (n, fn)
            break

print("wrapped method ->", wrapped[0] if wrapped else None)
print("method module  ->", getattr(wrapped[1], "__module__", None) if wrapped else None)
print("✅ amici patch check done")