import sys, site
print("sys.executable:", sys.executable)
print("sys.version:", sys.version.split()[0])
print("VIRTUAL_ENV:", __import__("os").environ.get("VIRTUAL_ENV"))

try:
    import torch
    print("torch.__version__:", torch.__version__)
    print("torch.__file__:", torch.__file__)
except Exception as e:
    print("torch import error:", e)

print("\nsite.getsitepackages():")
for p in site.getsitepackages(): print("  ", p)

print("\nsys.path (first 10):")
for p in sys.path[:10]: print("  ", p)
