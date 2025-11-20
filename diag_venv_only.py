import sys, site, os
print("sys.executable:", sys.executable)
print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE"))
import torch, numpy as np
print("torch:", torch.__version__, "from", torch.__file__)
print("numpy:", np.__version__, "from", np.__file__)
print("usersite:", site.getusersitepackages())
print("sys.path (first 8):", sys.path[:8])
