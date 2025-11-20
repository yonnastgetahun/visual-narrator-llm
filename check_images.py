import glob, os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "train2017"
patts = ("*.jpg","*.jpeg","*.png")
files = []
for p in patts:
    files.extend(glob.glob(os.path.join(root, p)))
print(f"Root: {root}")
print(f"Found: {len(files)} images")
for f in sorted(files)[:5]:
    print(" -", f)
