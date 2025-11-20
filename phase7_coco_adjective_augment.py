import os, json, random, argparse, pathlib

ADJ_BANK = [
  "vivid","textured","luminous","rugged","tranquil","gleaming","muted",
  "windswept","shadowy","glossy","velvety","weathered","crisp","dusky",
  "radiant","gritty","hazy","vibrant","serene","golden"
]

def adjectiveify(caption: str, p=0.85, max_inserts=2):
    if random.random() > p or not caption.strip():
        return caption
    words = caption.split()
    inserts = random.randint(1, max_inserts)
    for _ in range(inserts):
        pos = random.randrange(0, max(1,len(words)))
        words.insert(pos, random.choice(ADJ_BANK))
    return " ".join(words)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_json", default="annotations/captions_train2017.json")
    ap.add_argument("--out_json",  default="phase7/captions_train2017_adjective.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prob", type=float, default=0.85)
    ap.add_argument("--max_inserts", type=int, default=2)
    args = ap.parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.coco_json):
        raise FileNotFoundError(f"Missing {args.coco_json}")

    pathlib.Path(os.path.dirname(args.out_json) or ".").mkdir(parents=True, exist_ok=True)

    data = json.load(open(args.coco_json))
    n = len(data.get("annotations", []))
    for ann in data["annotations"]:
        ann["caption"] = adjectiveify(ann["caption"], p=args.prob, max_inserts=args.max_inserts)

    json.dump(data, open(args.out_json, "w"))
    print(f"✅ Wrote augmented captions: {args.out_json} (from {n} annotations)")

if __name__ == "__main__":
    main()
