import json, base64, requests, collections
from pathlib import Path

session = requests.Session()

def download_image(url):
    dst = Path("images") / Path(url).name
    if not dst.exists():
        print("Download", url)
        dst.parent.mkdir(exist_ok=True)
        with session.get(url) as r:
            r.raise_for_status()
            data = r.content
        dst.write_bytes(data)

mscoco_ids = [
    36, 77, 901, 909, 1204, 1681, 1781, 1804, 2106, 2244, 2823, 2998, 3219,
    3270, 3478, 3579, 3751, 3782, 4719, 5483, 5832, 5883, 6253, 6262, 6270,
    6464, 6465, 6935, 6943, 7539, 7753, 8055, 8965, 9287, 9321, 10130, 10784,
    11004, 11292, 11569, 11697, 11826, 11931, 12349, 12544, 12556, 12726,
    13616, 13720, 14230, 14319, 14446, 14472, 14726, 14812, 15165, 15239,
    15273, 15617, 16060, 16556, 16629, 17057, 17683, 17856, 17877, 18214,
    18293, 18439, 18466, 18559, 18773, 18801, 19349, 19499, 19980, 20644,
    20888, 20983, 21475, 21534, 21780, 21915, 22032, 22240, 22270, 22447,
    22484, 23015, 23569, 23874, 24091, 24125, 24149, 24257, 24380, 24393,
    24489, 24582, 24600, 24787, 25058, 25274, 25325, 26109, 26274, 26812,
    27015, 27163, 27675,
]

class RollingHash:
    def __init__(self, values, base=131, mod=10**9 + 7):
        self.mod = mod
        self.hashes = [0]
        self.powers = [1]
        for value in values:
            self.hashes.append((self.hashes[-1] * base + value) % self.mod)
            self.powers.append((self.powers[-1] * base) % self.mod)

    def hash(self, start, stop):
        return (self.hashes[stop] - self.hashes[start] * self.powers[(stop - start)]) % self.mod

def is_repeating(s, min_len=40, min_repeats=2):
    r = RollingHash([ord(c) for c in s])
    n = len(s)
    length = 1
    while length < n:
        # find starts of equal substrings of given length
        starts = collections.defaultdict(list)
        for start in range(n - length + 1):
            starts[r.hash(start, start + length)].append(start)

        # choose period as smallest distance between equal substrings
        period = n
        for st in starts.values():
            st = sorted(st)
            min_dist = min((b - a for a, b in zip(st, st[1:])), default=n)
            period = min(period, max(length, min_dist))

        # find "pattern" * num_repeats with total_len = period * num_repeats
        num_repeats = max(min_repeats, (min_len + period - 1) // period)
        total_len = num_repeats * period
        overlap_len = total_len - period
        for i in range(n - total_len + 1):
            j = i + period
            # do expensive check only when hashes of overlapping strings match
            if r.hash(i, i + overlap_len) == r.hash(j, j + overlap_len):
                pattern = s[i:i + period]
                if pattern * num_repeats == s[i:i + total_len]:
                    return True

        length = min(length * 2, period + 1)

    return False

def run(ocr):
    # download MSCOCO images
    for id in mscoco_ids:
        #url = f"http://images.cocodataset.org/train2014/COCO_train2014_{id:012d}.jpg" # alternative url
        url = f"https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2014/COCO_train2014_{id:012d}.jpg"
        download_image(url)

    # download misc images
    names = """
        100-to-1900-step-100-column.png
        diagram-1-to-8-circle-nodes.png
        duck.png
        e.png
        pepper.png
        repeated-text.png
        runes.png
        tree-diagram.jpg
        wikipedia-tables2.png
        wikipedia-tables3.png
        wikipedia-tables.png
        diagrams.png
        power-brick.png
        receipt-1.png
        receipt-2.png
    """
    for name in names.strip().split():
        url = f"https://github.com/99991/testing/raw/refs/heads/master/misc/images/{name}"
        download_image(url)

    prompts = [
        "OCR this image",
        "transcribe this image as markdown",
        "transcribe all the characters in this image exactly as written",
        "transcribe all the text in this image as json",
    ]

    paths = sorted(Path("images").glob("*"))

    ok = 0
    loopy = 0
    for path in paths:
        for prompt in prompts:
            text = ocr(path, prompt)

            repeating = is_repeating(text)

            if repeating:
                loopy += 1
            else:
                ok += 1

            progress = 100 * (ok + loopy) / len(paths) / len(prompts)
            percent_loopy = 100 * loopy / (ok + loopy)
            print(f"loopy: {loopy} ({percent_loopy:.1f} %), ok: {ok}, progress: {progress:.1f} %")

            with open("results.jsonl", "a") as f:
                f.write(json.dumps({"path": str(path), "prompt": prompt, "text": text, "repeating": repeating}) + "\n")

