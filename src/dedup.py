"""
dedup.py - find exact and near-duplicate images.

Idea (difference hash, "dHash"):
  1. shrink image to 9x8 grayscale  2. compare each pixel with its right
  neighbour -> 64 bits (1 if brighter).  Similar images -> similar bits.
  The number of differing bits is the "Hamming distance". <= ~4 of 64
  usually means "same picture, resized / recompressed / slightly cropped".
"""
import hashlib
import numpy as np
from PIL import Image

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def md5_file(path) -> str:
    """Exact-duplicate fingerprint (byte-identical files)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """64-bit perceptual hash of a PIL image."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_duplicate_groups(hashes, threshold: int = 4):
    """
    hashes: list of ints. Returns a group id per image (union-find), so images
    within `threshold` bits of each other end up in the same group.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    arr = np.array(hashes, dtype=np.uint64)
    for i in range(n - 1):
        xor = arr[i + 1:] ^ arr[i]                       # vectorised XOR
        dist = _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for j in np.nonzero(dist <= threshold)[0]:
            ri, rj = find(i), find(i + 1 + int(j))
            if ri != rj:
                parent[rj] = ri
    return [find(i) for i in range(n)]
