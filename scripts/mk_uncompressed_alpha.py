import random
import struct
from pathlib import Path


def riff_chunks(data):
    offset = 12
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset + 4:offset + 8], "little")
        end = offset + 8 + size + (size & 1)
        yield data[offset:offset + 4], data[offset + 8:offset + 8 + size]
        offset = end


def encode(plane, width, height, mode):
    if mode == 0:
        return bytes(plane)
    out = bytearray(len(plane))
    for y in range(height):
        for x in range(width):
            at = y * width + x
            if x == 0 and y == 0:
                predictor = 0
            elif y == 0:
                predictor = plane[at - 1]
            elif x == 0:
                predictor = plane[at - width]
            elif mode == 1:
                predictor = plane[at - 1]
            elif mode == 2:
                predictor = plane[at - width]
            else:
                predictor = max(
                    0,
                    min(255, plane[at - 1] + plane[at - width] -
                        plane[at - width - 1]),
                )
            out[at] = (plane[at] - predictor) & 0xff
    return bytes(out)


def chunk(tag, payload):
    return tag + struct.pack("<I", len(payload)) + payload + b"\0" * (len(payload) & 1)


def main():
    root = Path(__file__).resolve().parent.parent
    source = (root / "odd_a_lossy.webp").read_bytes()
    chunks = list(riff_chunks(source))
    vp8x = next(payload for tag, payload in chunks if tag == b"VP8X")
    width = 1 + int.from_bytes(vp8x[4:7], "little")
    height = 1 + int.from_bytes(vp8x[7:10], "little")
    rng = random.Random(7)
    plane = bytes(rng.randrange(256) for _ in range(width * height))
    names = ["none", "horizontal", "vertical", "gradient"]

    for mode, name in enumerate(names):
        body = b"".join(
            chunk(tag, bytes([mode << 2]) + encode(plane, width, height, mode))
            if tag == b"ALPH" else chunk(tag, payload)
            for tag, payload in chunks
        )
        output = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body
        path = root / f"alpha_uncompressed_{name}.webp"
        path.write_bytes(output)
        print(path)


if __name__ == "__main__":
    main()
