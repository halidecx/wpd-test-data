#!/usr/bin/env python3
"""Makes the two lossy-with-alpha files that do *not* take wpd's eight-bit
ALPH decode.

An ALPH chunk is a lossless image whose green channel is the alpha plane.
Almost every one an encoder emits is a palette and nothing else, with no
colour cache and a single symbol for red, blue and alpha -- libwebp's
Is8bOptimizable -- and both libwebp and wpd decode that shape straight into
one byte per pixel.  Every alpha-carrying file in this repo is that shape, so
the two paths for the ones that are not were reached by nothing:

  a_lossy_gradient  a horizontal alpha ramp, whose ALPH carries no palette
                    transform at all, so the plane is the green channel
                    extracted from a full ARGB canvas.
  a_lossy_cached    a radial alpha falloff, whose ALPH is a palette *with* a
                    colour cache, which fails Is8bOptimizable and takes the
                    32-bit palette lookup.

Which one cwebp picks is a property of the alpha content, not a flag, which
is why the two sources look the way they do.  The quality has to be 100:
-alpha_q below it, and -alpha_method 0, both come back out as shapes the
eight-bit path handles.

Made with cwebp 1.6.0.  Reproducing the exact bytes needs that version; any
version produces files that serve the same purpose, but the md5s in wpd's
tests/meson.build would have to be taken again.
"""

import math
import os
import random
import struct
import subprocess
import sys
import zlib

W, H = 160, 120


def write_png(path, rows):
    """The smallest 8-bit RGBA PNG that will do, so nothing but zlib is
    needed to make one."""
    raw = b"".join(b"\x00" + bytes(row) for row in rows)

    def chunk(tag, body):
        block = tag + body
        return (struct.pack(">I", len(body)) + block +
                struct.pack(">I", zlib.crc32(block) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", header))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def gradient_rows():
    """Alpha rises left to right over every value a byte has, and the colour
    underneath is noisy enough that the encoder cannot palette either."""
    for y in range(H):
        row = []
        for x in range(W):
            row += [(x * 3) % 256, (y * 5) % 256, (x * y) % 256,
                    (x * 255) // (W - 1)]
        yield row


def radial_rows():
    """Alpha falls off from the centre over a flat colour, which palettes
    but wants a colour cache for the runs the falloff makes."""
    far = math.hypot(W / 2, H / 2)
    for y in range(H):
        row = []
        for x in range(W):
            d = int(math.hypot(x - W / 2, y - H / 2) * 255 / far)
            row += [200, 120, 60, 255 - d]
        yield row


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    random.seed(7)

    for name, rows in [("a_lossy_gradient", gradient_rows()),
                       ("a_lossy_cached", radial_rows())]:
        png = os.path.join(out, name + ".png")
        webp = os.path.join(out, name + ".webp")

        write_png(png, rows)
        subprocess.run(["cwebp", "-quiet", "-q", "80", "-alpha_method", "1",
                        "-alpha_q", "100", png, "-o", webp], check=True)
        os.remove(png)
        print(webp)


if __name__ == "__main__":
    sys.exit(main())
