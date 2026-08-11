#!/usr/bin/env python3
"""Hand-writes a VP8L still whose transforms are declared in an order no
encoder emits: PREDICTOR, COLOR_TRANSFORM and SUBTRACT_GREEN all before
COLOR_INDEXING.

Transforms are inverted in reverse declaration order, so the palette is
unpacked first and the other three then have to run over the full canvas
width rather than the packed one.  libwebp keeps a per-transform xsize_ and
gets this right; it is the case a single global "current width" gets wrong.

Writes the file alongside the ARGB the spec says it must decode to.
"""

import struct
import sys

W, H = 27, 200
PRED_BITS = 3                      # 8x8 predictor tiles, sized on the full width
PRED_W = (W + (1 << PRED_BITS) - 1) >> PRED_BITS
PRED_H = (H + (1 << PRED_BITS) - 1) >> PRED_BITS

NCOLORS = 4
PAL_BITS = 2                       # 4 pixels per byte
RED_W = (W + (1 << PAL_BITS) - 1) >> PAL_BITS

PALETTE = [(255, 0x10, 0x20, 0x30),
           (255, 0x50, 0x50, 0x50),
           (255, 0x90, 0x80, 0x70),
           (255, 0xD0, 0xB0, 0x90)]

CC_BITS = 4                        # 16x16 cross-colour tiles, also full width
CC_W = (W + (1 << CC_BITS) - 1) >> CC_BITS
CC_H = (H + (1 << CC_BITS) - 1) >> CC_BITS

# (alpha, red_to_blue, green_to_blue, green_to_red)
CC_MULTS = ((255, 0x0A, 0x14, 0x1E), (255, 0xF6, 0xEC, 0xE2))

MODES = (3, 11)                    # TR and Select
GREENS = (0x1B, 0x4E)              # index runs 3,2,1,0 and 2,3,0,1


class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, val, bits):
        assert 0 <= val < (1 << bits), (val, bits)
        for i in range(bits):
            self.acc |= ((val >> i) & 1) << self.n
            self.n += 1
            if self.n == 8:
                self.buf.append(self.acc)
                self.acc = 0
                self.n = 0

    def done(self):
        if self.n:
            self.buf.append(self.acc)
            self.n = 0
        return bytes(self.buf)


class SimpleCode:
    """A "simple" prefix code: one or two symbols, canonical 0/1 bit codes."""

    def __init__(self, values):
        self.syms = sorted(set(values))
        assert 1 <= len(self.syms) <= 2, self.syms

    def write(self, bw):
        bw.put(1, 1)                       # simple code
        bw.put(len(self.syms) - 1, 1)      # symbol count
        bw.put(1, 1)                       # first symbol is 8 bits wide
        bw.put(self.syms[0], 8)
        if len(self.syms) == 2:
            bw.put(self.syms[1], 8)

    def emit(self, bw, sym):
        if len(self.syms) == 2:
            bw.put(self.syms.index(sym), 1)


def write_image_stream(bw, pixels, meta_huffman_bit=False):
    """One entropy-coded image: no colour cache, no meta prefix codes, all
    pixels coded as literals."""
    codes = [SimpleCode(p[c] for p in pixels) for c in (2, 1, 3, 0)]
    bw.put(0, 1)                           # no colour cache
    if meta_huffman_bit:
        bw.put(0, 1)                       # no meta prefix codes
    for c in codes:
        c.write(bw)
    bw.put(1, 1)                           # distance code: one symbol
    bw.put(0, 1)
    bw.put(1, 1)
    bw.put(0, 8)
    for a, r, g, b in pixels:
        codes[0].emit(bw, g)
        codes[1].emit(bw, r)
        codes[2].emit(bw, b)
        codes[3].emit(bw, a)


def pred_modes():
    return [(0, MODES[(x + y) % 2], 0, 0)
            for y in range(PRED_H) for x in range(PRED_W)]


def cc_mults():
    return [CC_MULTS[(x + y) % 2]
            for y in range(CC_H) for x in range(CC_W)]


def packed_rows():
    """Palette indices, four to a green byte."""
    return [(0, 0, GREENS[((x * 7 + y * 13) % 5) < 2], 0)
            for y in range(H) for x in range(RED_W)]


def build():
    bw = BitWriter()
    bw.put(0x2F, 8)
    bw.put(W - 1, 14)
    bw.put(H - 1, 14)
    bw.put(1, 1)                           # alpha is used
    bw.put(0, 3)                           # version

    bw.put(1, 1)                           # transform follows
    bw.put(0, 2)                           # PREDICTOR_TRANSFORM
    bw.put(PRED_BITS - 2, 3)
    write_image_stream(bw, pred_modes())

    bw.put(1, 1)                           # transform follows
    bw.put(1, 2)                           # COLOR_TRANSFORM
    bw.put(CC_BITS - 2, 3)
    write_image_stream(bw, cc_mults())

    bw.put(1, 1)                           # transform follows
    bw.put(2, 2)                           # SUBTRACT_GREEN, no payload

    bw.put(1, 1)                           # transform follows
    bw.put(3, 2)                           # COLOR_INDEXING_TRANSFORM
    bw.put(NCOLORS - 1, 8)
    deltas = [PALETTE[0]]
    for i in range(1, NCOLORS):
        deltas.append(tuple((PALETTE[i][c] - PALETTE[i - 1][c]) & 0xFF
                            for c in range(4)))
    write_image_stream(bw, deltas)

    bw.put(0, 1)                           # no more transforms
    write_image_stream(bw, packed_rows(), meta_huffman_bit=True)

    payload = bw.done()
    chunk = b'VP8L' + struct.pack('<I', len(payload)) + payload
    if len(payload) & 1:
        chunk += b'\0'
    return b'RIFF' + struct.pack('<I', len(chunk) + 4) + b'WEBP' + chunk


def average2(a, b):
    return tuple((a[c] + b[c]) >> 1 for c in range(4))


def select(t, l, tl):
    d = sum(abs(l[c] - tl[c]) - abs(t[c] - tl[c]) for c in range(4))
    return t if d <= 0 else l


def predict(mode, left, top, topright, topleft):
    if mode == 0:
        return (255, 0, 0, 0)
    if mode == 1:
        return left
    if mode == 2:
        return top
    if mode == 3:
        return topright
    if mode == 4:
        return topleft
    if mode == 5:
        return average2(average2(left, topright), top)
    if mode == 6:
        return average2(left, topleft)
    if mode == 7:
        return average2(left, top)
    if mode == 8:
        return average2(topleft, top)
    if mode == 9:
        return average2(top, topright)
    if mode == 10:
        return average2(average2(left, topleft), average2(top, topright))
    if mode == 11:
        return select(top, left, topleft)
    raise AssertionError(mode)


def s8(v):
    return v - 256 if v >= 128 else v


def cc_delta(mult, color):
    return (s8(mult) * s8(color)) >> 5


def expected():
    """The spec's answer: unpack the palette, then run the other three
    transforms over the full width, in reverse declaration order."""
    modes = pred_modes()
    mults = cc_mults()
    packed = packed_rows()

    px = [[None] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            green = packed[y * RED_W + (x >> PAL_BITS)][2]
            idx = (green >> ((x & 3) * 2)) & 3
            px[y][x] = PALETTE[idx]

    for y in range(H):
        for x in range(W):
            a, r, g, b = px[y][x]
            r = (r + g) & 0xFF                       # subtract green, inverse
            b = (b + g) & 0xFF
            m = mults[(y >> CC_BITS) * CC_W + (x >> CC_BITS)]
            r = (r + cc_delta(m[3], g)) & 0xFF       # cross colour, inverse
            b = (b + cc_delta(m[2], g) + cc_delta(m[1], r)) & 0xFF
            px[y][x] = (a, r, g, b)

    for y in range(H):
        for x in range(W):
            if x == 0 and y == 0:
                mode = 0
            elif y == 0:
                mode = 1
            elif x == 0:
                mode = 2
            else:
                mode = modes[(y >> PRED_BITS) * PRED_W + (x >> PRED_BITS)][2]
            left = px[y][x - 1] if x else None
            top = px[y - 1][x] if y else None
            topleft = px[y - 1][x - 1] if x and y else None
            topright = px[y - 1][x + 1] if y and x + 1 < W else (
                px[y][0] if y else None)
            p = predict(mode, left, top, topright, topleft)
            px[y][x] = tuple((px[y][x][c] + p[c]) & 0xFF for c in range(4))
    return px


if __name__ == '__main__':
    out = sys.argv[1]
    with open(out, 'wb') as f:
        f.write(build())
    px = expected()
    raw = bytearray()
    for row in px:
        for a, r, g, b in row:
            raw += bytes((a, r, g, b))
    with open(out + '.argb', 'wb') as f:
        f.write(raw)
    print('wrote %s (%dx%d)' % (out, W, H))
