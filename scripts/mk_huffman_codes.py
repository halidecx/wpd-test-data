#!/usr/bin/env python3
"""Hand-writes VP8L stills exercising Huffman code shapes the format permits
but no encoder emits, plus the two-level table lookup encoders rarely reach.

  huffman_simple_duplicate  a simple code naming the same symbol twice
  huffman_simple_single     the same code naming it once, byte-different but
                            pixel-identical, so the pair is a self-check
  huffman_simple_forms      an ordinary two-symbol code, and the one-bit form
                            of a simple code's first symbol
  huffman_long_codes        145 symbols over lengths 7 to 15, read back through
                            the root table and three secondary tables

A simple code names one or two symbols, and nothing stops both being the same
one.  That has to collapse to a single-symbol code, which consumes no bits at
all; counting it as two makes the tree one bit deep and desynchronises every
read after it.  The decode stays in bounds and returns the wrong pixels rather
than crashing, so neither a sanitizer nor a fuzzer sees it.

Every code here is derived from the specification -- canonical assignment out
of the length list, written high bit first so the decoder reads back its bit
reversal -- so the expected pixels are computed rather than recorded.  Each
file is written with the ARGB it must decode to and that ARGB's MD5, which is
what wpd's testdata suite checks with --verify.
"""

import hashlib
import os
import struct
import sys

MAX_CODE_LENGTH = 15
NUM_CODE_LENGTH_CODES = 19
CODE_LENGTH_ORDER = (17, 18, 0, 1, 2, 3, 4, 5, 16,
                     6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

GREEN = 0x40
RED = 0x37
RED2 = 0x9E
BLUE = 0x8A
ALPHA = 0xC1


class BitWriter:
    """VP8L is LSB-first: the first bit of a byte is its least significant."""

    def __init__(self):
        self.bits = []

    def put(self, value, n):
        for i in range(n):
            self.bits.append((value >> i) & 1)

    def put_code(self, code, length):
        """Canonical codes go out high bit first, so the decoder's LSB-first
        read reconstructs the bit reversal its tables are indexed by."""
        for i in reversed(range(length)):
            self.bits.append((code >> i) & 1)

    def bytes(self):
        out = bytearray((len(self.bits) + 7) // 8)
        for i, bit in enumerate(self.bits):
            if bit:
                out[i >> 3] |= 1 << (i & 7)
        return bytes(out)


def canonical(lengths):
    """Codes assigned the way the specification defines them: shortest first,
    and by symbol within a length."""
    counts = [0] * (MAX_CODE_LENGTH + 1)
    for length in lengths:
        counts[length] += 1
    counts[0] = 0

    code = 0
    nxt = [0] * (MAX_CODE_LENGTH + 1)
    for length in range(1, MAX_CODE_LENGTH + 1):
        code = (code + counts[length - 1]) << 1
        nxt[length] = code

    codes = {}
    for symbol, length in enumerate(lengths):
        if length:
            codes[symbol] = (nxt[length], length)
            nxt[length] += 1
    return codes


def put_simple_code(w, symbols, first_is_8bit=True):
    """The simple code: one or two symbols written literally, both one bit
    long, no length list at all."""
    w.put(1, 1)
    w.put(len(symbols) - 1, 1)
    w.put(1 if first_is_8bit else 0, 1)
    w.put(symbols[0], 8 if first_is_8bit else 1)
    if len(symbols) == 2:
        w.put(symbols[1], 8)


# Lengths 9 and up are what force secondary tables; spreading them over 9 to 15
# is what makes those tables differ in depth.  Complete by construction:
# 126/128 + 2/512 + 11/1024 + 1/2048 + 1/4096 + 1/8192 + 1/16384 + 2/32768 = 1.
LONG_CODE_COUNTS = {7: 126, 9: 2, 10: 11, 11: 1, 12: 1, 13: 1, 14: 1, 15: 2}


def long_code_lengths():
    lengths = []
    for value in sorted(LONG_CODE_COUNTS):
        lengths += [value] * LONG_CODE_COUNTS[value]
    return lengths


def put_long_code(w, lengths):
    """The normal code: a Huffman code over code lengths, then the lengths."""
    meta = [0] * NUM_CODE_LENGTH_CODES
    for value in sorted(set(lengths)):
        if value:
            meta[value] = 3
    for value in (14, 15):
        if meta[value]:
            meta[value] = 4
    meta[0] = 3
    kraft = sum(2 ** -length for length in meta if length)
    assert kraft == 1, 'code length code is not complete: %r' % kraft

    w.put(0, 1)
    w.put(NUM_CODE_LENGTH_CODES - 4, 4)
    for symbol in CODE_LENGTH_ORDER:
        w.put(meta[symbol], 3)

    # An explicit symbol limit, so the reader stops rather than running the
    # length list out to the full alphabet.
    w.put(1, 1)
    w.put(3, 3)
    w.put(len(lengths) - 2, 8)

    meta_codes = canonical(meta)
    for value in lengths:
        w.put_code(*meta_codes[value])


def put_header(w, width, height):
    w.put(0x2F, 8)
    w.put(width - 1, 14)
    w.put(height - 1, 14)
    w.put(1, 1)          # alpha is used
    w.put(0, 3)          # version 0
    w.put(0, 1)          # no transform
    w.put(0, 1)          # no colour cache
    w.put(0, 1)          # no meta-Huffman image


def wrap_riff(w):
    payload = w.bytes()
    padded = payload + b'\0' * (len(payload) & 1)
    body = b'WEBP' + b'VP8L' + struct.pack('<I', len(payload)) + padded
    return b'RIFF' + struct.pack('<I', len(body)) + body


def simple_still(codes, width, height):
    """A still whose five codes are all simple and single-symbol, so every tree
    consumes zero bits and the image needs no payload at all."""
    w = BitWriter()
    put_header(w, width, height)
    for symbols in codes:
        put_simple_code(w, symbols)
    return wrap_riff(w)


def build_simple_duplicate():
    """Red names 0x37 twice.  Both entries describe the same symbol, so the
    code has to collapse to one that consumes no bits."""
    data = simple_still([[GREEN], [RED, RED], [BLUE], [ALPHA], [0]], 4, 4)
    return data, [(ALPHA, RED, GREEN, BLUE)] * 16


def build_simple_single():
    """The same image with red naming 0x37 once: a different byte sequence that
    must decode identically."""
    data = simple_still([[GREEN], [RED], [BLUE], [ALPHA], [0]], 4, 4)
    return data, [(ALPHA, RED, GREEN, BLUE)] * 16


def build_simple_forms():
    """Red names two distinct symbols, so canonical order puts the smaller on
    the zero bit and each pixel spends one bit choosing.  Alpha uses the
    one-bit form of the first symbol, which can only be 0 or 1."""
    w = BitWriter()
    put_header(w, 4, 4)
    put_simple_code(w, [GREEN])
    put_simple_code(w, [RED, RED2])
    put_simple_code(w, [BLUE])
    put_simple_code(w, [1], first_is_8bit=False)
    put_simple_code(w, [0])

    low, high = min(RED, RED2), max(RED, RED2)
    pixels = []
    for i in range(16):
        w.put(i & 1, 1)
        pixels.append((1, high if i & 1 else low, GREEN, BLUE))
    return wrap_riff(w), pixels


def build_long_codes():
    """Green symbols of length 7, 9, 10 and 15: one answered by the root table
    and three by secondary tables of three different depths."""
    lengths = long_code_lengths()
    codes = canonical(lengths)
    greens = [0, 126, 138, 144]

    w = BitWriter()
    put_header(w, len(greens), 1)
    put_long_code(w, lengths)
    put_simple_code(w, [RED])
    put_simple_code(w, [BLUE])
    put_simple_code(w, [ALPHA])
    put_simple_code(w, [0])
    for green in greens:
        w.put_code(*codes[green])
    return wrap_riff(w), [(ALPHA, RED, green, BLUE) for green in greens]


BUILDERS = (
    ('huffman_simple_duplicate', build_simple_duplicate),
    ('huffman_simple_single', build_simple_single),
    ('huffman_simple_forms', build_simple_forms),
    ('huffman_long_codes', build_long_codes),
)


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    for name, build in BUILDERS:
        data, pixels = build()
        raw = b''.join(bytes(p) for p in pixels)
        with open(os.path.join(outdir, name + '.webp'), 'wb') as f:
            f.write(data)
        with open(os.path.join(outdir, name + '.webp.argb'), 'wb') as f:
            f.write(raw)
        print("['%s', 'argb', '%s'],  # %d bytes" %
              (name, hashlib.md5(raw).hexdigest(), len(data)))
