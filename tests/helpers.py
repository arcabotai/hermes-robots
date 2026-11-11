import struct
import zlib

GEOFENCE = {"safety": {"geofence": {"enabled": True, "x_min": -1.0, "x_max": 3.0, "y_min": -2.0, "y_max": 2.0}}}


def decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    w = h = 0
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = 1 + w * 3
    px = []
    for row in range(h):
        assert raw[row * stride] == 0  # filter None (our encoder)
        px.append(raw[row * stride + 1:(row + 1) * stride])
    return w, h, px


def red_pixels(png):
    w, h, rows = decode_png(png)
    n = 0
    for r in rows:
        for i in range(0, len(r), 3):
            if r[i] > 180 and r[i + 1] < 90 and r[i + 2] < 90:
                n += 1
    return n


def red_centroid(png):
    w, h, rows = decode_png(png)
    cols = [i // 3 for r in rows for i in range(0, len(r), 3) if r[i] > 180 and r[i + 1] < 90 and r[i + 2] < 90]
    return sum(cols) / len(cols)
