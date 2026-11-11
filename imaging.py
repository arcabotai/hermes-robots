"""Pure-Python image helpers: PNG encode, sensor_msgs/Image decode, and the mock camera renderers.

No Pillow / numpy: the plugin must install into any Hermes environment and run on a CI box.
"""
from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Iterable, List, Optional, Sequence, Tuple

RGB = Tuple[int, int, int]


# --------------------------------------------------------------------------- PNG
def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_png(width: int, height: int, rgb: bytes) -> bytes:
    """Encode packed RGB bytes (``width*height*3``) as a PNG."""
    if len(rgb) != width * height * 3:
        raise ValueError("rgb buffer does not match width*height*3")
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += rgb[y * stride:(y + 1) * stride]
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _chunk(b"IEND", b""))


def png_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        i += 2 + seglen
    return None


def sniff_image(data: bytes) -> Tuple[str, str, Optional[Tuple[int, int]]]:
    """Return ``(mime, extension, (w, h) | None)`` for PNG/JPEG bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png", png_size(data)
    if data[:2] == b"\xff\xd8":
        return "image/jpeg", "jpg", jpeg_size(data)
    raise ValueError("unrecognised image format (expected PNG or JPEG bytes)")


# ------------------------------------------------------- sensor_msgs/Image decode
_CHANNELS = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8UC1": 1,
             "mono16": 2, "16UC1": 2, "32FC1": 4}


def image_msg_to_png(msg: dict) -> bytes:
    """Convert a rosbridge ``sensor_msgs/Image`` dict (base64 ``data``) to PNG bytes."""
    width, height = int(msg["width"]), int(msg["height"])
    encoding = str(msg.get("encoding", "rgb8"))
    if encoding not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r}; supported: {sorted(_CHANNELS)}")
    data = msg["data"]
    if isinstance(data, str):
        data = base64.b64decode(data)
    elif isinstance(data, (list, tuple)):
        data = bytes(data)
    bpp = _CHANNELS[encoding]
    step = int(msg.get("step") or width * bpp)
    big = bool(msg.get("is_bigendian", 0))
    if len(data) < step * height:
        raise ValueError("image data shorter than step*height")
    out = bytearray(width * height * 3)
    o = 0
    depth_vals: List[float] = []
    for y in range(height):
        row = data[y * step:y * step + width * bpp]
        if encoding == "rgb8":
            out[o:o + width * 3] = row
            o += width * 3
        elif encoding == "bgr8":
            r = bytearray(row)
            r[0::3], r[2::3] = row[2::3], row[0::3]
            out[o:o + width * 3] = r
            o += width * 3
        elif encoding in ("rgba8", "bgra8"):
            px = bytearray(width * 3)
            if encoding == "rgba8":
                px[0::3], px[1::3], px[2::3] = row[0::4], row[1::4], row[2::4]
            else:
                px[0::3], px[1::3], px[2::3] = row[2::4], row[1::4], row[0::4]
            out[o:o + width * 3] = px
            o += width * 3
        elif encoding in ("mono8", "8UC1"):
            for v in row:
                out[o] = out[o + 1] = out[o + 2] = v
                o += 3
        elif encoding in ("mono16", "16UC1"):
            for i in range(width):
                v = int.from_bytes(row[2 * i:2 * i + 2], "big" if big else "little")
                depth_vals.append(float(v))
        else:  # 32FC1 (depth in metres)
            fmt = (">" if big else "<") + f"{width}f"
            depth_vals.extend(struct.unpack(fmt, row))
    if encoding in ("mono16", "16UC1", "32FC1"):
        finite = [v for v in depth_vals if math.isfinite(v) and v > 0]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        span = (hi - lo) or 1.0
        o = 0
        for v in depth_vals:
            g = 0 if not (math.isfinite(v) and v > 0) else int(255 * (1 - (v - lo) / span) * 0.9 + 12)
            g = max(0, min(255, g))
            out[o] = out[o + 1] = out[o + 2] = g
            o += 3
    return encode_png(width, height, bytes(out))


def compressed_msg_to_bytes(msg: dict) -> bytes:
    data = msg["data"]
    if isinstance(data, str):
        return base64.b64decode(data)
    return bytes(data)


# ------------------------------------------------------------- mock camera views
SKY_TOP: RGB = (92, 150, 214)
SKY_HORIZON: RGB = (190, 220, 240)
FLOOR_FAR: RGB = (120, 118, 112)
FLOOR_NEAR: RGB = (70, 68, 64)
WALL_COLORS = {"north": (176, 96, 72), "south": (86, 132, 176), "east": (96, 160, 104), "west": (200, 176, 88)}
CONE_RED: RGB = (226, 40, 32)
CONE_WHITE: RGB = (245, 245, 240)


def _lerp(a: RGB, b: RGB, t: float) -> RGB:
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t), int(a[2] + (b[2] - a[2]) * t))


def _ray_wall(x: float, y: float, ang: float, hx: float, hy: float) -> Tuple[float, str]:
    """Distance from (x, y) along ``ang`` to the rectangular arena wall [-hx,hx]x[-hy,hy]."""
    dx, dy = math.cos(ang), math.sin(ang)
    best, which = 1e9, "north"
    if dx > 1e-9:
        t = (hx - x) / dx
        if 0 < t < best:
            best, which = t, "east"
    elif dx < -1e-9:
        t = (-hx - x) / dx
        if 0 < t < best:
            best, which = t, "west"
    if dy > 1e-9:
        t = (hy - y) / dy
        if 0 < t < best:
            best, which = t, "north"
    elif dy < -1e-9:
        t = (-hy - y) / dy
        if 0 < t < best:
            best, which = t, "south"
    return best, which


def render_forward(x: float, y: float, yaw: float, half_x: float, half_y: float,
                   cones: Sequence[Tuple[float, float]], width: int = 320, height: int = 240,
                   hfov: float = 1.5, cam_height: float = 0.3, raw_rgb: bool = False):
    """Render a forward-looking camera view of the arena (software raycaster).

    Returns PNG bytes, or the raw ``rgb8`` pixel buffer when ``raw_rgb`` is true (used by the ROS demo camera node)."""
    horizon = height // 2
    f = (width / 2) / math.tan(hfov / 2)
    buf = bytearray(width * height * 3)
    # sky + floor gradient, one row colour at a time
    for row in range(height):
        if row < horizon:
            c = _lerp(SKY_TOP, SKY_HORIZON, row / max(1, horizon))
        else:
            c = _lerp(FLOOR_FAR, FLOOR_NEAR, (row - horizon) / max(1, height - horizon))
        buf[row * width * 3:(row + 1) * width * 3] = bytes(c) * width
    wall_h = 2.2
    depth = [0.0] * width
    for col in range(width):
        phi = math.atan2((width / 2 - col - 0.5), f)  # +left
        ang = yaw + phi
        dist, which = _ray_wall(x, y, ang, half_x, half_y)
        perp = max(0.05, dist * math.cos(phi))
        depth[col] = perp
        top = int(horizon - f * (wall_h - cam_height) / perp)
        bot = int(horizon + f * cam_height / perp)
        shade = max(0.35, min(1.0, 1.15 - perp / 14.0))
        base = WALL_COLORS[which]
        # light vertical banding so motion/distance is legible
        band = 0.92 if int((x + dist * math.cos(ang) + y + dist * math.sin(ang)) * 2) % 2 else 1.0
        c = bytes(int(v * shade * band) for v in base)
        for row in range(max(0, top), min(height, bot)):
            i = (row * width + col) * 3
            buf[i:i + 3] = c
    for cx, cy in cones:
        rx, ry = cx - x, cy - y
        dist = math.hypot(rx, ry)
        if dist < 0.12:
            continue
        bearing = math.atan2(ry, rx) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        if abs(bearing) > hfov / 2 + 0.4:
            continue
        perp = dist * math.cos(bearing) if abs(bearing) < math.pi / 2 else 0.0
        if perp < 0.08:
            continue
        r_base, h_cone = 0.2, 0.7
        row_bot = int(horizon + f * cam_height / perp)
        row_top = int(horizon + f * (cam_height - h_cone) / perp)
        cx_px = width / 2 - f * math.tan(bearing)
        for row in range(max(0, row_top), min(height, row_bot + 1)):
            frac = (row - row_top) / max(1, row_bot - row_top)  # 0 apex .. 1 base
            half_w = f * r_base * frac / perp
            c0, c1 = int(cx_px - half_w), int(cx_px + half_w)
            colour = CONE_WHITE if 0.38 < frac < 0.58 else CONE_RED
            shade = max(0.5, min(1.0, 1.2 - perp / 12.0))
            cb = bytes(int(v * shade) for v in colour)
            for col in range(max(0, c0), min(width, c1 + 1)):
                if depth[col] >= perp - 0.05:  # not hidden behind a nearer wall
                    i = (row * width + col) * 3
                    buf[i:i + 3] = cb
    return bytes(buf) if raw_rgb else encode_png(width, height, bytes(buf))


def _disc(buf: bytearray, w: int, h: int, cx: float, cy: float, r: float, colour: RGB) -> None:
    for yy in range(max(0, int(cy - r)), min(h, int(cy + r) + 1)):
        for xx in range(max(0, int(cx - r)), min(w, int(cx + r) + 1)):
            if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                i = (yy * w + xx) * 3
                buf[i:i + 3] = bytes(colour)


def _line(buf: bytearray, w: int, h: int, x0: float, y0: float, x1: float, y1: float, colour: RGB) -> None:
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for k in range(n + 1):
        t = k / max(1, n)
        xx, yy = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
        if 0 <= xx < w and 0 <= yy < h:
            i = (yy * w + xx) * 3
            buf[i:i + 3] = bytes(colour)


def render_topdown(x: float, y: float, yaw: float, half_x: float, half_y: float,
                   cones: Sequence[Tuple[float, float]], trail: Iterable[Tuple[float, float]] = (),
                   size: int = 360) -> bytes:
    """Top-down map of the arena (robot = blue disc + heading, cone = red disc)."""
    margin = 18
    scale = (size - 2 * margin) / (2 * max(half_x, half_y))
    cx0 = size / 2
    cy0 = size / 2

    def px(wx: float, wy: float) -> Tuple[float, float]:
        return cx0 + wx * scale, cy0 - wy * scale

    buf = bytearray(bytes((236, 238, 240)) * (size * size))
    for gx in range(-int(half_x), int(half_x) + 1):  # 1 m grid
        a, b = px(gx, -half_y), px(gx, half_y)
        _line(buf, size, size, a[0], a[1], b[0], b[1], (214, 218, 222))
    for gy in range(-int(half_y), int(half_y) + 1):
        a, b = px(-half_x, gy), px(half_x, gy)
        _line(buf, size, size, a[0], a[1], b[0], b[1], (214, 218, 222))
    corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    for i in range(4):
        a, b = px(*corners[i]), px(*corners[(i + 1) % 4])
        for d in range(-1, 2):
            _line(buf, size, size, a[0] + d, a[1], b[0] + d, b[1], (60, 60, 66))
            _line(buf, size, size, a[0], a[1] + d, b[0], b[1] + d, (60, 60, 66))
    pts = list(trail)
    for i in range(1, len(pts)):
        a, b = px(*pts[i - 1]), px(*pts[i])
        _line(buf, size, size, a[0], a[1], b[0], b[1], (140, 170, 230))
    for cx, cy in cones:
        p = px(cx, cy)
        _disc(buf, size, size, p[0], p[1], 7, CONE_RED)
        _disc(buf, size, size, p[0], p[1], 3, CONE_WHITE)
    p = px(x, y)
    _disc(buf, size, size, p[0], p[1], 8, (30, 90, 200))
    tip = (p[0] + math.cos(yaw) * 16, p[1] - math.sin(yaw) * 16)
    for d in (-1, 0, 1):
        _line(buf, size, size, p[0] + d, p[1], tip[0] + d, tip[1], (255, 255, 255))
    return encode_png(size, size, bytes(buf))
