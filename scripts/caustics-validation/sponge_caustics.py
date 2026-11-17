#!/usr/bin/env python3
"""Light that reaches a floor through a glass object, from three renders of one scene.

The three renders of a renderer are: the bare floor, the floor under an OPAQUE copy of the object
(its shadow), and the floor under the GLASS object. Per floor pixel,

    T = (glass - opaque) / (bare - opaque)

is the fraction of the light the opaque object blocks that the glass gives back (menger#150). This
module reads the renders (PFM), works out from the camera which pixels see the floor unobstructed
and how much floor each covers, and prints, per renderer:

  fp      energy fraction returned inside the shadow footprint (pixels where the opaque object
          blocks at least FOOTPRINT_FRACTION of the full shadow depth)
  all     the same energy summed over every floor pixel, so light the glass throws outside the
          footprint counts too; the denominator is still the blocked energy inside the footprint
  Tmean   mean of the per-pixel T in the footprint; Tp99 / Tmax its upper tail

Energy is weighted by the floor area each pixel covers, so a perspective camera does not skew it.
Pixels that see the object's bounding box (not the floor behind it) and pixels off the floor
quad are left out of every sum. Standard library only.
"""
import argparse
import math
import struct
import sys
from array import array

FOOTPRINT_FRACTION = 0.5     # a pixel is in the shadow footprint at this share of the full depth
FULL_SHADOW_PERCENTILE = 95  # "full shadow depth" = this percentile of (bare - opaque)
HORIZONTAL_FOV_DEG = 45.0    # menger's OptiX camera; fixed


def load_pfm(path):
    """(width, height, pixels), pixels row-major from the TOP row, each pixel (r, g, b)."""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        width, height = (int(v) for v in f.readline().split())
        scale = float(f.readline())
        data = f.read()
    channels = 3 if kind == b"PF" else 1
    fmt = ("<" if scale < 0 else ">") + "f" * (width * height * channels)
    values = struct.unpack(fmt, data[: 4 * width * height * channels])
    rows = []
    for y in range(height - 1, -1, -1):  # PFM stores the bottom row first
        for x in range(width):
            i = (y * width + x) * channels
            rows.append(tuple(values[i:i + 3]) if channels == 3 else (values[i],) * 3)
    return width, height, rows


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return (a[0] / n, a[1] / n, a[2] / n)


class Camera:
    """menger's pinhole camera: u = up x w, v = w x u, scaled by the horizontal fov; pbrt's
    LookAt agrees, so one description serves both renderers."""

    def __init__(self, eye, lookat, up, width, height):
        self.eye = eye
        self.w = _norm(_sub(lookat, eye))
        u = _norm(_cross(up, self.w))
        v = _cross(self.w, u)
        half = math.tan(math.radians(HORIZONTAL_FOV_DEG) / 2)
        self.u = tuple(c * half for c in u)
        self.v = tuple(c * half * height / width for c in v)
        self.width, self.height = width, height

    def direction(self, px, py):
        """Ray direction through the (fractional) pixel coordinate; py counts from the top."""
        a = (px / self.width) * 2 - 1
        b = 1 - (py / self.height) * 2
        return tuple(a * self.u[i] + b * self.v[i] + self.w[i] for i in range(3))

    def floor_point(self, px, py, floor_y):
        d = self.direction(px, py)
        if d[1] >= 0:
            return None
        t = (floor_y - self.eye[1]) / d[1]
        return tuple(self.eye[i] + t * d[i] for i in range(3))

    def hits_box_before(self, px, py, half, t_limit):
        """True when the pixel's ray enters the origin-centred cube (edge 2*half) before t_limit."""
        d = self.direction(px, py)
        t_enter, t_exit = 0.0, t_limit
        for i in range(3):
            if abs(d[i]) < 1e-12:
                if abs(self.eye[i]) > half:
                    return False
                continue
            t0, t1 = (-half - self.eye[i]) / d[i], (half - self.eye[i]) / d[i]
            t_enter, t_exit = max(t_enter, min(t0, t1)), min(t_exit, max(t0, t1))
        return t_enter < t_exit


def pixel_floor_areas(camera, floor_y, box_half, extent):
    """Per pixel: the floor area it covers, or None when it does not see free floor (the
    horizon side, off the floor quad of half-width `extent`, or the object's bounding cube)."""
    w, h = camera.width, camera.height
    corners = [[camera.floor_point(x, y, floor_y) for x in range(w + 1)] for y in range(h + 1)]
    areas = []
    for y in range(h):
        for x in range(w):
            quad = (corners[y][x], corners[y][x + 1], corners[y + 1][x + 1], corners[y + 1][x])
            centre = camera.floor_point(x + 0.5, y + 0.5, floor_y)
            if None in quad or centre is None or max(abs(centre[0]), abs(centre[2])) > extent:
                areas.append(None)
                continue
            t_floor = (floor_y - camera.eye[1]) / camera.direction(x + 0.5, y + 0.5)[1]
            if camera.hits_box_before(x + 0.5, y + 0.5, box_half, t_floor):
                areas.append(None)
                continue
            tri1 = _cross(_sub(quad[1], quad[0]), _sub(quad[2], quad[0]))
            tri2 = _cross(_sub(quad[2], quad[0]), _sub(quad[3], quad[0]))
            areas.append(0.5 * (math.sqrt(sum(c * c for c in tri1)) + math.sqrt(sum(c * c for c in tri2))))
    return areas


def _percentile(values, pct):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct / 100))]


def fractions(bare, opaque, glass, areas, without=None):
    """The statistics for one renderer; bare/opaque/glass are pixel lists as from load_pfm.
    `without`, when given, replaces `opaque` as what `glass` is measured against: the light the
    glass render adds to it, relative to what the opaque object blocks (menger's photons alone:
    the render with caustics minus the same render without)."""
    base = opaque if without is None else without
    lum = lambda p: (p[0] + p[1] + p[2]) / 3
    valid = [i for i, a in enumerate(areas) if a is not None]
    depth = {i: lum(bare[i]) - lum(opaque[i]) for i in valid}
    full = _percentile([depth[i] for i in valid], FULL_SHADOW_PERCENTILE)
    footprint = [i for i in valid if depth[i] > FOOTPRINT_FRACTION * full]
    if not footprint or full <= 0:
        raise ValueError("no shadow footprint found: bare and opaque renders do not differ")

    def returned(i, channel=None):
        if channel is None:
            return lum(glass[i]) - lum(base[i])
        return glass[i][channel] - base[i][channel]

    def blocked(channel=None):
        return sum(areas[i] * ((lum(bare[i]) - lum(opaque[i])) if channel is None
                               else bare[i][channel] - opaque[i][channel]) for i in footprint)

    per_pixel_t = sorted(returned(i) / depth[i] for i in footprint)
    return {
        "footprint_pixels": len(footprint),
        "fp": sum(areas[i] * returned(i) for i in footprint) / blocked(),
        "all": sum(areas[i] * returned(i) for i in valid) / blocked(),
        "fp_rgb": tuple(sum(areas[i] * returned(i, c) for i in footprint) / blocked(c) for c in range(3)),
        "all_rgb": tuple(sum(areas[i] * returned(i, c) for i in valid) / blocked(c) for c in range(3)),
        "t_mean": sum(per_pixel_t) / len(per_pixel_t),
        "t_p99": per_pixel_t[min(len(per_pixel_t) - 1, int(len(per_pixel_t) * 0.99))],
        "t_max": per_pixel_t[-1],
        "bare_max": max(lum(bare[i]) for i in valid),
        "glass_max": max(max(glass[i]) for i in valid),
    }


def _gate_passes(stats, want_fp, want_all, rel_tol, label):
    ok = True
    for name, want in (("fp", want_fp), ("all", want_all)):
        got = stats[name]
        if abs(got - want) > rel_tol * want:
            print(f"GATE FAIL {label}: {name} {got:.3f} is not within {rel_tol:.0%} of {want:.3f}")
            ok = False
        if got > 1.0:
            print(f"GATE FAIL {label}: {name} {got:.3f} exceeds 1: more light than the glass blocks")
            ok = False
    if ok:
        print(f"GATE PASS {label}")
    return ok


def _vec(text):
    return tuple(float(v) for v in text.split(","))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye", type=_vec, required=True)
    p.add_argument("--lookat", type=_vec, required=True)
    p.add_argument("--up", type=_vec, default=(0.0, 1.0, 0.0))
    p.add_argument("--size", type=float, required=True, help="edge length of the object's cube")
    p.add_argument("--floor-y", type=float, required=True)
    p.add_argument("--extent", type=float, required=True, help="half-width of the floor quad")
    p.add_argument("--triple", nargs=4, action="append", metavar=("LABEL", "BARE", "OPAQUE", "GLASS"),
                   default=[], help="one renderer's three PFMs (repeatable)")
    p.add_argument("--delta", nargs=5, action="append", metavar=("LABEL", "BARE", "OPAQUE", "WITH", "WITHOUT"),
                   default=[], help="WITH - WITHOUT relative to what OPAQUE blocks (repeatable)")
    p.add_argument("--gate", nargs=4, action="append", metavar=("LABEL", "FP", "ALL", "REL_TOL"), default=[],
                   help="fail (exit 1) unless LABEL's fp and all are within REL_TOL (a fraction) of "
                        "FP and ALL, and neither exceeds 1 (the glass cannot return more light than "
                        "it blocks); repeatable")
    args = p.parse_args(argv)

    areas = None
    results = {}
    print(f"{'renderer':<28}{'fp px':>6}{'fp':>7}{'all':>7}{'fp R/G/B':>22}{'Tmean':>7}{'Tp99':>7}{'Tmax':>7}")
    rows = [(l, b, o, g, None) for l, b, o, g in args.triple] + args.delta
    for label, bare_path, opaque_path, glass_path, without_path in rows:
        width, height, bare = load_pfm(bare_path)
        _, _, opaque = load_pfm(opaque_path)
        _, _, glass = load_pfm(glass_path)
        without = load_pfm(without_path)[2] if without_path else None
        if areas is None:
            camera = Camera(args.eye, args.lookat, args.up, width, height)
            areas = pixel_floor_areas(camera, args.floor_y, args.size / 2, args.extent)
        s = fractions(bare, opaque, glass, areas, without)
        results[label] = s
        rgb = "/".join(f"{v:.2f}" for v in s["fp_rgb"])
        print(f"{label:<28}{s['footprint_pixels']:>6}{s['fp']:>7.3f}{s['all']:>7.3f}{rgb:>22}"
              f"{s['t_mean']:>7.2f}{s['t_p99']:>7.2f}{s['t_max']:>7.2f}")
        if s["glass_max"] >= 0.99:
            print(f"  note: {label} has pixels at 1.0, clipped values understate T there", file=sys.stderr)
    return 0 if all(_gate_passes(results[label], float(fp), float(al), float(tol), label)
                    for label, fp, al, tol in args.gate) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
