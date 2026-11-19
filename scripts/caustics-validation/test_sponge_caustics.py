#!/usr/bin/env python3
"""Tests for make_sponge_ply.py and sponge_caustics.py (no GPU, no pbrt). Run: python3 -m unittest"""
import math
import os
import struct
import tempfile
import unittest
from collections import Counter

import make_sponge_ply as ply
import sponge_caustics as sc


def write_pfm(path, width, height, rows_top_down):
    with open(path, "wb") as f:
        f.write(f"PF\n{width} {height}\n-1.0\n".encode())
        for row in reversed(rows_top_down):  # PFM: bottom row first
            for r, g, b in row:
                f.write(struct.pack("<3f", r, g, b))


class SpongeMeshTest(unittest.TestCase):
    def test_triangle_counts_match_menger_sponge_volume(self):
        # SpongeByVolume.triangleCount: 12, 144, 2112 for levels 0..2
        for level, expected in ((0, 12), (1, 144), (2, 2112)):
            _, triangles = ply.surface_mesh(level, 1.0)
            self.assertEqual(len(triangles), expected)

    def test_surface_is_closed_and_faces_outward(self):
        for level in (1, 2):
            vertices, triangles = ply.surface_mesh(level, 0.6)
            edges = Counter()
            volume = 0.0
            for a, b, c in triangles:
                for u, v in ((a, b), (b, c), (c, a)):
                    edges[tuple(sorted((tuple(vertices[u]), tuple(vertices[v]))))] += 1
                pa, pb, pc = vertices[a], vertices[b], vertices[c]
                volume += (pa[0] * (pb[1] * pc[2] - pb[2] * pc[1])
                           - pa[1] * (pb[0] * pc[2] - pb[2] * pc[0])
                           + pa[2] * (pb[0] * pc[1] - pb[1] * pc[0])) / 6.0
            # Each edge is shared by an even number of triangles (a closed surface; quad
            # diagonals count twice, edges where two cubes touch at a corner four times).
            self.assertTrue(all(n % 2 == 0 for n in edges.values()))
            # Outward-facing triangles enclose a positive volume: the solid's, (20/27)^level.
            self.assertAlmostEqual(volume, 0.6 ** 3 * (20 / 27) ** level, places=6)

    def test_ply_header_counts_match_body(self):
        vertices, triangles = ply.surface_mesh(1, 0.6)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.ply")
            ply.write_ply(path, vertices, triangles)
            with open(path) as f:
                lines = f.read().splitlines()
        body = lines[lines.index("end_header") + 1:]
        self.assertEqual(len(body), len(vertices) + len(triangles))
        self.assertIn(f"element face {len(triangles)}", lines)


class FractionsTest(unittest.TestCase):
    EYE, LOOKAT, UP = (0.0, 3.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, -1.0)

    def setUp(self):
        self.w, self.h = 40, 30
        camera = sc.Camera(self.EYE, self.LOOKAT, self.UP, self.w, self.h)
        self.areas = sc.pixel_floor_areas(camera, 0.0, box_half=0.0, extent=100.0)

    def image(self, value_at):
        return [value_at(i % self.w, i // self.w) for i in range(self.w * self.h)]

    def grey(self, v):
        return (v, v, v)

    def test_floor_areas_add_up_to_the_visible_floor(self):
        half_w = 3.0 * math.tan(math.radians(sc.HORIZONTAL_FOV_DEG) / 2)
        visible = (2 * half_w) * (2 * half_w * self.h / self.w)
        self.assertAlmostEqual(sum(a for a in self.areas if a is not None), visible, delta=1e-6)

    def test_object_box_pixels_are_left_out(self):
        camera = sc.Camera(self.EYE, self.LOOKAT, self.UP, self.w, self.h)
        areas = sc.pixel_floor_areas(camera, 0.0, box_half=0.5, extent=100.0)
        self.assertIsNone(areas[(self.h // 2) * self.w + self.w // 2])
        self.assertIsNotNone(areas[0])

    def test_glass_returning_half_the_blocked_light_gives_one_half(self):
        in_shadow = lambda x, y: 10 <= x < 20 and 10 <= y < 20
        bare = self.image(lambda x, y: self.grey(1.0))
        opaque = self.image(lambda x, y: self.grey(0.0 if in_shadow(x, y) else 1.0))
        glass = self.image(lambda x, y: self.grey(0.5 if in_shadow(x, y) else 1.0))
        s = sc.fractions(bare, opaque, glass, self.areas)
        self.assertEqual(s["footprint_pixels"], 100)
        self.assertAlmostEqual(s["fp"], 0.5, places=6)
        self.assertAlmostEqual(s["all"], 0.5, places=6)
        self.assertAlmostEqual(s["t_max"], 0.5, places=6)

    def test_light_thrown_outside_the_footprint_counts_in_all_only(self):
        in_shadow = lambda x, y: 10 <= x < 20 and 10 <= y < 20
        bare = self.image(lambda x, y: self.grey(1.0 if in_shadow(x, y) else 0.5))
        opaque = self.image(lambda x, y: self.grey(0.0 if in_shadow(x, y) else 0.5))
        glass = self.image(lambda x, y: self.grey(0.25 if in_shadow(x, y) else 0.5 + (0.1 if x >= 30 else 0)))
        s = sc.fractions(bare, opaque, glass, self.areas)
        self.assertAlmostEqual(s["fp"], 0.25, places=6)
        self.assertGreater(s["all"], s["fp"])

    def test_without_replaces_the_opaque_baseline(self):
        in_shadow = lambda x, y: 10 <= x < 20 and 10 <= y < 20
        bare = self.image(lambda x, y: self.grey(1.0))
        opaque = self.image(lambda x, y: self.grey(0.0 if in_shadow(x, y) else 1.0))
        without = self.image(lambda x, y: self.grey(0.6 if in_shadow(x, y) else 1.0))
        with_ = self.image(lambda x, y: self.grey(0.8 if in_shadow(x, y) else 1.0))
        s = sc.fractions(bare, opaque, with_, self.areas, without)
        self.assertAlmostEqual(s["fp"], 0.2, places=6)

    def test_no_shadow_is_an_error(self):
        flat = self.image(lambda x, y: self.grey(1.0))
        with self.assertRaises(ValueError):
            sc.fractions(flat, flat, flat, self.areas)

    def test_gate_passes_within_tolerance_and_fails_outside_or_above_one(self):
        stats = {"fp": 0.40, "all": 0.60}
        self.assertTrue(sc._gate_passes(stats, 0.42, 0.62, 0.1, "x"))
        self.assertFalse(sc._gate_passes(stats, 0.60, 0.62, 0.1, "x"))
        self.assertFalse(sc._gate_passes({"fp": 1.2, "all": 1.2}, 1.2, 1.2, 0.1, "x"))

    def test_footprint_only_gate_ignores_the_whole_floor(self):
        # Without caustics nothing lands outside the footprint: "all" equals "fp" there, and a
        # gate on it would only repeat the footprint check against a number pbrt's does not match.
        stats = {"fp": 0.40, "all": 0.40}
        self.assertTrue(sc._gate_passes(stats, 0.42, None, 0.1, "x"))
        self.assertFalse(sc._gate_passes(stats, 0.60, None, 0.1, "x"))
        self.assertFalse(sc._gate_passes({"fp": 1.2, "all": 1.2}, 1.2, None, 0.1, "x"))

    def test_pfm_round_trip_keeps_the_top_row_first(self):
        rows = [[(float(x), float(y), 0.0) for x in range(3)] for y in range(2)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.pfm")
            write_pfm(path, 3, 2, rows)
            w, h, px = sc.load_pfm(path)
        self.assertEqual((w, h), (3, 2))
        self.assertEqual(px[0], (0.0, 0.0, 0.0))
        self.assertEqual(px[3], (0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
