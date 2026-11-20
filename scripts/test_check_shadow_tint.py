#!/usr/bin/env python3
"""Tests for check_shadow_tint.py on synthetic images (no GPU, no renderer). Run: python3 -m unittest"""
import unittest

import check_shadow_tint as tint

WIDTH, HEIGHT = 60, 40
FLOOR = (150, 150, 150)
SPHERE = (200, 30, 30)
SPHERE_CENTRE, SPHERE_RADIUS = (10, 10), 6
SHADOW_BOX = (20, 50, 20, 36)  # x0, x1, y0, y1: a 30x16 patch on the floor, clear of the sphere


def image(shadow_colour=None, sphere_colour=SPHERE):
    rows = [[FLOOR] * WIDTH for _ in range(HEIGHT)]
    cy, cx = SPHERE_CENTRE
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (y - cy) ** 2 + (x - cx) ** 2 <= SPHERE_RADIUS ** 2 and sphere_colour:
                rows[y][x] = sphere_colour
    if shadow_colour:
        x0, x1, y0, y1 = SHADOW_BOX
        for y in range(y0, y1):
            for x in range(x0, x1):
                rows[y][x] = shadow_colour
    return rows


class ShadowTintTest(unittest.TestCase):
    def check(self, scene, channel="R", **kwargs):
        return tint.check(scene, image(), image(sphere_colour=None), channel, **kwargs)

    def test_a_red_shadow_passes_as_red(self):
        passed, message = self.check(image((100, 40, 40)))
        self.assertTrue(passed, message)

    def test_the_region_is_the_patch_and_its_blurred_rim(self):
        patch = 30 * 16
        pixels = tint.shadow_pixels(image((100, 40, 40)), image(), image(sphere_colour=None))
        self.assertGreaterEqual(len(pixels), patch)
        self.assertLess(len(pixels), 1.5 * patch)

    def test_the_same_shadow_fails_as_another_channel(self):
        self.assertFalse(self.check(image((100, 40, 40)), channel="G")[0])

    def test_a_grey_shadow_is_not_tinted(self):
        passed, message = self.check(image((70, 70, 70)))
        self.assertFalse(passed)
        self.assertIn("need more than", message)

    def test_a_weakly_tinted_shadow_fails(self):
        self.assertFalse(self.check(image((80, 60, 60)))[0])  # 1.33x

    def test_no_shadow_fails_on_the_pixel_count(self):
        passed, message = self.check(image())
        self.assertFalse(passed)
        self.assertIn("only 0 shadow pixels", message)

    def test_a_shadow_smaller_than_the_minimum_fails(self):
        self.assertFalse(self.check(image((100, 40, 40)), min_pixels=10_000)[0])

    def test_the_dark_tinted_sphere_is_not_counted_as_shadow(self):
        scene = image(sphere_colour=(60, 5, 5))  # darker than the floor, but the sphere
        self.assertFalse(self.check(scene)[0])

    def test_a_shadow_on_a_black_floor_square_is_not_counted(self):
        floor = [[(10, 10, 10)] * WIDTH for _ in range(HEIGHT)]
        scene = [[(5, 1, 1)] * WIDTH for _ in range(HEIGHT)]
        self.assertFalse(tint.check(scene, image(), floor, "R")[0])


class ReadPpmTest(unittest.TestCase):
    def test_parses_a_binary_ppm(self):
        data = b"P6\n2 1\n255\n" + bytes([1, 2, 3, 4, 5, 6])
        self.assertEqual(tint.read_ppm(data), [[(1, 2, 3), (4, 5, 6)]])

    def test_rejects_other_formats(self):
        with self.assertRaises(ValueError):
            tint.read_ppm(b"P3\n1 1\n255\n0 0 0\n")


if __name__ == "__main__":
    unittest.main()
