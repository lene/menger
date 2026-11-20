#!/usr/bin/env python3
"""Fails unless a colored shadow is visible on the floor of a render (lene/menger#195).

Usage: check_shadow_tint.py SCENE.png SCALAR.png FLOOR.png CHANNEL [--min-pixels N] [--min-ratio R]

SCENE   the render under test (colored shadows on)
SCALAR  the same scene with scalar shadows; only used to find the sphere's silhouette
FLOOR   the same scene without the object: the lit floor
CHANNEL R, G or B: the channel the shadow must be dominated by

The shadow region is every pixel outside the sphere where the (blurred) SCENE is at least
SHADOW_DROP darker than the (blurred) FLOOR, and the floor there is not black. It passes when the
region has at least --min-pixels pixels and its mean colour has CHANNEL above --min-ratio times the
mean of the other two channels. Reads images through ImageMagick, which the suite requires anyway.
"""
import argparse
import subprocess
import sys

CHANNELS = "RGB"
SHADOW_DROP = 0.8        # a shadow pixel is at most this fraction of the bare floor's brightness
MIN_LIT_BRIGHTNESS = 60  # sum of R+G+B; a darker floor pixel (a black checker square) shows no shadow
BLUR_RADIUS = 2          # averages out the area-light noise before comparing
SPHERE_MIN_CHROMA = 12   # max-min of the channels: the glass sphere is tinted, the floor is grey
SPHERE_MIN_DIFFERENCE = 12
SPHERE_MARGIN = 6        # pixels added around the sphere's circle: its rim, the blur's reach
DEFAULT_MIN_PIXELS = 300
DEFAULT_MIN_RATIO = 1.5


def read_ppm(data):
    """Pixels (rows of (r, g, b)) of a binary 8-bit PPM (P6)."""
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1  # the single whitespace after maxval
    if tokens[0] != b"P6" or int(tokens[3]) != 255:
        raise ValueError("expected an 8-bit binary PPM")
    width, height = int(tokens[1]), int(tokens[2])
    body = data[pos:pos + 3 * width * height]
    return [[tuple(body[3 * (y * width + x):3 * (y * width + x) + 3]) for x in range(width)]
            for y in range(height)]


def read_image(path):
    converted = subprocess.run(["convert", path, "-alpha", "off", "-depth", "8", "ppm:-"],
                               capture_output=True, check=True)
    return read_ppm(converted.stdout)


def blur_row(values, radius):
    windows = (values[max(0, i - radius):i + radius + 1] for i in range(len(values)))
    return [sum(window) / len(window) for window in windows]


def blur(image, radius=BLUR_RADIUS):
    """Box blur per channel, edges clamped."""
    height, width = len(image), len(image[0])
    channels = [[[px[c] for px in row] for row in image] for c in range(3)]
    result = []
    for rows in channels:
        horizontal = [blur_row(row, radius) for row in rows]
        columns = [blur_row([horizontal[y][x] for y in range(height)], radius) for x in range(width)]
        result.append([[columns[x][y] for x in range(width)] for y in range(height)])
    return [[(result[0][y][x], result[1][y][x], result[2][y][x]) for x in range(width)]
            for y in range(height)]


def brightness(px):
    return px[0] + px[1] + px[2]


def chroma(px):
    return max(px) - min(px)


def sphere_circle(scalar, floor):
    """(centre_y, centre_x, radius) around the tinted pixels that differ from the bare floor, or None."""
    hits = [(y, x) for y, row in enumerate(scalar) for x, px in enumerate(row)
            if chroma(px) > SPHERE_MIN_CHROMA
            and sum(abs(a - b) for a, b in zip(px, floor[y][x])) > SPHERE_MIN_DIFFERENCE]
    if not hits:
        return None
    ys, xs = [y for y, _ in hits], [x for _, x in hits]
    radius = max(max(ys) - min(ys), max(xs) - min(xs)) / 2 + SPHERE_MARGIN
    return (min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2, radius


def inside(circle, y, x):
    if circle is None:
        return False
    cy, cx, r = circle
    return (y - cy) ** 2 + (x - cx) ** 2 <= r * r


def shadow_pixels(scene, scalar, floor):
    """The scene's pixels in the shadow region: darker than the floor, outside the sphere."""
    circle = sphere_circle(scalar, floor)
    scene_blurred, floor_blurred = blur(scene), blur(floor)
    return [scene[y][x]
            for y in range(len(scene)) for x in range(len(scene[0]))
            if brightness(floor_blurred[y][x]) > MIN_LIT_BRIGHTNESS
            and brightness(scene_blurred[y][x]) < SHADOW_DROP * brightness(floor_blurred[y][x])
            and not inside(circle, y, x)]


def tint_ratio(pixels, channel):
    """Mean of CHANNEL over the mean of the other two, in the mean colour of PIXELS."""
    mean = [sum(px[c] for px in pixels) / len(pixels) for c in range(3)]
    k = CHANNELS.index(channel)
    others = [mean[c] for c in range(3) if c != k]
    return mean[k] / max(sum(others) / 2, 1e-6)


def check(scene, scalar, floor, channel, min_pixels=DEFAULT_MIN_PIXELS, min_ratio=DEFAULT_MIN_RATIO):
    """(passed, message)."""
    pixels = shadow_pixels(scene, scalar, floor)
    if len(pixels) < min_pixels:
        return False, f"only {len(pixels)} shadow pixels on the floor, need {min_pixels}"
    ratio = tint_ratio(pixels, channel)
    message = f"{len(pixels)} shadow pixels, {channel} is {ratio:.2f}x the other channels"
    return ratio > min_ratio, message + ("" if ratio > min_ratio else f", need more than {min_ratio}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene")
    parser.add_argument("scalar")
    parser.add_argument("floor")
    parser.add_argument("channel", choices=list(CHANNELS))
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--min-ratio", type=float, default=DEFAULT_MIN_RATIO)
    args = parser.parse_args(argv)
    passed, message = check(read_image(args.scene), read_image(args.scalar), read_image(args.floor),
                            args.channel, args.min_pixels, args.min_ratio)
    print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
