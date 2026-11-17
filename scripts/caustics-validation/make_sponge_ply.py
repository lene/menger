#!/usr/bin/env python3
"""Write the surface of a Menger sponge (the solid's boundary) as an ASCII PLY triangle mesh.

This is the pbrt twin of menger's `type=sponge-volume`: the boundary of the union of the kept
sub-cubes, outward-facing, centred on the origin. A face between two kept sub-cubes is not part
of the surface (a glass solid has no wall there), exactly as `SpongeByVolume` drops it.

Usage: make_sponge_ply.py <level> <size> <out.ply>
"""
import itertools
import sys


def kept_voxels(level):
    """Voxel indices (x, y, z) of the solid on the 3**level grid: a cell is kept unless at some
    scale two of its three base-3 digits are 1 (the removed centre and face centres)."""
    n = 3 ** level

    def is_kept(cell):
        for _ in range(level):
            digits = [c % 3 for c in cell]
            if sum(1 for d in digits if d == 1) >= 2:
                return False
            cell = [c // 3 for c in cell]
        return True

    return {c for c in itertools.product(range(n), repeat=3) if is_kept(c)}


# Per axis direction: the four corner offsets of the face, counter-clockwise seen from outside.
FACES = {
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
}


def surface_mesh(level, size):
    """(vertices, triangles) of the sponge surface, edge length `size`, centred on the origin."""
    voxels = kept_voxels(level)
    n = 3 ** level
    cell = size / n
    half = size / 2.0
    vertices, triangles = [], []
    for (x, y, z) in sorted(voxels):
        for (dx, dy, dz), corners in FACES.items():
            if (x + dx, y + dy, z + dz) in voxels:
                continue
            base = len(vertices)
            for (ox, oy, oz) in corners:
                vertices.append(((x + ox) * cell - half, (y + oy) * cell - half, (z + oz) * cell - half))
            triangles.append((base, base + 1, base + 2))
            triangles.append((base, base + 2, base + 3))
    return vertices, triangles


def write_ply(path, vertices, triangles):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]:.7g} {v[1]:.7g} {v[2]:.7g}\n")
        for t in triangles:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def main(argv):
    if len(argv) != 4:
        sys.exit(__doc__)
    level, size, out = int(argv[1]), float(argv[2]), argv[3]
    vertices, triangles = surface_mesh(level, size)
    write_ply(out, vertices, triangles)
    print(f"{out}: level {level}, size {size}, {len(triangles)} triangles")


if __name__ == "__main__":
    main(sys.argv)
