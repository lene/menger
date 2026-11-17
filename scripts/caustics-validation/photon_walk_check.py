#!/usr/bin/env python3
"""How much of the light that hits a glass sponge reaches the floor, by a CPU replica of the
photon walk in optix-jni's caustics_ppm.cu, for several photon bounce caps (menger#150).

Photons start parallel to the light across the sponge's cross-section; only those that meet the
sponge are counted (that is the light an opaque sponge blocks). At every surface a photon reflects
with probability F (exact unpolarised Fresnel, total internal reflection included) or refracts,
carrying its full flux either way, as __closesthit__photon does. A photon that leaves the sponge
and flies down onto the floor quad counts as returned. The cap is the number of traced segments
per photon (MAX_PHOTON_BOUNCES in OptiXData.h): a photon still inside after the last one is lost.

The result is the "all" fraction of sponge_caustics.py for an ideal gather over the whole floor,
so it is what a correct renderer with that cap should return, free of the camera, the gather
radius, the tone map and the units of the direct light.

Usage: photon_walk_check.py <bounce cap>... [--photons N] [--seed S]
Needs numpy.
"""
import argparse

import numpy as np

import make_sponge_ply as ply

LEVEL, SIZE, IOR = 2, 0.6, 1.5
LIGHT = np.array([0.3, -1.0, 0.6])
FLOOR_Y, FLOOR_HALF_EXTENT = -0.5, 2.0
RAY_OFFSET = 1e-3          # CONTINUATION_RAY_OFFSET
START_DISTANCE = 3.0
CROSS_SECTION_RADIUS = SIZE * 0.9   # covers the cube's projection (half diagonal 0.52)
BATCH = 4000               # photons per intersection batch: the memory of photons x triangles


class Mesh:
    def __init__(self):
        vertices, triangles = (np.array(a) for a in ply.surface_mesh(LEVEL, SIZE))
        self.origin = vertices[triangles[:, 0]]
        self.edge1 = vertices[triangles[:, 1]] - self.origin
        self.edge2 = vertices[triangles[:, 2]] - self.origin
        normals = np.cross(self.edge1, self.edge2)
        self.normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    def nearest(self, origins, directions):
        """(distance, triangle index) of each ray's nearest hit; inf where it hits nothing.
        Moeller-Trumbore against every triangle."""
        h = np.cross(directions[:, None, :], self.edge2[None])
        a = np.einsum("tk,ptk->pt", self.edge1, h)
        usable = np.abs(a) > 1e-12
        f = np.where(usable, 1 / np.where(usable, a, 1), 0)
        s = origins[:, None, :] - self.origin[None]
        u = f * np.einsum("ptk,ptk->pt", s, h)
        q = np.cross(s, self.edge1[None])
        v = f * np.einsum("pk,ptk->pt", directions, q)
        t = f * np.einsum("tk,ptk->pt", self.edge2, q)
        hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > RAY_OFFSET)
        t = np.where(hit, t, np.inf)
        index = t.argmin(axis=1)
        return t[np.arange(len(origins)), index], index

    def nearest_batched(self, origins, directions):
        parts = [self.nearest(origins[i:i + BATCH], directions[i:i + BATCH])
                 for i in range(0, len(origins), BATCH)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def fresnel_reflectance(cos_i, n_i, n_t):
    """(R, cos of the refraction angle); R is 1 under total internal reflection."""
    sin_t_sq = (n_i / n_t) ** 2 * (1 - cos_i ** 2)
    cos_t = np.sqrt(np.clip(1 - sin_t_sq, 0, None))
    r_parallel = (n_t * cos_i - n_i * cos_t) / (n_t * cos_i + n_i * cos_t)
    r_perpendicular = (n_i * cos_i - n_t * cos_t) / (n_i * cos_i + n_t * cos_t)
    return np.where(sin_t_sq > 1, 1.0, 0.5 * (r_parallel ** 2 + r_perpendicular ** 2)), cos_t


def start_rays(mesh, count, rng):
    light = LIGHT / np.linalg.norm(LIGHT)
    tangent = np.cross(light, [0, 1, 0])
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(light, tangent)
    offsets = (rng.random((count, 2)) * 2 - 1) * CROSS_SECTION_RADIUS
    origins = -light * START_DISTANCE + offsets[:, :1] * tangent + offsets[:, 1:] * bitangent
    directions = np.tile(light, (count, 1))
    distance, _ = mesh.nearest_batched(origins, directions)
    meets = np.isfinite(distance)
    return origins[meets], directions[meets]


def lands_on_floor(origin, direction):
    if direction[1] >= 0:
        return False
    p = origin + (FLOOR_Y - origin[1]) / direction[1] * direction
    return abs(p[0]) <= FLOOR_HALF_EXTENT and abs(p[2]) <= FLOOR_HALF_EXTENT


def walk(mesh, origins, directions, cap, rng):
    """Fractions of the photons that reach the floor, are cut off by the cap, or go elsewhere."""
    count = len(origins)
    alive = np.ones(count, bool)
    touched = np.zeros(count, bool)
    returned = elsewhere = 0
    for _ in range(cap):
        active = np.where(alive)[0]
        if len(active) == 0:
            break
        distance, triangle = mesh.nearest_batched(origins[active], directions[active])
        missed = ~np.isfinite(distance)
        for k in active[missed]:
            alive[k] = False
            if touched[k] and lands_on_floor(origins[k], directions[k]):
                returned += 1
            else:
                elsewhere += 1
        hit = ~missed
        photon = active[hit]
        if len(photon) == 0:
            continue
        d = directions[photon]
        entering = np.einsum("pk,pk->p", d, mesh.normals[triangle[hit]]) < 0
        normal = np.where(entering[:, None], mesh.normals[triangle[hit]], -mesh.normals[triangle[hit]])
        cos_i = np.abs(np.einsum("pk,pk->p", d, normal))
        n_i, n_t = np.where(entering, 1.0, IOR), np.where(entering, IOR, 1.0)
        reflectance, cos_t = fresnel_reflectance(cos_i, n_i, n_t)
        eta = n_i / n_t
        reflected = d + 2 * cos_i[:, None] * normal
        refracted = eta[:, None] * d + (eta * cos_i - cos_t)[:, None] * normal
        new_d = np.where((rng.random(len(photon)) < reflectance)[:, None], reflected, refracted)
        new_d /= np.linalg.norm(new_d, axis=1)[:, None]
        point = origins[photon] + d * distance[hit][:, None]
        origins[photon] = point + new_d * RAY_OFFSET
        directions[photon] = new_d
        touched[photon] = True
    return returned / count, alive.sum() / count, elsewhere / count


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("caps", type=int, nargs="+")
    p.add_argument("--photons", type=int, default=40000, help="photons started (about half meet the sponge)")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    mesh = Mesh()
    for cap in args.caps:
        rng = np.random.default_rng(args.seed)
        origins, directions = start_rays(mesh, args.photons, rng)
        returned, cut, elsewhere = walk(mesh, origins.copy(), directions.copy(), cap, rng)
        print(f"cap {cap:>3}: {len(origins)} photons hit the sponge; reach the floor {returned:.3f}, "
              f"cut off by the cap {cut:.3f}, go elsewhere {elsewhere:.3f}")


if __name__ == "__main__":
    main()
