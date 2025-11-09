# OptiX Physics

Physics models for transparent sphere rendering.

## Fresnel Reflectance

Surface reflectance via **Schlick approximation** of Fresnel equations:

```
R₀ = ((n₁ - n₂) / (n₁ + n₂))²
R(θ) = R₀ + (1 - R₀)(1 - cos θ)⁵
```

**Variables:**
- `n₁` = IOR of surrounding medium (1.0 for air/vacuum)
- `n₂` = IOR of sphere material (from `--ior` parameter)
- `θ` = angle between ray direction and surface normal
- `R₀` = reflectance at perpendicular incidence (0°)
- `R(θ)` = reflectance at angle θ

**Examples:**
- Glass (IOR 1.5): R₀ = 0.04 (4% perpendicular → 100% grazing)
- Water (IOR 1.33): R₀ = 0.02 (2% perpendicular)
- Diamond (IOR 2.42): R₀ = 0.17 (17% perpendicular)

## Beer-Lambert Law (Volume Absorption)

Exponential intensity decrease through absorbing medium:

```
I(d) = I₀ · exp(-α · d)
```

**Variables:**
- `I₀` = initial light intensity
- `d` = distance traveled through medium
- `α` = absorption coefficient (from color alpha + RGB)

**Color interpretation:**
- **Alpha**: Opacity/absorption strength (0.0 = transparent/no absorption, 1.0 = opaque/max absorption)
- **RGB**: Wavelength-dependent absorption (color tint)

**Examples:**
- `#00ff8080` (semi-transparent green-cyan): RGB (0,255,128) = green-cyan tint, alpha 0.5 = 50% opacity
- `#ffffffff` = no tint, fully opaque
- `#ffffff00` = no tint, fully transparent

**Implementation:**
```
α_r = -log(color.r) * color.a
α_g = -log(color.g) * color.a
α_b = -log(color.b) * color.a
```
Where `color.a` is alpha/opacity (0.0 = transparent, 1.0 = opaque)

Apply Beer-Lambert during ray traversal through sphere.

## Snell's Law (Refraction)

Ray refraction at sphere entry/exit:

```
n₁ · sin(θ₁) = n₂ · sin(θ₂)
```

**Variables:**
- `θ₁` = angle of incidence
- `θ₂` = angle of refraction
- `n₁`, `n₂` = IORs of two media

**Refraction direction:**
```
r = η · d + (η · cos(θ₁) - cos(θ₂)) · n
```

**Variables:**
- `r` = refracted ray direction
- `d` = incident ray direction
- `n` = surface normal
- `η = n₁ / n₂` = relative IOR
