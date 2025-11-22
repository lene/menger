# Caustics Reference Images

Reference images and scenes for validating Progressive Photon Mapping (PPM) caustics implementation.

## Quick Start

1. **Render reference images** (requires PBRT v4, Mitsuba 3, or Appleseed):
   ```bash
   ./render-references.sh
   ```

2. **Copy reference to test resources**:
   ```bash
   cp output/canonical-caustics-pbrt.png ../../src/test/resources/caustics-reference.png
   ```

3. **Run validation tests**:
   ```bash
   sbt "testOnly *ReferenceMatchSpec"
   ```

## Directory Structure

```
caustics-references/
├── appleseed/                    # Appleseed Cornell box scene
│   └── cornell-box-caustics.appleseed
├── mitsuba/                      # Mitsuba tutorial
│   └── caustics_optimization.ipynb
├── pbrt/                         # PBRT v4 scenes
│   └── pbrt-v4-scenes/
│       ├── crown/                # Gems with caustics
│       ├── lte-orb/              # Spherical test scenes
│       └── transparent-machines/ # Complex glass
├── renders/                      # Our canonical scene
│   └── canonical-caustics.pbrt   # Primary reference scene
├── output/                       # Rendered images (gitignored)
├── render-references.sh          # Render script
└── README.md                     # This file
```

## Canonical Test Scene

The primary reference scene (`renders/canonical-caustics.pbrt`) matches the parameters defined
in the arc42 quality requirements:

| Parameter | Value |
|-----------|-------|
| Sphere center | (0, 0, 0) |
| Sphere radius | 1.0 |
| Sphere IOR | 1.5 |
| Floor position | Y = -2.0 |
| Light position | (0, 10, 0) |
| Camera position | (0, 1, 4) |
| Resolution | 800 × 600 |

**Expected Result:** Circular caustic centered at (0, -2, 0) with radius ~0.3 units.

## Renderer Installation

### PBRT v4 (Recommended)
```bash
git clone https://github.com/mmp/pbrt-v4.git
cd pbrt-v4
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### Mitsuba 3
```bash
pip install mitsuba
```

### Appleseed
Download from https://appleseedhq.net/download.html

## Documentation

- [CAUSTICS_REFERENCES.md](../CAUSTICS_REFERENCES.md) - Full reference documentation
- [CAUSTICS_TEST_LADDER.md](../CAUSTICS_TEST_LADDER.md) - Test validation framework
- [arc42 Section 10](../../docs/arc42/10-quality-requirements.md) - Quality requirements

## Validation Workflow

```
1. Render reference with PBRT → canonical-caustics-reference.png
2. Render test image with our implementation → test-caustics.png
3. Compare using SSIM → should be > 0.90
```

## Troubleshooting

**"Reference image not found"**: Run `./render-references.sh pbrt` first.

**"SSIM too low"**: Check:
- IOR matches (should be 1.5)
- Light position matches
- Camera position matches
- Resolution matches

**"PBRT not found"**: Install PBRT v4 from source or use Mitsuba instead.
