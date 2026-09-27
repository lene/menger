# Menger Ray Tracer

GPU-accelerated ray tracer for 3D and 4D objects, built with NVIDIA OptiX and Scala 3.

## Features

- **3D Menger sponges** — cube sponges via surface or volume subdivision, fractional levels with alpha blending
- **4D objects** — tesseracts and tesseract sponges projected into 3D with real-time rotation
- **Physically-based materials** — glass, metallic, thin-film interference, caustics
- **Parametric surfaces** — torus, Klein bottle, Möbius strip, and user-defined surfaces
- **Interactive exploration** — orbit camera, 4D rotation controls, live parameter adjustment
- **Animation** — frame sequences, direct MP4/MKV video output, fractional-level transitions
- **Scala DSL** — describe scenes in type-safe Scala, compiled into the app or loaded from a `.scala` file at startup

## Quick Start

Requires Linux and an NVIDIA RTX GPU with driver 580.65 or newer.

Download the latest release from the [releases page](https://github.com/lene/menger/releases), then:

```bash
unzip menger-*.zip
./menger-app-*/bin/menger-app --help
```

Render a Menger sponge:

```bash
./menger-app-*/bin/menger-app --objects type=cube-sponge:level=3 --headless --save-name output.png
```

## Links

- [Source code (GitHub)](https://github.com/lene/menger)
- [Mirror (GitLab)](https://gitlab.com/lilacashes/menger)
- [Releases](https://github.com/lene/menger/releases)
- [User Guide](https://github.com/lene/menger/blob/main/docs/USER_GUIDE.md)
