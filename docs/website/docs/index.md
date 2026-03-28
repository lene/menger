# Menger Ray Tracer

GPU-accelerated ray tracer for 3D and 4D objects, built with NVIDIA OptiX and Scala 3.

## Features

- **3D Menger sponges** — cube sponges via surface or volume subdivision, fractional levels with alpha blending
- **4D objects** — tesseracts and tesseract sponges projected into 3D with real-time rotation
- **Physically-based materials** — glass, metallic, thin-film interference, caustics
- **Parametric surfaces** — torus, Klein bottle, Möbius strip, and user-defined surfaces
- **Interactive exploration** — orbit camera, 4D rotation controls, live parameter adjustment
- **Animation** — frame sequences and fractional-level transitions

## Quick Start

Requires an NVIDIA GPU with CUDA support.

Download the latest release from the [releases page](https://gitlab.com/lilacashes/menger/-/releases), then:

```bash
unzip menger-app-*.zip
./menger-app-*/bin/menger-app --help
```

Render a Menger sponge:

```bash
./menger-app-*/bin/menger-app --optix --sponge-type cube-sponge --level 3 --save-name output.png
```

## Links

- [Source code (GitLab)](https://gitlab.com/lilacashes/menger)
- [Mirror (GitHub)](https://github.com/lene/menger)
- [Releases](https://gitlab.com/lilacashes/menger/-/releases)
- [User Guide](https://gitlab.com/lilacashes/menger/-/blob/main/docs/guide/user-guide.md)
