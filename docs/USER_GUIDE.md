# Menger User Guide

**Version**: 0.9.0
**Last Updated**: September 2026

Menger is a 3D/4D fractal visualization and GPU ray tracing tool. This index links to each part
of the documentation.

---

## Documentation Map

| Document | Contents | Audience |
|----------|----------|----------|
| [Quick Start](guide/quickstart.md) | What Menger is, system requirements, installation, first render | New users |
| [Usage & Rendering](guide/user-guide.md) | CLI options, interactive controls, rendering modes, geometry types, materials, lighting | Regular users |
| [Advanced Features](guide/advanced.md) | `--animate` sweeps, animated DSL scenes (t-parameter), video output, animation preview, interactive session control, caustics (PPM), antialiasing, multi-object scenes | Power users |
| [Scala DSL Reference](guide/dsl-reference.md) | Type-safe scene description language, all DSL types, parametric surfaces | DSL / developers |
| [Tutorials](guide/tutorials.md) | Step-by-step walkthroughs: first render, glass, animation, 4D, complex scenes | All users |
| [Cloud GPU Development](guide/cloud.md) | AWS EC2 spot instances: setup, launch, renders, state management, cost control | Cloud users |
| [Troubleshooting](TROUBLESHOOTING.md) | Common errors, performance tips, getting help | All users |
| [Reference](guide/reference.md) | Complete CLI option list, every `--objects` key, keyboard shortcuts, file formats | All users |

---

## Quick Links

- **Install and first render** → [Quick Start](guide/quickstart.md#quick-start)
- **CLI option list** → [Reference](guide/reference.md#complete-option-list)
- **Glass/material setup** → [Usage & Rendering](guide/user-guide.md#materials-and-lighting)
- **Define a scene in Scala** → [DSL Reference](guide/dsl-reference.md)
- **Caustics rendering** → [Advanced Features](guide/advanced.md#caustics-light-focusing-effects)
- **Window vs. headless vs. video output** → [Usage & Rendering](guide/user-guide.md#rendering-modes)
- **4D objects and rotation** → [Tutorials](guide/tutorials.md#tutorial-4-4d-visualization)
- **Render a video** → [Advanced Features](guide/advanced.md#video-output)
- **Upgrading from 0.8.x (`--optix` removed)** → [Usage & Rendering](guide/user-guide.md#running-the-application)
- **GPU cloud rendering** → [Cloud GPU Development](guide/cloud.md)

---

For questions, issues, or contributions:
- GitHub: https://github.com/lene/menger (GitLab `lilacashes/menger` is a read-only mirror)
- Issues: https://github.com/lene/menger/issues
