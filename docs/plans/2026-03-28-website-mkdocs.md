# MkDocs Website (Task 16.4) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a project website for Menger using MkDocs, hosted on GitLab Pages, with a full render gallery and feedback buttons linking to both GitHub and GitLab issues.

**Architecture:** MkDocs source lives in `docs/website/` with its own `mkdocs.yml`. A new `pages:website` CI job builds the site with `mkdocs build` and publishes the output to the `public/` artifact. The existing `pages` job (scoverage report) is left unchanged. Selected showcase renders from `scripts/reference-images/` are copied into `docs/website/docs/images/`.

**Tech Stack:** MkDocs with Material theme, GitLab CI `pages:website` job, Python 3, pip.

---

### Task A: MkDocs configuration and site skeleton

**Files:**
- Create: `docs/website/mkdocs.yml`
- Create: `docs/website/docs/index.md`
- Create: `docs/website/docs/gallery.md`
- Create: `docs/website/docs/feedback.md`

**Step 1: Create `docs/website/mkdocs.yml`**

```yaml
site_name: Menger Ray Tracer
site_description: GPU-accelerated ray tracer for 3D and 4D objects using NVIDIA OptiX
site_url: https://lilacashes.gitlab.io/menger/
repo_url: https://gitlab.com/lilacashes/menger
repo_name: GitLab

theme:
  name: material
  palette:
    scheme: slate
    primary: indigo
    accent: cyan
  features:
    - navigation.instant
    - navigation.top

nav:
  - Home: index.md
  - Gallery: gallery.md
  - Feedback: feedback.md

markdown_extensions:
  - attr_list
  - md_in_html
```

**Step 2: Create `docs/website/docs/index.md`**

```markdown
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

```bash
# Install (requires CUDA-capable GPU)
# Download the latest release from https://gitlab.com/lilacashes/menger/-/releases
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
```

**Step 3: Create `docs/website/docs/gallery.md`**

```markdown
# Gallery

Sample renders produced by Menger with NVIDIA OptiX ray tracing.

## Menger Sponge

![Menger sponge with anti-aliasing](images/save_with_AA.png)
*Level-3 cube sponge with 4× anti-aliasing*

## Sphere

![Ray-traced sphere](images/headless_sphere.png)
*Glass sphere with reflections and refraction*

## Cube

![Ray-traced cube](images/headless_cube.png)
*Cube with directional lighting*

## Tesseract (4D → 3D projection)

![Tesseract wireframe](images/headless_tesseract.png)
*4D hypercube projected into 3D*

## Caustics

![Caustics reference](images/pbrt-reference.png)
*Caustic patterns: physically-based reference render*
```

**Step 4: Create `docs/website/docs/feedback.md`**

```markdown
# Feedback & Issues

Found a bug? Have a feature request? We'd love to hear from you.

## Report an Issue

Choose your preferred platform:

- **[Open an issue on GitLab](https://gitlab.com/lilacashes/menger/-/issues/new)** — primary repository
- **[Open an issue on GitHub](https://github.com/lene/menger/issues/new)** — mirror repository

## Contributing

Pull requests are welcome on GitLab. Please read the contributing guidelines in `AGENTS.md` before submitting.
```

**Step 5: Copy showcase images into website docs**

```bash
mkdir -p docs/website/docs/images
cp scripts/reference-images/save_with_AA.png docs/website/docs/images/
cp scripts/reference-images/headless_sphere.png docs/website/docs/images/
cp scripts/reference-images/headless_cube.png docs/website/docs/images/
cp scripts/reference-images/headless_tesseract.png docs/website/docs/images/
cp optix-jni/test-resources/caustics-references/pbrt-reference.png docs/website/docs/images/
```

**Step 6: Test locally (optional)**

```bash
cd docs/website
pip install mkdocs-material
mkdocs serve
# Browse to http://localhost:8000
```

**Step 7: Commit**

```bash
git add docs/website/
git commit -m "feat(website): add MkDocs site skeleton with gallery and feedback pages"
```

---

### Task B: GitLab CI pages:website job

**Files:**
- Modify: `.gitlab-ci.yml`

**Step 1: Locate the existing `pages` job in `.gitlab-ci.yml`**

Find it (around line 490) and note its stage, needs, rules, and artifact structure.

**Step 2: Add `pages:website` job after the existing `pages` job**

The new job must use `pages` stage (GitLab Pages requires a job named exactly `pages` that writes to `public/`). Since the existing `pages` job already uses that name, the new job must be named `pages` and must subsume the scoverage publishing too.

Strategy: rename the existing `pages` job to `pages:coverage` (non-deploying, artifact only for download), then create a new `pages` job that:
1. Installs MkDocs
2. Builds the site into `public/`
3. Copies scoverage report into `public/coverage/` (needs CheckCoverage artifact)

New job definition:

```yaml
pages:
  image: python:3.12-slim
  stage: package
  needs:
    - job: CheckCoverage
      artifacts: true
    - ScalaVersionConsistent
  rules:
    - if: "$CI_MERGE_REQUEST_ID"
    - if: "$CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH"
  script:
    - pip install mkdocs-material --quiet
    - mkdocs build --config-file docs/website/mkdocs.yml --site-dir public
    - mkdir -p public/coverage
    - mv menger-app/target/scala-${SCALA_VERSION}/scoverage-report/* public/coverage/
  artifacts:
    paths:
      - public
```

Old `pages` job becomes:

```yaml
pages:coverage:
  image: ubuntu:latest
  stage: package
  needs:
    - CheckCoverage
    - ScalaVersionConsistent
  rules:
    - if: "$CI_MERGE_REQUEST_ID"
  script:
    - mv menger-app/target/scala-${SCALA_VERSION}/scoverage-report/ coverage-report/
  artifacts:
    paths:
      - coverage-report
```

**Step 3: Verify CI syntax**

```bash
# Use GitLab CI lint if GITLAB_ACCESS_TOKEN is available
# Or simply confirm the yaml is valid
python3 -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))"
```

**Step 4: Commit**

```bash
git add .gitlab-ci.yml
git commit -m "ci: add pages:website job for MkDocs site on GitLab Pages"
```

---

### Task C: Update CHANGELOG.md

Add to `[Unreleased]` section under `### Added`:

```
- Project website at https://lilacashes.gitlab.io/menger/ — MkDocs site with render gallery and feedback links
```

**Commit:**
```bash
git add CHANGELOG.md
git commit -m "docs: note website in CHANGELOG"
```

---

### Verification

After pushing to a feature branch:
1. Check GitLab CI for `pages:website` job — it should succeed
2. GitLab Pages URL: `https://lilacashes.gitlab.io/menger/`
3. Coverage report sub-path: `https://lilacashes.gitlab.io/menger/coverage/`
