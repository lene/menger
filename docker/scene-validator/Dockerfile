# syntax=docker/dockerfile:1
#
# Renderer-side gauntlet validation image (spec-ai-scene-agent story 5, AD-18's sandbox).
# Runs `menger.tools.SceneValidator` against a compiled/loaded scene. Every invocation of
# this image is expected to go through run-sandboxed.sh, which wraps it with --network none,
# --cap-drop ALL, --read-only, a single read-only scene-file bind mount and resource/wall-clock
# bounds -- this Dockerfile only builds the image; it does not itself enforce AD-18's
# containment (that's the run wrapper's job).
#
# Build via ./build.sh, never `docker build .` directly: this Dockerfile expects a
# `stage/` directory (menger-app's host-built `sbt Universal/stage` output, native
# libmengergeometry.so included) and an `optix-sdk/` directory (a trimmed copy of the host's
# OPTIX_ROOT -- headers + lib64/ only, not the ~300MB doc/SDK samples) already present next
# to it as build context; build.sh produces both.
#
# CUDA "runtime" (not "devel"): SceneValidator's own code path -- compile via SceneCompiler,
# load via SceneLoader, check via PolytopeInvariants (see SceneValidator.scala) -- is pure
# JVM/Scala and never calls into the native CUDA/OptiX JNI bridge or invokes nvcc. The image
# still carries CUDA runtime libraries and a trimmed OPTIX_ROOT for parity with AD-18's
# stated runtime-needs list (JDK, packaged jar, CUDA toolkit, OptiX SDK) and to keep the door
# open for a later story (stage 5, semantic readback) that will actually render.
# Pinned by digest, not by tag (review round 2): this is the image that contains adversarial
# generated input, so "which bytes did we actually run" has to be answerable. `13.0.0-runtime-
# ubuntu24.04` is a mutable tag. Re-resolve with:
#   docker buildx imagetools inspect nvidia/cuda:13.0.0-runtime-ubuntu24.04 \
#     --format '{{.Manifest.Digest}}'
#
# Two targets share every layer up to the staged app (review 2026-10-14, menger#77):
#   scene-validator (the default, last stage) -- validation, no GPU;
#   render-window -- the interactive window the agent shows accepted scenes in, run by
#                    run-render-window.sh with the GPU and the user's X display. It runs the
#                    same generated code, so it gets the same containment.
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3 AS base

# Pinned too, for the same reason -- an unpinned apt install makes the image irreproducible
# even with the base digest fixed, since the archive moves underneath it. Re-resolve with:
#   docker run --rm ubuntu:24.04 sh -c \
#     'apt-get update -qq && apt-cache policy openjdk-21-jre-headless'
ARG JRE_VERSION=21.0.12+8-1~24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       "openjdk-21-jre-headless=${JRE_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

# menger's own runtime needs (Boundaries & Constraints: "the container image must also carry
# menger's actual runtime needs: JDK, sbt or a packaged jar, CUDA toolkit, OptiX SDK at
# OPTIX_ROOT"). Trimmed to headers + lib64/ -- see build.sh and the note above.
ENV OPTIX_ROOT=/usr/local/optix
COPY optix-sdk/ /usr/local/optix/

# The packaged, host-built distribution: menger-app's own classes, every dependency jar
# (including the scala3-compiler jar SceneCompiler itself drives), and the native library --
# laid out exactly as `sbt Universal/stage` produces it.
COPY stage/ /opt/menger-app/

# --- render-window -----------------------------------------------------------------------------
FROM base AS render-window

# What the window loads beyond the JVM: GLFW's X11 libraries, the GL dispatch libraries the
# NVIDIA driver's GLX plugs into (the driver itself comes in through CDI), and the FFmpeg
# libraries libmengergeometry.so links for video textures.
# ponytail: unpinned, unlike the JRE above; pin them if this image is ever published.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglx0 libegl1 libx11-6 libxext6 libxrandr2 libxcursor1 libxi6 libxinerama1 \
       libxxf86vm1 libavcodec60 libavformat60 libavutil58 libswscale7 libswresample4 \
    && rm -rf /var/lib/apt/lists/*

# The staged launchers are owner-executable only; the window runs as the host user's uid.
RUN chmod -R a+rX /opt/menger-app

# OptiX and CUDA (compute), GLX on the user's display (graphics, display).
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
# run-render-window.sh runs this as the host user (`--user`), so files the window writes next
# to the scene stay the user's; nothing in the image is that user's home.
ENV HOME=/tmp
WORKDIR /tmp
ENTRYPOINT ["/opt/menger-app/bin/menger-app"]

# --- scene-validator (default target) ----------------------------------------------------------
FROM base AS scene-validator

# AD-18 names exactly one capability -- `display` -- and this image's own code path
# (SceneValidator: compile -> load -> geometric-invariant check) is pure JVM/Scala and never
# reaches the CUDA/OptiX JNI bridge. `compute,utility` was widened for an `nvidia-smi` health
# check that the run wrapper no longer performs; run-sandboxed.sh drops `--gpus` entirely, so
# these are inert unless a caller deliberately re-adds GPU passthrough (review round 2). The
# story that actually renders in-container re-widens this, with a Spec Change Log entry.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=display

# Runs as a non-root user: nothing in SceneValidator's own code path needs root, and AD-18's
# containment is defense-in-depth, not a substitute for least privilege.
RUN useradd --create-home --shell /usr/sbin/nologin scene-validator
USER scene-validator

WORKDIR /workspace

# `/opt/menger-app/lib/*` is a shell glob the JVM launcher itself expands (java -cp supports
# a trailing `*` wildcard) into every jar `sbt Universal/stage` placed there -- the same
# mechanism `RestrictedClasspath.fullClasspath`'s `java.class.path` inspection relies on to
# see each jar as its own classpath entry, individually name-matched against the DSL-surface
# allowlist (see RestrictedClasspath.scala).
# -XX:MaxRAMPercentage (review round 2): without it the JVM sizes its heap from the *host's*
# memory view at ~25%, which under run-sandboxed.sh's `--memory 2g` lands near 512MB while
# SceneCompiler drives the full Scala 3 compiler inside it. The resulting OutOfMemoryError is
# now a tagged `lint-findings` rather than a crash (see SceneValidator.validate), but sizing the
# heap to the cgroup limit is what stops it happening at all.
ENTRYPOINT ["java", "-XX:MaxRAMPercentage=75.0", "-cp", "/opt/menger-app/lib/*", \
            "menger.tools.SceneValidator"]
