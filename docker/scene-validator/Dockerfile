# syntax=docker/dockerfile:1
#
# Renderer-side gauntlet validation image (spec-ai-scene-agent story 5, AD-18's sandbox).
# Runs `menger.tools.SceneValidator` against a compiled/loaded scene. Every invocation of
# this image is expected to go through run-sandboxed.sh, which wraps it with --gpus all,
# --network none, scoped bind mounts, and a seccomp profile -- this Dockerfile only builds
# the image; it does not itself enforce AD-18's containment (that's the run wrapper's job).
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
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# GPU passthrough (AD-18): `display` is the capability AD-18's own text names; `compute` and
# `utility` are added because `nvidia-smi` (the Verification section's own health check) and
# any CUDA runtime probing need the compute/utility capability sets, not just display output.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,display

# menger's own runtime needs (Boundaries & Constraints: "the container image must also carry
# menger's actual runtime needs: JDK, sbt or a packaged jar, CUDA toolkit, OptiX SDK at
# OPTIX_ROOT"). Trimmed to headers + lib64/ -- see build.sh and the note above.
ENV OPTIX_ROOT=/usr/local/optix
COPY optix-sdk/ /usr/local/optix/

# The packaged, host-built distribution: menger-app's own classes, every dependency jar
# (including the scala3-compiler jar SceneCompiler itself drives), and the native library --
# laid out exactly as `sbt Universal/stage` produces it.
COPY stage/ /opt/menger-app/

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
ENTRYPOINT ["java", "-cp", "/opt/menger-app/lib/*", "menger.tools.SceneValidator"]
