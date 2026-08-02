# Architecture documentation moved

The arc42 architecture documentation is no longer in this repo. It moved to the **workspace
repo** (`menger-toplevel`), at `docs/arc42/`, because it documents the Menger *system* — all
three repos as one whole. Section 5 gives `optix-jni` and `menger-common` their own
building-block entries and records the native-struct ABI contract between them; section 9
holds the decisions that split them into separate published artifacts.

From a full workspace checkout it is at `../../../docs/arc42/`. `ARCHITECTURE_MODULES.md`
(the three-module dependency graph) moved alongside it to `../../../docs/ARCHITECTURE_MODULES.md`.

A standalone clone of just this repo does not carry these — that is the tradeoff of
documenting one system that lives in three repos from a single place. Clone the workspace
(`github.com/lene/menger-toplevel`) for the architecture docs.
