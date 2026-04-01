# User Guide Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `docs/USER_GUIDE.md` (2,551 lines) into role-based files under `docs/guide/`, leaving `docs/USER_GUIDE.md` as a navigation index.

**Architecture:** Pure content migration — no Scala/C++ changes, no tests to run. Each task extracts a contiguous block of the original file into a new file, then updates the index and cross-links. The original file is not deleted until the final task. Section numbering is reset to start at `##` in each new file (they are no longer chapters of one document). The existing `docs/TROUBLESHOOTING.md` is extended in-place with the richer §9 content from the user guide.

**Tech Stack:** Markdown, git. No build tools needed.

---

## Line Map (reference for all tasks)

```
docs/USER_GUIDE.md line ranges:
  Header + ToC    :   1 –  60
  §1 Introduction :  61 – 121
  §2 Getting Started: 122 – 247
  §3 Basic Usage  : 248 – 391
  §4 Rendering Modes: 392 – 459
  §5 Geometry     : 460 – 562
  §6 Materials & Lighting: 563 – 855
  §7.1 Animations : 858 – 933
  §7.2 t-Parameter: 934 – 1035
  §7.3 Caustics   : 1036 – 1099
  §7.4 Antialiasing: 1100 – 1150
  §7.5 Multi-objects: 1151 – 1241
  §7.6 DSL Reference: 1242 – 1853
  §8 Tutorials    : 1854 – 2085
  §9 Troubleshooting: 2086 – 2301
  §10 Reference   : 2302 – 2551  (includes footer)
```

---

## Task 1: Create `docs/guide/` directory and `quickstart.md`

**Goal:** Extract §1 + §2 into the new quickstart file.

**Files:**
- Create: `docs/guide/quickstart.md`
- (Do NOT modify `docs/USER_GUIDE.md` yet)

**Step 1: Create the directory**

```bash
mkdir -p docs/guide
```

**Step 2: Create `docs/guide/quickstart.md`**

The file content is lines 61–247 of `docs/USER_GUIDE.md` (§1 Introduction through §2 Getting
Started), with these changes:

- Add this header at the very top (before §1):

```markdown
# Menger — Quick Start

**Version**: 0.5.5
**Last Updated**: March 2026

← [User Guide Index](../USER_GUIDE.md)

---

```

- Keep all §1 and §2 content verbatim.
- Re-number headings: `## 1. Introduction` → `## Introduction`, `### 1.1 …` → `### …` etc.
  (drop the numeric prefixes throughout this file).
- Add a navigation footer at the very end:

```markdown
---

← [User Guide Index](../USER_GUIDE.md) | → [Usage & Rendering](user-guide.md)
```

**Step 3: Verify**

Open `docs/guide/quickstart.md`. It should be ~190 lines. All code blocks and headings render
correctly in a Markdown viewer.

**Step 4: Commit**

```bash
git add docs/guide/quickstart.md
git commit -m "docs: add guide/quickstart.md (§1 intro + §2 installation)"
```

---

## Task 2: Create `docs/guide/user-guide.md`

**Goal:** Extract §3 Basic Usage + §4 Rendering Modes + §5 Geometry + §6 Materials & Lighting.

**Files:**
- Create: `docs/guide/user-guide.md`

**Step 1: Create `docs/guide/user-guide.md`**

Content is lines 248–855 of `docs/USER_GUIDE.md` (§3 through §6), with these changes:

- Add this header at the very top:

```markdown
# Menger — Usage & Rendering

**Version**: 0.5.5
**Last Updated**: March 2026

← [Quick Start](quickstart.md) | [User Guide Index](../USER_GUIDE.md)

---

```

- Keep all §3–§6 content verbatim.
- Re-number headings (drop numeric chapter prefixes):
  - `## 3. Basic Usage` → `## Basic Usage`
  - `## 4. Rendering Modes` → `## Rendering Modes`
  - `## 5. Geometry Types` → `## Geometry Types`
  - `## 6. Materials and Lighting` → `## Materials and Lighting`
  - All `### N.N` subsection numbers likewise stripped.
- Add a navigation footer at the very end:

```markdown
---

← [Quick Start](quickstart.md) | [User Guide Index](../USER_GUIDE.md) | → [Advanced Features](advanced.md)
```

**Step 2: Verify**

File should be ~610 lines. Check that all internal anchor links within §6 (e.g. material preset
tables, the lighting setup sub-sections) still resolve — they are self-referential within the file.

**Step 3: Commit**

```bash
git add docs/guide/user-guide.md
git commit -m "docs: add guide/user-guide.md (§3 usage, §4 rendering, §5 geometry, §6 materials)"
```

---

## Task 3: Create `docs/guide/advanced.md`

**Goal:** Extract §7.1–§7.5 (Animations, t-parameter, Caustics, Antialiasing, Multi-objects).

**Files:**
- Create: `docs/guide/advanced.md`

**Step 1: Create `docs/guide/advanced.md`**

Content is lines 856–1241 of `docs/USER_GUIDE.md` (the `## 7. Advanced Features` heading plus
subsections 7.1 through 7.5 only — stop before `### 7.6 Scala DSL`), with these changes:

- Add this header at the very top:

```markdown
# Menger — Advanced Features

**Version**: 0.5.5
**Last Updated**: March 2026

← [Usage & Rendering](user-guide.md) | [User Guide Index](../USER_GUIDE.md)

---

```

- Re-number: `## 7. Advanced Features` → `## Advanced Features`; strip `### 7.x` prefixes from all subsection headings.
- In §7.3 Caustics, the existing "For more details, see [docs/caustics/CAUSTICS.md](caustics/CAUSTICS.md)" link uses a relative path. Update it to `../caustics/CAUSTICS.md` since the file now lives in `docs/guide/`.
- In §7.6 forward-reference: after the last subsection (7.5) add a callout box:

```markdown
> **DSL users:** The Scala DSL for Scene Description has its own reference document.
> See [DSL Reference](dsl-reference.md).
```

- Add a navigation footer at the very end:

```markdown
---

← [Usage & Rendering](user-guide.md) | [User Guide Index](../USER_GUIDE.md) | → [DSL Reference](dsl-reference.md)
```

**Step 2: Verify**

File should be ~390 lines. Confirm the caustics link path is correct relative to `docs/guide/`.

**Step 3: Commit**

```bash
git add docs/guide/advanced.md
git commit -m "docs: add guide/advanced.md (§7.1–7.5: animations, caustics, AA, multi-objects)"
```

---

## Task 4: Create `docs/guide/dsl-reference.md`

**Goal:** Extract §7.6 Scala DSL for Scene Description (the largest single block at ~610 lines).

**Files:**
- Create: `docs/guide/dsl-reference.md`

**Step 1: Create `docs/guide/dsl-reference.md`**

Content is lines 1242–1853 of `docs/USER_GUIDE.md` (§7.6 only), with these changes:

- Add this header at the very top:

```markdown
# Menger — Scala DSL Reference

**Version**: 0.5.5
**Last Updated**: March 2026

← [Advanced Features](advanced.md) | [User Guide Index](../USER_GUIDE.md)

---

```

- Re-number: `### 7.6 Scala DSL for Scene Description` → `## Scala DSL for Scene Description` (promote to `##`).
- All sub-headings within (currently `####`) promote one level each (`####` → `###`).
- Add a navigation footer at the very end:

```markdown
---

← [Advanced Features](advanced.md) | [User Guide Index](../USER_GUIDE.md) | → [Tutorials](tutorials.md)
```

**Step 2: Verify**

File should be ~615 lines. Confirm the parametric surfaces section is present and the built-in
scene name list (`parametric-sphere`, `parametric-torus`, etc.) is intact.

**Step 3: Commit**

```bash
git add docs/guide/dsl-reference.md
git commit -m "docs: add guide/dsl-reference.md (§7.6 full DSL + parametric surfaces reference)"
```

---

## Task 5: Create `docs/guide/tutorials.md`

**Goal:** Extract §8 Examples and Tutorials.

**Files:**
- Create: `docs/guide/tutorials.md`

**Step 1: Create `docs/guide/tutorials.md`**

Content is lines 1854–2085 of `docs/USER_GUIDE.md` (§8), with these changes:

- Add this header at the very top:

```markdown
# Menger — Tutorials

**Version**: 0.5.5
**Last Updated**: March 2026

← [DSL Reference](dsl-reference.md) | [User Guide Index](../USER_GUIDE.md)

---

```

- Re-number: `## 8. Examples and Tutorials` → `## Examples and Tutorials`; `### 8.1 Tutorial 1:` → `### Tutorial 1:` etc.
- Any relative links to `caustics/CAUSTICS.md` within tutorials: update to `../caustics/CAUSTICS.md`.
- Add a navigation footer at the very end:

```markdown
---

← [DSL Reference](dsl-reference.md) | [User Guide Index](../USER_GUIDE.md) | → [Reference](reference.md)
```

**Step 2: Verify**

File should be ~235 lines and contain all 5 tutorials.

**Step 3: Commit**

```bash
git add docs/guide/tutorials.md
git commit -m "docs: add guide/tutorials.md (§8: all 5 tutorials)"
```

---

## Task 6: Merge §9 Troubleshooting into `docs/TROUBLESHOOTING.md`

**Goal:** Extend the existing developer-focused `docs/TROUBLESHOOTING.md` with the user-facing
troubleshooting content from USER_GUIDE §9.

**Files:**
- Modify: `docs/TROUBLESHOOTING.md`

**Step 1: Read both files before editing**

Read `docs/TROUBLESHOOTING.md` (169 lines) and lines 2086–2301 of `docs/USER_GUIDE.md` to
understand what overlaps.

**Step 2: Extend `docs/TROUBLESHOOTING.md`**

Append the following structure at the end of `docs/TROUBLESHOOTING.md`:

```markdown
---

## User-Facing Common Issues

> This section covers runtime issues when using the rendered application.
> For build and JNI issues, see the sections above.

```

Then append lines 2086–2301 of `docs/USER_GUIDE.md` (the full §9 content), with these changes:

- Drop the `## 9. Troubleshooting` top-level heading (already provided by the section header above).
- Keep `### 9.1 Common Issues`, `### 9.2 Performance Tips`, `### 9.3 Getting Help` as-is
  (these become `###` sub-headings within the new section, which is correct).
- Strip the `9.x` numeric prefixes from each heading.

Also add a navigation header at the very top of `docs/TROUBLESHOOTING.md` (before the first line):

```markdown
← [User Guide Index](USER_GUIDE.md)

```

**Step 3: Verify**

`docs/TROUBLESHOOTING.md` should now be ~380 lines. The arc42 link
`[TROUBLESHOOTING.md](../TROUBLESHOOTING.md)` in `docs/arc42/11-risks-and-technical-debt.md`
still resolves correctly (path unchanged).

**Step 4: Commit**

```bash
git add docs/TROUBLESHOOTING.md
git commit -m "docs: merge USER_GUIDE §9 troubleshooting content into TROUBLESHOOTING.md"
```

---

## Task 7: Create `docs/guide/reference.md`

**Goal:** Extract §10 Complete Option List, Keyboard Shortcuts, and File Formats.

**Files:**
- Create: `docs/guide/reference.md`

**Step 1: Create `docs/guide/reference.md`**

Content is lines 2302–2551 of `docs/USER_GUIDE.md` (§10 through end of file, including the
footer "Thank you for using Menger"), with these changes:

- Add this header at the very top:

```markdown
# Menger — Reference

**Version**: 0.5.5
**Last Updated**: March 2026

← [Tutorials](tutorials.md) | [User Guide Index](../USER_GUIDE.md)

---

```

- Re-number: `## 10. Reference` → `## Reference`; strip `### 10.x` prefixes.
- The "Thank you for using Menger" footer at the end: keep it as-is, it belongs here.
- Add a navigation line just before the footer:

```markdown
---

← [Tutorials](tutorials.md) | [User Guide Index](../USER_GUIDE.md)

```

**Step 2: Verify**

File should be ~255 lines. Check that the complete option list table is intact and the keyboard
shortcuts table renders correctly.

**Step 3: Commit**

```bash
git add docs/guide/reference.md
git commit -m "docs: add guide/reference.md (§10: complete option list, keyboard shortcuts, file formats)"
```

---

## Task 8: Rewrite `docs/USER_GUIDE.md` as index page

**Goal:** Replace the monolithic USER_GUIDE.md with a concise navigation index.

**Files:**
- Modify: `docs/USER_GUIDE.md` (replace entirely)

**Step 1: Write the new `docs/USER_GUIDE.md`**

Replace the entire file with the following content (adapt version/date to match the other files):

```markdown
# Menger User Guide

**Version**: 0.5.5
**Last Updated**: March 2026

Menger is a 3D/4D fractal visualization and GPU ray tracing tool. This index links to each part
of the documentation.

---

## Documentation Map

| Document | Contents | Audience |
|----------|----------|----------|
| [Quick Start](guide/quickstart.md) | What Menger is, system requirements, installation, first render | New users |
| [Usage & Rendering](guide/user-guide.md) | CLI options, interactive controls, rendering modes, geometry types, materials, lighting | Regular users |
| [Advanced Features](guide/advanced.md) | Animations, t-parameter system, caustics (PPM), antialiasing, multi-object scenes | Power users |
| [Scala DSL Reference](guide/dsl-reference.md) | Type-safe scene description language, all DSL types, parametric surfaces | DSL / developers |
| [Tutorials](guide/tutorials.md) | Step-by-step walkthroughs: first render, glass, animation, 4D, complex scenes | All users |
| [Troubleshooting](TROUBLESHOOTING.md) | Common errors, performance tips, getting help | All users |
| [Reference](guide/reference.md) | Complete CLI option list, keyboard shortcuts, file formats | All users |

---

## Quick Links

- **Install and first render** → [Quick Start](guide/quickstart.md#quick-start)
- **CLI option list** → [Reference](guide/reference.md#complete-option-list)
- **Glass/material setup** → [Usage & Rendering](guide/user-guide.md#materials-and-lighting)
- **Define a scene in Scala** → [DSL Reference](guide/dsl-reference.md)
- **Caustics rendering** → [Advanced Features](guide/advanced.md#caustics-light-focusing-effects)
- **OptiX ray tracing mode** → [Usage & Rendering](guide/user-guide.md#rendering-modes)

---

For questions, issues, or contributions:
- GitLab: https://gitlab.com/lilacashes/menger
- Issues: https://gitlab.com/lilacashes/menger/issues
```

**Step 2: Verify**

The file should be ~45 lines. Open README.md and confirm the link
`[Complete User Guide](docs/USER_GUIDE.md)` still points to a real, useful file.

**Step 3: Commit**

```bash
git add docs/USER_GUIDE.md
git commit -m "docs: replace USER_GUIDE.md with navigation index (content moved to docs/guide/)"
```

---

## Task 9: Update cross-references

**Goal:** Fix all remaining references that pointed to specific sections of the old USER_GUIDE.md.

**Files to check and update:**

1. **`AGENTS.md`** line ~386:
   - Find: `docs/USER_GUIDE.md` (Version field in header)
   - The version field now lives in `docs/guide/user-guide.md` (and all other guide files).
   - Update the line to: `docs/guide/user-guide.md` (or list all guide files — they all carry the version header).
   - Also update the description: "Version field in header of all `docs/guide/*.md` files"

2. **`.claude/commands/sprint-close.md`** line ~145:
   - Find: `grep -m1 "version" docs/USER_GUIDE.md`
   - Update to: `grep -m1 "version" docs/guide/user-guide.md`

3. **`CODE_IMPROVEMENTS.md`**:
   - Line ~16: `docs/USER_GUIDE.md section 7.6 "Included Example Scenes"` → `docs/guide/dsl-reference.md`
   - Line ~201: `docs/USER_GUIDE.md section 6.2 (lines ~628–631)` → `docs/guide/user-guide.md`
   - Line ~231: `USER_GUIDE §7.3` → `docs/guide/advanced.md §Caustics`

4. **`docs/sprints/SPRINT15.md`**, **`SPRINT14.md`**, **`SPRINT17.md`**, **`SPRINT18.md`**, **`SPRINT19.md`**:
   - Replace occurrences of `USER_GUIDE.md` with the specific file that now holds each section:
     - §6 → `guide/user-guide.md`
     - §7.1/7.2 → `guide/advanced.md`
     - §7.3 caustics → `guide/advanced.md`
     - §7.6 DSL → `guide/dsl-reference.md`
     - §8 tutorials → `guide/tutorials.md`

5. **`docs/arc42/11-risks-and-technical-debt.md`** line ~52:
   - The `TROUBLESHOOTING.md` link is already correct — no change needed.

6. **`CHANGELOG.md`**:
   - Occurrences are narrative/historical and don't need updating (they describe past work).

**Step 1: Make the updates**

Work through each file above, making the targeted replacements. Use Read + Edit (not sed).

**Step 2: Verify no dead USER_GUIDE references remain in active docs**

```bash
grep -rn "USER_GUIDE" docs/ AGENTS.md CODE_IMPROVEMENTS.md .claude/ \
  --include="*.md" \
  | grep -v "archive/" \
  | grep -v "CHANGELOG"
```

Expected: only `README.md` (index link — correct) and any historical sprint docs that are
acceptable to leave as-is.

**Step 3: Commit**

```bash
git add AGENTS.md .claude/commands/sprint-close.md CODE_IMPROVEMENTS.md \
        docs/sprints/SPRINT14.md docs/sprints/SPRINT15.md \
        docs/sprints/SPRINT17.md docs/sprints/SPRINT18.md docs/sprints/SPRINT19.md
git commit -m "docs: update cross-references to point to new docs/guide/ structure"
```

---

## Task 10: Final verification

**Goal:** Confirm the split is complete and nothing is broken.

**Step 1: Check all new files exist**

```bash
ls -la docs/guide/
```

Expected output — six files:
```
quickstart.md
user-guide.md
advanced.md
dsl-reference.md
tutorials.md
reference.md
```

**Step 2: Check line counts are reasonable**

```bash
wc -l docs/guide/*.md docs/USER_GUIDE.md docs/TROUBLESHOOTING.md
```

Expected approximate counts:
```
  190  docs/guide/quickstart.md
  610  docs/guide/user-guide.md
  390  docs/guide/advanced.md
  615  docs/guide/dsl-reference.md
  235  docs/guide/tutorials.md
  255  docs/guide/reference.md
   45  docs/USER_GUIDE.md
  380  docs/TROUBLESHOOTING.md
```

**Step 3: Verify README link is still valid**

```bash
grep "USER_GUIDE" README.md
```

Should show the index link pointing to `docs/USER_GUIDE.md`.

**Step 4: Verify no broken relative links in guide files**

Check all relative links in guide files point to existing paths:

```bash
grep -h "\](\.\./" docs/guide/*.md | sort -u
```

Every `../` path should resolve correctly from `docs/guide/`.

**Step 5: Verify arc42 TROUBLESHOOTING link**

```bash
grep "TROUBLESHOOTING" docs/arc42/11-risks-and-technical-debt.md
```

Should still show `[TROUBLESHOOTING.md](../TROUBLESHOOTING.md)` — valid from `docs/arc42/`.

**Step 6: Final commit if any cleanup needed**

```bash
git add -p   # review any remaining changes
git commit -m "docs: final cleanup after user guide split"
```

---

## Summary

After all 10 tasks:

| File | Status | Lines (approx) |
|------|--------|---------------|
| `docs/USER_GUIDE.md` | Replaced with index | ~45 |
| `docs/guide/quickstart.md` | New | ~190 |
| `docs/guide/user-guide.md` | New | ~610 |
| `docs/guide/advanced.md` | New | ~390 |
| `docs/guide/dsl-reference.md` | New | ~615 |
| `docs/guide/tutorials.md` | New | ~235 |
| `docs/guide/reference.md` | New | ~255 |
| `docs/TROUBLESHOOTING.md` | Extended | ~380 |

Total content: ~2,720 lines across 8 files (same total, now navigable).
