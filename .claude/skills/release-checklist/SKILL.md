---
name: release-checklist
description: Step-by-step guide for releasing a new Menger version — version bumps, changelog, CI pipeline, post-release verification, retrospective, and sprint opening.
---

# Menger Release Checklist

Comprehensive step-by-step guide for releasing a new version of Menger.

**Estimated total time:** 30-60 minutes from commit to next sprint opened
- Pre-release preparation: 15-30 minutes (manual)
- Pre-push validation: 8-10 minutes (automated)
- Commit & push: 1-2 minutes (manual)
- Release pipeline: 12-15 minutes (automated)
- Post-release verification: 5-10 minutes (manual)
- Retrospective & sprint opening: 15-30 minutes (interactive)

---

## Phase 1: Pre-Release Preparation (Manual - 15-30 minutes)

### 0. Confirm Release Version With User

**Before touching any files, ask the user:**

> "What version number should this release be? (current version is X.Y.Z)"

**Wait for an explicit answer. Do not infer, guess, or derive the version from sprint numbers, roadmap labels, or any other source.** The user decides the version.

Only proceed to step 1 once you have the version number confirmed in this conversation.

---

### 1. Version Management

Decide on version number following semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes only

**Update version in 4 locations (must match):**

- [ ] `menger-app/build.sbt` (line 2):
  ```scala
  version := "X.Y.Z"
  ```

- [ ] `.gitlab-ci.yml` (line 15):
  ```yaml
  DEPLOYABLE_VERSION: X.Y.Z
  ```

- [ ] `menger-app/src/main/scala/menger/MengerCLIOptions.scala` (line ~30):
  ```scala
  version("menger vX.Y.Z ...")
  ```

- [ ] `docs/USER_GUIDE.md` (line 3) — checked by `TagIsNewAndConsistent` CI job:
  ```markdown
  **Version**: X.Y.Z
  **Last Updated**: Month YYYY
  ```

**Automated validation:** Pre-push hook verifies version consistency in the first 3 files.
`TagIsNewAndConsistent` CI job additionally checks `docs/USER_GUIDE.md` — pre-push hook
does NOT catch this mismatch, so update it manually during this step.

### 2. Documentation Updates

#### CHANGELOG.md (Required)

- [ ] Update `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD`
- [ ] **Remove any remaining `## [Unreleased]` block** — `ChangelogIsUpdated` CI job fails if stale content remains after the release entry
- [ ] Follow keepachangelog.com format:
  - **Added**: New features
  - **Changed**: Changes to existing functionality
  - **Deprecated**: Soon-to-be removed features
  - **Removed**: Removed features
  - **Fixed**: Bug fixes
  - **Security**: Security fixes

- [ ] Write clear, user-focused descriptions
- [ ] Document only bugs that existed in previous releases (not bugs introduced and fixed in same version)
- [ ] Add version comparison link at bottom:
  ```markdown
  [X.Y.Z]: https://gitlab.com/lilacashes/menger/-/compare/PREV...X.Y.Z
  ```

**Example entry:**
```markdown
## [0.4.3] - 2026-02-05

### Fixed
- Corrected directional light direction for accurate shadow rays
- Fixed PushToGithub CI job by adding GIT_DEPTH: 0 for complete history

### Added
- Reference images for 3D fractional sponge tests
```

#### ROADMAP.md

- [ ] Mark completed sprint as "✅ Complete"
- [ ] Update milestone status if version completes a milestone
- [ ] Remove obsolete items from TODO.md (move to ROADMAP backlog or delete)
- [ ] Archive completed sprint plan — **always required**:
  ```bash
  git mv docs/sprints/SPRINT_N.md docs/archive/sprints/SPRINT_N.md
  ```
  Where `SPRINT_N.md` is the completed sprint's file (e.g. `SPRINT18.md`).
  Update the index tables in `docs/sprints/README.md` and `docs/archive/README.md`.
  Verify no copy of the file remains in `docs/sprints/` afterwards.

#### arc42 Documentation (If Applicable)

Update if architectural changes were made:

- [ ] `docs/arc42/09-architectural-decisions.md` - Document new architectural decisions
- [ ] `docs/arc42/10-quality-requirements.md` - Update if performance characteristics changed
- [ ] `docs/arc42/11-risks-and-technical-debt.md` - Document new technical debt or resolved risks

#### USER_GUIDE.md (If New Features)

- [ ] Update usage examples if new features added
- [ ] Add new command-line options to reference
- [ ] Update screenshots/examples if UI changed

### 3. Code Quality Assessment

Run comprehensive codebase assessment and document in CODE_IMPROVEMENTS.md:

- [ ] Check clean code guidelines:
  - Descriptive naming (functions, variables, classes)
  - Function length (max ~50 lines, prefer smaller)
  - Cyclomatic complexity (avoid deeply nested logic)

- [ ] Verify architectural quality:
  - Separation of concerns (rendering vs business logic)
  - Code duplication (DRY principle)
  - Hardcoded constants (extract to named constants)
  - Over-long functions and classes

- [ ] Review test coverage quality:
  - Are critical paths tested?
  - Are edge cases covered?
  - Are tests readable and maintainable?

- [ ] Document findings in CODE_IMPROVEMENTS.md:
  - Overall grade (A+ to F)
  - Strengths to maintain
  - Areas for improvement
  - Action items for future sprints

**IMPORTANT:** Always address critical issues before release. Do not defer critical bugs or security issues.

### 4. Dependency Updates (Required)

- [ ] Check Scala and sbt versions — compare against latest releases:
  - Scala: https://www.scala-lang.org/download/
  - sbt: https://www.scala-sbt.org/download/
  - Current versions in `build.sbt` and `project/build.properties`

- [ ] Check CUDA Toolkit version — compare against latest driver-compatible release:
  ```bash
  nvcc --version
  ```
  Check: https://developer.nvidia.com/cuda-downloads

- [ ] Check OptiX SDK version — compare against latest:
  - Current version in `optix-jni/README.md` or `CUDA_HOME`/`OPTIX_ROOT` paths
  - Check: https://developer.nvidia.com/designworks/optix/download

- [ ] Check for outdated Scala/Java dependencies:
  ```bash
  sbt dependencyUpdates
  ```

- [ ] Review for security vulnerabilities:
  ```bash
  sbt dependencyCheck
  ```

- [ ] Update dependencies if needed:
  - Security patches: **Update immediately**
  - Major versions: Test thoroughly, document breaking changes
  - Minor versions: Update if improvements justify risk

- [ ] After updates, run full pre-push hook:
  ```bash
  .git_hooks/pre-push
  ```

---

## Phase 2: Pre-Push Validation (Automated - 8-10 minutes)

The pre-push hook (`.git_hooks/pre-push`) runs automatically on `git push`. You can also run it manually:

```bash
.git_hooks/pre-push
```

**What it validates:**

### Environment Validation
- [x] `CUDA_HOME` set and valid (points to CUDA Toolkit)
- [x] `OPTIX_ROOT` set and valid (points to OptiX SDK)
- [x] GitLab CI config valid (requires `GITLAB_ACCESS_TOKEN` env var)

### Version Consistency
- [x] Scala version matches between `build.sbt` and `.gitlab-ci.yml`
- [x] Version number consistent across all 3 files
- [x] Git tag for version doesn't already exist

### Build & Tests (~8 minutes)
- [x] Full compilation: `sbt compile`
- [x] All tests pass: `xvfb-run sbt test` (~1,070 tests)
  - 27 C++ tests (Google Test)
  - 1,043 Scala tests (AnyFlatSpec)
- [x] Code quality: `sbt "scalafix --check"`
- [x] Package builds: `sbt "mengerApp / Universal / packageBin"`

### Test Coverage (Ratchet Mechanism)
- [x] Coverage ≥ 80% (absolute minimum)
- [x] Coverage drop ≤ 1% from baseline
- [x] If dropped, must stay above 80%

**Coverage baseline:** `.coverage_baseline`

### Memory Safety (~2-3 minutes)
- [x] GPU memory leaks: `compute-sanitizer --tool memcheck`
- [x] Host memory leaks: `valgrind --leak-check=full`

### Integration Tests (27 scenarios)
- [x] Basic rendering (spheres, cubes, sponges, tesseracts)
- [x] Materials (glass, chrome, gold, matte, etc.)
- [x] Multi-object scenes (IAS)
- [x] Textures
- [x] Shadows and lighting
- [x] 4D objects and rotations
- [x] Fractional sponge levels
- [x] Headless rendering
- [x] Error handling

**Test script:** `./scripts/integration-tests.sh`

**If pre-push hook fails:**
- Read error messages carefully
- Follow **TEST FAILURE PROTOCOL** from AGENTS.md (below)
- Fix issues and re-run
- **Never bypass tests for releases**

#### TEST FAILURE PROTOCOL

When tests fail during release preparation:

1. **Investigate root cause** - Don't just make tests pass
   - Run failing test in isolation
   - Check if test expectations are wrong OR implementation is wrong
   - Trace through the code to understand the failure

2. **Determine if test or code needs fixing:**
   - If implementation is wrong: Fix the code, verify test passes
   - If test expectations are wrong: Update test, document WHY in commit message

3. **Document test changes in commit message:**
   ```
   test: Update shadow direction test expectations

   The test was checking for inverted light direction. Updated to match
   the corrected implementation where directional lights point FROM the
   light source (standard convention).

   Previous test expected: direction = normalize(light_pos - hit_point)
   Corrected test expects: direction = -light.direction
   ```

4. **Never blindly rewrite tests to pass** - This hides bugs
   - Understand the failure completely before changing anything
   - If uncertain, ask for human review

---

## Phase 3: Commit & Push via Merge Request (Manual - 5-10 minutes)

**IMPORTANT:** The main branch is protected. All changes must go through merge requests.

### Create Feature Branch

- [ ] Create release branch from main:
  ```bash
  git checkout main
  git pull
  git checkout -b release/vX.Y.Z
  ```

### Review Changes

- [ ] Review all changes:
  ```bash
  git diff main
  ```

- [ ] Verify only intended changes present

### Stage Files

**Never use `git add -A` - add files explicitly:**

```bash
git add menger-app/build.sbt
git add .gitlab-ci.yml
git add menger-app/src/main/scala/menger/MengerCLIOptions.scala
git add CHANGELOG.md
git add ROADMAP.md
git add CODE_IMPROVEMENTS.md
git add docs/arc42/  # If updated
# Add other modified files explicitly
```

### Create Release Commit

- [ ] Write clear commit message:
  ```bash
  git commit -m "release: Version X.Y.Z

  Brief summary of major changes (1-2 sentences):
  - Fixed critical shadow direction bug affecting all scenes
  - Improved CI pipeline reliability with complete git history

  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
  ```

### Push and Create Merge Request

- [ ] Push release branch:
  ```bash
  git push origin release/vX.Y.Z
  ```

**Pre-push hook runs automatically:**
- If it fails, fix issues and push again
- Never use `--no-verify` for releases

- [ ] Create merge request:
  ```bash
  glab mr create --title "Release vX.Y.Z" --description "Release version X.Y.Z

  ## Changes
  - Updated version to X.Y.Z across all files
  - Updated CHANGELOG.md with release notes
  - Updated ROADMAP.md marking sprint complete
  - Code quality assessment in CODE_IMPROVEMENTS.md

  ## Pre-Push Validation
  - ✅ All tests pass (1,070 tests)
  - ✅ Coverage: XX.X% (baseline: XX.X%)
  - ✅ No memory leaks detected
  - ✅ All integration tests pass (27 scenarios)

  Merge to trigger release pipeline." --target-branch main
  ```

### Wait for Pipeline and Merge

- [ ] Wait for pipeline to complete (all jobs must pass)
- [ ] Review the MR (code review if working with team)
- [ ] Merge the MR when pipeline is green

**After merge to main, the release pipeline triggers automatically**

---

## Phase 4: Release Pipeline (Fully Automated - 12-15 minutes)

GitLab CI automatically runs the full release pipeline when the MR merges to main.

### Monitor Pipeline Progress

- [ ] Navigate to: https://gitlab.com/lilacashes/menger/-/pipelines
- [ ] Find the latest pipeline (should be running)
- [ ] Watch jobs progress through stages

### Pipeline Stages

**Build Stage:**
- [x] Test:Full - Full test suite with GPU
- [x] Test:OptiXJni - OptiX JNI C++ tests
- [x] Test:ComputeSanitizer - GPU memory leak detection
- [x] Test:Valgrind - Host memory leak detection
- [x] Test:OptiXIntegration - 27 integration scenarios
- [x] CheckCoverage - Coverage ratchet validation
- [x] Scalafix - Code quality enforcement

**Release Stage:**
- [x] BuildDeployable - Package distribution `.zip`
- [x] **CreateRelease** - Creates GitLab release and git tag

**Tag Pipeline (Triggered by CreateRelease):**
- [x] Upload - Upload artifact to GitLab Package Registry
- [x] PushToGithub - Mirror to GitHub repository (with GIT_DEPTH: 0)
- [x] CreateGithubRelease - Create GitHub release with CHANGELOG link

### Verify All Jobs Pass

- [ ] All jobs show green checkmarks ✅
- [ ] No failed (red ❌) or canceled (gray ⊘) jobs

**If any job fails:**
1. Click on failed job to view logs
2. Identify root cause
3. Fix issue locally
4. Create fix commit and push to release branch
5. Pipeline re-runs automatically
6. Merge MR when green

**Common failures:**
- Test failures: Follow TEST FAILURE PROTOCOL (Phase 2)
- Coverage drop: Add tests or investigate removed code
- Memory leaks: Use compute-sanitizer/valgrind locally to debug
- Integration test failures: Compare rendered output to reference images

---

## Phase 5: Post-Release Verification (Manual - 5-10 minutes)

### Verify Releases Created

**GitLab Release:**
- [ ] Navigate to: https://gitlab.com/lilacashes/menger/-/releases
- [ ] Verify release `vX.Y.Z` exists
- [ ] Check release notes include CHANGELOG link
- [ ] Verify artifact is attached

**GitHub Release:**
- [ ] Navigate to: https://github.com/lene/menger/releases
- [ ] Verify release `vX.Y.Z` exists
- [ ] Check release notes match GitLab
- [ ] Verify tag points to correct commit

### Verify Git Tags

- [ ] Fetch tags:
  ```bash
  git fetch --all --tags
  ```

- [ ] Verify tag exists:
  ```bash
  git tag | grep X.Y.Z
  ```

- [ ] Verify tag points to correct commit:
  ```bash
  git show vX.Y.Z
  ```

### Test Packaged Distribution

Download and test the release artifact:

```bash
# Download from GitLab Package Registry or GitHub Releases
wget https://gitlab.com/lilacashes/menger/-/releases/vX.Y.Z/downloads/menger-X.Y.Z.zip

# Extract
unzip menger-X.Y.Z.zip
cd menger-app-X.Y.Z

# Verify version
./bin/menger-app --version
# Should output: menger vX.Y.Z ...

# Quick smoke test (headless render)
xvfb-run ./bin/menger-app --object sphere --timeout 0.1 --level 2 --headless --save-name /tmp/smoke-test.png

# Verify output
file /tmp/smoke-test.png
# Should be: PNG image data, 800 x 600 ...
```

### Documentation Verification

- [ ] CHANGELOG.md renders correctly on GitLab and GitHub
- [ ] GitHub release shows correct CHANGELOG link (not 404)
- [ ] README.md installation instructions still accurate
- [ ] Version number appears correctly in docs

### Close GitLab Issues and Milestone

- [ ] Close all issues linked to this sprint/version:
  ```bash
  # List open issues for this milestone
  glab issue list --milestone "vX.Y.Z"
  # Close each resolved issue
  glab issue close <issue-id>
  ```

- [ ] Close the GitLab milestone for vX.Y.Z:
  ```bash
  # List milestones to find the ID
  glab api projects/:fullpath/milestones --method GET | python3 -c "
  import sys, json
  for m in json.load(sys.stdin):
      print(f\"{m['id']:4d}  {m['title']:<20s}  {m['state']}\")"
  # Close the milestone
  glab api projects/:fullpath/milestones/<milestone-id> --method PUT -f state_event=close
  ```

  Alternatively: close via GitLab UI at
  https://gitlab.com/lilacashes/menger/-/milestones

### Cleanup & Next Steps

- [ ] Update `.coverage_baseline` if coverage improved significantly
- [ ] Archive sprint documentation — **always required, verify with `ls docs/sprints/`** that no completed sprint file remains there
- [ ] Create next sprint planning document (if ready)
- [ ] Clean up release branch:
  ```bash
  git branch -d release/vX.Y.Z
  git push origin --delete release/vX.Y.Z
  ```

- [ ] Clean up old worktrees — **interactive, required**:
  1. List all worktrees and their branches:
     ```bash
     git worktree list
     ```
  2. For each worktree that is not the main workspace, show the user the path, branch name,
     and whether the branch is merged into main:
     ```bash
     git branch --merged main | grep <branch-name>
     ```
     Present a summary like:
     > `.worktrees/feature-sprint-17` → branch `feature/sprint-17` (merged into main)
     > `.worktrees/feature-foo` → branch `feature/foo` (NOT merged)
  3. Ask the user: *"Which of these worktrees should I remove?"* Wait for explicit confirmation.
  4. For each confirmed worktree, check for uncommitted changes first:
     ```bash
     git -C <path> status --short
     ```
     If there are changes, tell the user and ask whether to force-remove or rescue files first.
  5. Remove confirmed worktrees:
     ```bash
     git worktree remove <path>        # or --force if user confirmed discarding changes
     git worktree prune                 # clean up stale .git/worktrees entries
     ```

- [ ] Announce release (if applicable):
  - GitLab/GitHub release notes
  - Internal team notification
  - Community announcement (if public)

---

## Common Issues & Solutions

### Version Mismatch Errors

**Problem:** Pre-push hook reports version inconsistency

**Solution:**
- Check all 3 files (build.sbt, .gitlab-ci.yml, MengerCLIOptions.scala)
- Ensure exact version match (no extra spaces, consistent format)
- Commit and try again

### Tag Already Exists

**Problem:** `error: tag 'vX.Y.Z' already exists`

**Solution:**
- If pushing to GitHub after GitLab release: This is normal, PushToGithub job handles it
- If genuine duplicate: Increment version number and update all 3 files

### Pipeline Failures

**Test failures:**
- Follow TEST FAILURE PROTOCOL (Phase 2)
- Investigate root cause before changing tests
- Never blindly update tests to pass

**Coverage drops:**
- Identify which code was removed/changed
- Add tests for uncovered code paths
- Or justify coverage drop in commit message

**Memory leaks:**
- Run locally: `compute-sanitizer --tool memcheck ./standalone_test`
- Fix leaks before releasing
- Update suppressions if false positive

### Protected Branch Error

**Problem:** `remote: GitLab: You are not allowed to push code to protected branch main`

**Solution:**
- This is expected and correct - main branch is protected
- Always create a feature/release branch
- Push to that branch and create an MR
- Merge MR after pipeline passes

### Pre-Push Hook Skipped

**Problem:** Push succeeded but pre-push hook didn't run

**Solution:**
- Verify hook is installed: `ls -la .git/hooks/pre-push`
- Should be symlink to: `../../.git_hooks/pre-push`
- Re-run manually: `.git_hooks/pre-push`
- If tests fail, create fix commit

---

## Emergency Hotfix Process

For critical bugs that need immediate release:

1. **Create hotfix branch from main:**
   ```bash
   git checkout main
   git pull
   git checkout -b hotfix/vX.Y.Z-critical-bug-description
   ```

2. **Make minimal fix** (smallest possible change)

3. **Test thoroughly - run full pre-push hook:**
   ```bash
   .git_hooks/pre-push
   ```

4. **Update version** (bump PATCH number):
   - Example: 0.4.3 → 0.4.4
   - Update all 3 version files

5. **Update CHANGELOG.md:**
   ```markdown
   ## [0.4.4] - 2026-02-05

   ### Fixed
   - Critical: [Description of bug and fix]
   ```

6. **Commit and push hotfix branch:**
   ```bash
   git add [files]
   git commit -m "fix: Critical bug in [component]"
   git push origin hotfix/vX.Y.Z-critical-bug-description
   ```

7. **Create merge request and merge to main after CI passes:**
   ```bash
   glab mr create --title "Hotfix vX.Y.Z: Critical bug" --target-branch main
   ```

8. **Follow normal release verification** (Phase 5)

**Hotfix criteria:**
- Production is broken or severely degraded
- Security vulnerability
- Data loss risk

**For non-critical bugs:** Follow normal sprint cycle

---

## Phase 6: Retrospective & Sprint Opening (Interactive - 15-30 minutes)

This phase is **mandatory and fully interactive**. Do not skip it, do not summarise it
into a checklist tick. Work through it conversationally with the user.

### Step 6.1: Release Retrospective

Open the retrospective explicitly. Ask the user each question in turn and wait for their
response before moving on. Do not batch all questions into one message.

Suggested opening: *"Now that the release is out, let's do a quick retrospective. I'll
ask you a few questions — feel free to be brief or go deep on anything that stands out."*

**Questions to work through:**

1. **What went well this sprint?**
   (Probe: anything that felt smooth, any tools/processes that worked better than expected?)

2. **What was harder than expected?**
   (Probe: scope creep, technical surprises, estimates that were badly off?)

3. **Did the estimate match reality?** Read the sprint document's estimate aloud, compare
   to actual time spent, and note the delta.

4. **What would you do differently next time?**
   (Probe: process changes, different task ordering, things to avoid?)

5. **Should anything be added to or changed in this checklist?**
   If yes, offer to edit this skill file right now.

**After the retrospective:** Summarise the key takeaways in 2-3 bullet points, then ask
the user to confirm before proceeding to sprint opening.

---

### Step 6.2: Review the Roadmap

**Always do this before reviewing the next sprint plan.** The roadmap determines which
sprint comes next and may have changed since the sprint file was written.

Read `ROADMAP.md` and present the planned sprint table to the user. Check:

- Sprint numbers, titles, and estimates match the sprint files
- The completed sprint is marked correctly in the Completed Sprints table
- The next sprint in the roadmap is still the right one to start
- Milestone assignments still make sense
- The Timeline Estimate table at the bottom is accurate

Ask: *"Does the roadmap look accurate and up to date? Do you want to do a different sprint next, or make any other changes before we look at the sprint plan?"*

Make any agreed edits now. **All edits in Phase 6 (roadmap, sprint plan, pointer, CODE_IMPROVEMENTS,
CHANGELOG, etc.) must be committed to `feature/sprint-N` — never directly to `main`.**

---

### Step 6.3: Review the Upcoming Sprint Plan

After the roadmap is confirmed, read `docs/sprints/SPRINT.md` (the pointer) to find the
next sprint file, then read it in full. Present the sprint to the user:

- Sprint number, title, and total estimate
- Full task list with estimates
- Dependencies on other sprints or tasks
- Any tasks that look under-specified or risky

Then ask: *"Does this sprint plan still look right to you? Is there anything you'd like
to restructure, re-estimate, or move before we kick it off?"*

**This is a genuine collaborative review — not a rubber-stamp.** If the plan looks too
heavy (as with Sprint 14 at 27–36h), proactively flag it and suggest restructuring.
Work through any changes interactively, making edits to the sprint files as agreed.

Common things to discuss:
- Is the sprint scope realistic given current velocity?
- Are any tasks better moved to a later sprint or the backlog?
- Are there deferred tasks from the *completed* sprint that need to be placed somewhere?
- Does any task need more detail before implementation starts?

---

### Step 6.3b: Review TODO.md for Sprint Candidates

Read `TODO.md` in full. For each unscheduled item, consider whether it belongs in the
upcoming sprint. Present any candidates to the user and ask whether to promote them.

Ask: *"Are there any items from TODO.md you'd like to pull into Sprint N before we kick it off?"*

Make any agreed edits to `TODO.md`, the sprint file, and `ROADMAP.md` now.

---

### Step 6.4: Update the Sprint Pointer

Confirm that `docs/sprints/SPRINT.md` points to the new sprint file. If it does not,
update it now.

```bash
cat docs/sprints/SPRINT.md   # Should show: See SPRINT-N.md
```

---

### Step 6.5: Commit Sprint Opening Changes

**All changes from Phase 6 belong on `feature/sprint-N`, not on `main`.** The new sprint
branch is the correct home for roadmap updates, CODE_IMPROVEMENTS housekeeping, sprint pointer
changes, and any sprint plan restructuring.

Create the branch first (if not already on it), then commit:

```bash
git checkout -b feature/sprint-N
git add docs/sprints/SPRINT.md docs/sprints/SPRINT-N.md ROADMAP.md CODE_IMPROVEMENTS.md
git commit -m "docs: open Sprint N — [title]

[1-2 sentence summary of any restructuring done during sprint opening]"
```

If no changes were needed, confirm the branch exists and is ready:
```bash
git checkout -b feature/sprint-N
```

---

### Step 6.6: Final Confirmation

Ask the user: *"Sprint N is now open. Is there anything else you want to sort out before
we start implementing, or are we good to go?"*

Wait for their response. Only conclude the checklist after they confirm.

---

*Last updated: 2026-05-31*
*Maintained by: Development team with Claude Code assistance*
