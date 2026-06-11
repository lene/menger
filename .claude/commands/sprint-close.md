# Sprint Close & New Sprint Start

Interactive workflow to close a completed sprint, verify the release, and collaboratively plan the next sprint.

**Usage:** `/sprint-close` — run when sprint work is complete, all commits are on the feature branch, and CI is green. The skill handles version bumps, archiving, MR creation and merge, and sprint opening.

**Required skill:** Invoke the `release-checklist` skill at the start of this command.
Phases 1–4 of the release-checklist (version bump, changelog, pre-push validation, commit/push)
are performed during Phase 3 of this skill (Sprint Archiving). Pick up the release-checklist
from **Phase 5 (Post-Release Verification)** and **Phase 6 (Retrospective & Sprint Opening)**
after the MR is merged in Phase 5c below.

---

## Phase 1: Gather Context

Run these commands and read the output:

```bash
git fetch origin --tags
git log origin/main --oneline -5
head -5 docs/sprints/SPRINT.md
```

Then ask the user:
> "Which sprint are we closing and what version is to be released (e.g. Sprint 13 / v0.5.3)?"

Store SPRINT_NUM and VERSION. Derive:
- NEXT_SPRINT_NUM = SPRINT_NUM + 1

---

## Phase 1b: Code Review

Run `/code-review` on the sprint branch before proceeding. This catches architectural drift,
dead code, and style issues while the work is still fresh and before the branch is archived.

```
/code-review
```

Wait for the review to complete. Categorise findings into three tiers:

- **Blockers** — must fix before archiving (security issues, broken contracts, data loss risk, broken invariants)
- **Should-fix** — worth fixing but not urgent (design problems, significant tech debt introduced, missing test coverage for tricky paths)
- **Notes** — informational (style, minor nits, speculative improvements)

If blockers exist, fix them on the feature branch before continuing to Phase 2.

For non-blocking items, build a prioritised list ordered by **impact ÷ effort** (highest wins
first). For each item evaluate:
- *Impact*: correctness risk, maintenance burden added, user-visible effect
- *Effort*: estimated time to fix (quick = minutes, medium = <1h, heavy = multiple hours)

Present the prioritised list to the user:

> "Here are the non-blocking review findings, ranked by impact vs effort. Which should we
> fix now before closing the sprint?"

For each item the user confirms, fix it immediately on the feature branch and commit.
Remaining items go into `CODE_IMPROVEMENTS.md` (for tech debt) or the next sprint plan
(for features/larger refactors).

---

## Phase 2: Pre-Merge CI Verification

### 2a. Next-To-Last Pre-merge push

If any files are still uncommitted, commit them. If there is any commit on the branch
that is not yet on `origin`, push and monitor the resulting pipeline.

### 2b. GitLab Pipeline Status

Get the main SHA and recent pipelines:

```bash
git rev-parse origin/main
```

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" \
  "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines?per_page=10" \
  | python3 -c "
import sys, json
ps = json.load(sys.stdin)
for p in ps:
    print(f\"{p['iid']:4d}  {p['sha'][:8]}  {p['ref']:<45s}  {p['status']:<10s}  {p['created_at'][:16]}\")"
```

Look for and verify these pipelines all show `success` on the main SHA:
- MR pipeline (`refs/merge-requests/N/head`)
- Tag pipeline (`refs/tags/vVERSION`)
- Main branch pipeline (`main`)

If any pipeline is `running` or `pending`, wait and retry. If `failed`, stop and report the failing job before proceeding.

---

## Phase 3: Sprint Archiving

### 3a. Archive Completed Sprint Plan

```bash
ls docs/archive/sprints/
head -3 docs/sprints/SPRINT.md
```

Move the completed sprint:

```bash
git mv docs/sprints/SPRINTSPRINT_NUM.md docs/archive/sprints/SPRINTSPRINT_NUM.md
```

(Replace SPRINT_NUM with the actual number, e.g. `git mv docs/sprints/SPRINT13.md docs/archive/sprints/SPRINT13.md`)

Update `docs/sprints/SPRINT.md` to point to the next sprint number.

### 3b. Clean docs/plans/

```bash
ls docs/plans/
```

Read each file and check if its work is complete (implemented in the archived sprint). Ask the user to confirm before removing anything that isn't clearly obsolete. Remove confirmed completed plan files with `git rm`.

### 3c. Update ROADMAP.md

```bash
grep -n "Sprint SPRINT_NUM\|vVERSION" ROADMAP.md
```

Read the relevant section and:
- Mark the completed sprint milestone as `✅ Complete`
- Update next sprint status to `🔄 In Progress`

### 3d. Check arc42 Staleness

```bash
grep "Last Updated" docs/arc42/README.md
git log --after="ARC42_DATE" --oneline -- menger-app/src optix-jni/src
```

(Use the date from the arc42 README.) If significant source changes exist since the last arc42 update, list the affected files and ask the user which sections may need updating. Do not update arc42 automatically.

### 3e. Refresh Review Guidelines

Read `standards/review-guidelines.md` and `standards/architecture-review-guidelines.md`.
Check whether any sprint findings revealed gaps or outdated guidance. If updates are
needed, edit and commit them now while the sprint work is fresh.

Ask the user:
> "Did this sprint surface anything that should update the review guidelines? (y/n)"

If yes, apply the edits and stage them for the archiving commit (3i).

### 3f. Review ENFORCEMENT.md

```bash
cat docs/ENFORCEMENT.md | grep "❌"
```

Check the open-issues table. For each ❌ row:
1. Has it been resolved since ENFORCEMENT.md was last updated? If yes, update the row to ✅ and close the linked GitLab issue.
2. Is the linked GitLab issue still open and accurate?

Ask the user:
> "Any enforcement gaps closed this sprint that we should mark ✅? (y/n)"

If yes, apply edits and stage for the archiving commit (3i).

### 3g. Run Standards Drift Check

```bash
./scripts/check-standards-drift.sh --local
```

Fix any drift found before archiving. If sibling repos have diverged, run
`./scripts/sync-standards.sh` and open PRs in the affected repos.

### 3h. Trigger Dependency Updates

Trigger a Renovate run so fresh dependency bump MRs/PRs are ready at the start of the new sprint:

```bash
glab pipeline run --ref main --variables "RUN_RENOVATE=true"
```

For the GitHub repos, the Renovate app runs on its own schedule; open the Renovate dashboard to trigger immediately if needed. Check for open dependency PRs in both GitHub repos before closing:

```bash
gh pr list --repo lene/menger-common --label dependencies
gh pr list --repo lene/optix-jni --label dependencies
```

### 3i. Clean CODE_IMPROVEMENTS.md

Read `CODE_IMPROVEMENTS.md` in full. Identify issues marked as resolved, complete, or fixed (look for ✅, "Resolved:", "Fixed in", "Closed"). Present the list:

> "These CODE_IMPROVEMENTS.md issues appear resolved — remove them? (y/n/all)"

Remove confirmed ones. Do not touch deferred or in-progress items.

### 3j. Version Consistency Check

```bash
grep "version" menger-app/build.sbt
grep "DEPLOYABLE_VERSION" .gitlab-ci.yml
grep "menger v" menger-app/src/main/scala/menger/MengerCLIOptions.scala
grep '^\*\*Version\*\*:' docs/guide/user-guide.md
grep '^\*\*Version\*\*:' docs/USER_GUIDE.md
grep "^\## \[" CHANGELOG.md | head -1
```

All six must agree on VERSION and CHANGELOG top entry must be `[VERSION] - TODAY`. Report any mismatches as ❌.

### 3k. Commit Archiving Work

```bash
git add docs/archive/sprints/
git add docs/sprints/SPRINT.md
git add ROADMAP.md
git add CODE_IMPROVEMENTS.md
git add docs/plans/
```

Commit: `docs: archive sprint SPRINT_NUM, clean up for sprint NEXT_SPRINT_NUM`

---

## Phase 4: Interactive Sprint Planning

### 4a. Load and Summarise Next Sprint

Read `docs/sprints/SPRINTNEXT_SPRINT_NUM.md` and present a summary:

> "Sprint NEXT_SPRINT_NUM — **TITLE**
> Tasks: [numbered list with estimates]
> Total estimate: ~X hours"

### 4b. Collaborative Task Review

Walk through each task and ask:
1. Still relevant? (y/n/modified)
2. Priority change needed?
3. Estimate still accurate?

Then prompt:

> "Any items from CODE_IMPROVEMENTS.md to add to Sprint NEXT_SPRINT_NUM?"

```bash
grep -E "^### [HM][0-9]" CODE_IMPROVEMENTS.md
```

> "Any technical debt from arc42 to schedule?"

```bash
grep -A2 "^### TD-" docs/arc42/11-risks-and-technical-debt.md
```

### 4c. Dependency Upgrade Check

Check all subprojects for outdated dependencies:

```bash
grep -E '"%' menger-app/build.sbt menger-common/build.sbt optix-jni/build.sbt project/plugins.sbt 2>/dev/null | grep -v '//'
```

Also check any non-sbt package managers present (npm, pip, etc.):

```bash
find . -name "package.json" -not -path "*/node_modules/*" | head -5
```

Ask the user:
> "Upgrade all dependencies to newest compatible versions now? (y/n)"

If yes, for each dependency look up the latest version and update the relevant build file. After all updates:
- Run `./.git_hooks/pre-push` to verify CI gates pass
- Commit: `chore(deps): upgrade all dependencies to latest versions`

### 4d. Update Sprint Plan if Changed

If the user approved any changes, update `docs/sprints/SPRINTNEXT_SPRINT_NUM.md` to reflect the agreed scope.

Present the agreed task list and ask:
> "Confirm final Sprint NEXT_SPRINT_NUM scope? (y/n)"

## Phase 5: Final Push and Merge

### 5a. Last Pre-merge push

If any files are still uncommitted, commit them. If there is any commit on the branch
that is not yet on `origin`, push and monitor the resulting pipeline.

### 5b. GitLab Pipeline Status

Get the main SHA and recent pipelines:

```bash
git rev-parse origin/main
```

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" \
  "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines?per_page=10" \
  | python3 -c "
import sys, json
ps = json.load(sys.stdin)
for p in ps:
    print(f\"{p['iid']:4d}  {p['sha'][:8]}  {p['ref']:<45s}  {p['status']:<10s}  {p['created_at'][:16]}\")"
```

Look for and verify these pipelines all show `success` on the main SHA:
- MR pipeline (`refs/merge-requests/N/head`)
- Tag pipeline (`refs/tags/vVERSION`)
- Main branch pipeline (`main`)

If any pipeline is `running` or `pending`, wait and retry. If `failed`, stop and report the failing job before proceeding.

### 5c. Create and Merge MR

Create the MR if it doesn't exist yet:

```bash
glab mr create \
  --title "Sprint SPRINT_NUM: TITLE (vVERSION)" \
  --description "Closes Sprint SPRINT_NUM. See CHANGELOG.md for release notes." \
  --target-branch main \
  --squash=false \
  --remove-source-branch=false
```

Then wait for the MR pipeline to pass (`glab ci status`). Once green, merge:

```bash
glab mr merge --yes --squash=false
```

Confirm the merge completed:

```bash
git fetch origin main
git log origin/main --oneline -3
```

The merge triggers the release pipeline (CreateRelease → tag → PushToGithub).

### 5d. PushToGithub Job

Find the tag pipeline ID from the output above, then inspect its jobs:

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" \
  "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines/PIPELINE_ID/jobs" \
  | python3 -c "
import sys, json
jobs = json.load(sys.stdin)
for j in jobs:
    print(f\"{j['name']:<30s}  {j['status']:<10s}  {j.get('failure_reason','') or ''}\")"
```

Report the status of each job. The `PushToGithub` and `CreateGithubRelease` jobs must be `success`.

### 5e. GitHub Mirror Verification

```bash
git ls-remote github refs/heads/main
git ls-remote github refs/tags/vVERSION
```

Verify:
- `refs/heads/main` SHA matches `origin/main`
- `refs/tags/vVERSION` exists and points to same SHA

Report ✅ or ❌ for each. If GitHub mirror is wrong, suggest re-running the PushToGithub job from the GitLab UI.

## Phase 6: Create New Sprint

### 6a. Ensure Sprint Branch Exists

**IMPORTANT:** The completed sprint MR has already been merged. Do NOT commit directly to
the old sprint branch (`feature/sprint-SPRINT_NUM`) — it is done. All archiving work and
the sprint kickoff commit must go on `feature/sprint-NEXT_SPRINT_NUM`.

Create the new sprint branch from `origin/main` and cherry-pick any archiving commits
that were made on the old branch after the merge:

```bash
git log --oneline feature/sprint-SPRINT_NUM ^origin/main
# lists commits NOT on main yet — cherry-pick these onto the new branch

git fetch origin main
git checkout -b feature/sprint-NEXT_SPRINT_NUM origin/main
git cherry-pick <sha1> <sha2> ...   # cherry-pick archiving commits from old branch
git push -u origin feature/sprint-NEXT_SPRINT_NUM
```

If you haven't committed any archiving work yet, simply create the branch and proceed.

### 6b. Commit Sprint Kickoff

```bash
git add docs/sprints/SPRINTNEXT_SPRINT_NUM.md
```

Commit: `docs: kick off sprint NEXT_SPRINT_NUM — TITLE`

```bash
git push -u origin feature/sprint-NEXT_SPRINT_NUM
```

---

## Phase 7: Summary Report

Print this report, filling in ✅ or ❌ for each item:

```
╔══════════════════════════════════════════════════════╗
║         Sprint SPRINT_NUM Close Summary              ║
╠══════════════════════════════════════════════════════╣
║ Release vVERSION                                     ║
║   GitLab MR pipeline:       ✅/❌                    ║
║   GitLab tag pipeline:      ✅/❌                    ║
║   PushToGithub job:         ✅/❌                    ║
║   GitHub mirror (SHA+tag):  ✅/❌                    ║
╠══════════════════════════════════════════════════════╣
║ Housekeeping                                         ║
║   Sprint plan archived:     ✅/❌                    ║
║   docs/plans/ cleaned:      ✅/❌  (N files removed) ║
║   ROADMAP.md updated:       ✅/❌                    ║
║   CODE_IMPROVEMENTS:        ✅/❌  (N issues removed)║
║   Version consistency:      ✅/❌                    ║
║   arc42 staleness:          ✅/⚠️ (flagged/ok)       ║
╠══════════════════════════════════════════════════════╣
║ Sprint NEXT_SPRINT_NUM Ready                         ║
║   Branch: feature/sprint-NEXT_SPRINT_NUM             ║
║   Tasks agreed: N tasks, ~X hours                    ║
╚══════════════════════════════════════════════════════╝
```

List any remaining ❌ items with specific actions needed to resolve them.
