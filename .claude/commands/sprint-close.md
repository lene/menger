# Sprint Close & New Sprint Start

Interactive workflow to close a completed sprint, verify the release, and collaboratively plan the next sprint.

**Usage:** `/sprint-close` — run after the sprint MR has been merged in the GitLab web UI.

---

## Phase 1: Gather Context

First, collect the current state.

```bash
!`git fetch origin --tags 2>&1 | head -5`
!`git log origin/main --oneline -5`
!`cat docs/sprints/SPRINT.md | head -5`
```

Ask the user:
> "Which sprint was just closed and what version was released (e.g. Sprint 13 / v0.5.3)?"

Store these as SPRINT_NUM and VERSION for the rest of the workflow. Then derive:
- PREV_SPRINT_FILE: `docs/sprints/SPRINT{SPRINT_NUM}.md`
- NEXT_SPRINT_NUM: SPRINT_NUM + 1
- NEXT_SPRINT_FILE: `docs/sprints/SPRINT{NEXT_SPRINT_NUM}.md`

---

## Phase 2: Post-Merge CI Verification

### 2a. GitLab Pipeline Status

Using the `GITLAB_ACCESS_TOKEN` environment variable, fetch the latest pipelines for the merged SHA on main:

```bash
!`MAIN_SHA=$(git rev-parse origin/main); echo "Main SHA: $MAIN_SHA"`
!`curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines?per_page=10" | python3 -c "
import sys, json
ps = json.load(sys.stdin)
for p in ps:
    print(f\"{p['iid']:4d}  {p['sha'][:8]}  {p['ref']:<45s}  {p['status']:<10s}  {p['created_at'][:16]}\")"
`
```

Look for:
- The **MR pipeline** (`refs/merge-requests/N/head`) on the released SHA — must be `success`
- The **tag pipeline** (`refs/tags/vVERSION`) on the same SHA — must be `success`
- The **main pipeline** (`main`) — must be `success`

If any pipeline is still `running` or `pending`, wait 60 seconds and retry. If any is `failed`, report the specific job that failed and stop — the release needs to be investigated first.

### 2b. PushToGithub Job

Fetch the tag pipeline ID, then check the PushToGithub job specifically:

```bash
!`curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines?ref=v${VERSION}&per_page=3" | python3 -c "
import sys, json
ps = json.load(sys.stdin)
for p in ps: print(p['id'], p['status'], p['ref'])"
`
```

```bash
!`PIPELINE_ID=<from above>; curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines/${PIPELINE_ID}/jobs" | python3 -c "
import sys, json
jobs = json.load(sys.stdin)
for j in jobs: print(f\"{j['name']:<30s}  {j['status']:<10s}  {j.get('failure_reason','') or ''}\")"
`
```

Report the status of each job, highlighting any failures.

### 2c. GitHub Mirror Verification

Check that the GitHub remote has the correct commit and tag:

```bash
!`git ls-remote github refs/heads/main refs/tags/v${VERSION} 2>/dev/null || git ls-remote git@github.com:lene/menger.git refs/heads/main refs/tags/v${VERSION}`
```

Verify:
- `refs/heads/main` SHA matches `origin/main` SHA
- `refs/tags/v{VERSION}` exists and points to the same commit

Report ✅ or ❌ for each check. If GitHub mirror is wrong, report the discrepancy and suggest manually re-running the PushToGithub job from GitLab.

---

## Phase 3: Sprint Archiving

### 3a. Archive Completed Sprint Plan

```bash
!`ls docs/archive/sprints/`
!`cat docs/sprints/SPRINT.md | head -3`
```

Move the completed sprint file to the archive:

```bash
git mv docs/sprints/SPRINT${SPRINT_NUM}.md docs/archive/sprints/SPRINT${SPRINT_NUM}.md
```

Update `docs/sprints/SPRINT.md` to point to the next sprint number.

### 3b. Clean docs/plans/

List all files in `docs/plans/` and check each one against the archived sprint to determine if it's completed work:

```bash
!`ls docs/plans/`
```

For each file, check whether its content relates to completed sprint work. Remove any plan files whose work is captured in the archived sprint or code. Ask the user to confirm before removing anything that isn't clearly obsolete.

### 3c. Update ROADMAP.md

```bash
!`grep -n "Sprint ${SPRINT_NUM}\|v${VERSION}" ROADMAP.md | head -10`
```

- Mark the completed sprint milestone as `✅ Complete`
- Update the next sprint/milestone status to `🔄 In Progress` or `📋 Planned`

### 3d. Update arc42 if Needed

```bash
!`cat docs/arc42/README.md | grep "Last Updated"`
!`git log --since="$(cat docs/arc42/README.md | grep 'Last Updated' | grep -oP '\d{4}-\d{2}-\d{2}')" --oneline -- 'menger-app/src' 'optix-jni/src' | wc -l`
```

If there have been significant architectural changes since the arc42 docs were last updated, flag this to the user with a list of changed source files and suggest updating the relevant sections. Do not update arc42 automatically — ask the user.

### 3e. Clean CODE_IMPROVEMENTS.md

```bash
!`grep -c "Resolved\|✅\|DONE\|Fixed" CODE_IMPROVEMENTS.md`
```

Read the full `CODE_IMPROVEMENTS.md`. Identify any issues marked as resolved, fixed, or completed. Present a list to the user:

> "The following issues in CODE_IMPROVEMENTS.md appear to be resolved. Shall I remove them? (y/n for each, or 'all')"

Remove confirmed resolved issues. Do not remove issues that are deferred or in progress.

### 3f. Verify Version Consistency

```bash
!`grep -n "version" menger-app/build.sbt | head -3`
!`grep -n "DEPLOYABLE_VERSION" .gitlab-ci.yml | head -3`
!`grep -n "menger v" menger-app/src/main/scala/menger/MengerCLIOptions.scala | head -3`
!`grep -n "version" docs/USER_GUIDE.md | head -3`
```

All four locations must agree on version `VERSION`. Report any mismatches.

### 3g. Commit Sprint Archive Changes

Stage and commit the archiving work:

```bash
git add docs/archive/sprints/SPRINT${SPRINT_NUM}.md
git add docs/sprints/SPRINT.md
git add docs/sprints/SPRINT${NEXT_SPRINT_NUM}.md   # updated pointer
git add ROADMAP.md
git add CODE_IMPROVEMENTS.md
# Any removed docs/plans/ files
```

Commit message: `docs: archive sprint ${SPRINT_NUM}, clean up for sprint ${NEXT_SPRINT_NUM}`

---

## Phase 4: Interactive Sprint Planning

### 4a. Load Next Sprint

```bash
!`cat docs/sprints/SPRINT${NEXT_SPRINT_NUM}.md`
```

Present a summary of the next sprint's tasks to the user:

> "Here is the planned scope for Sprint {NEXT_SPRINT_NUM}: **{sprint title}**
>
> Tasks:
> {numbered list of tasks with estimates}
>
> Total estimate: X hours"

### 4b. Collaborative Task Review

For each task, ask:
1. Is this still relevant? (y/n/modified)
2. Should the priority change?
3. Are there new tasks to add from CODE_IMPROVEMENTS.md, ROADMAP.md, or recent findings?

Specifically prompt with:
> "Are there any items from CODE_IMPROVEMENTS.md that should be scheduled for Sprint {NEXT_SPRINT_NUM}?"

```bash
!`cat CODE_IMPROVEMENTS.md | grep -E "^### (H|M)[0-9]" | head -20`
```

> "Any technical debt items from arc42 Section 11 to include?"

```bash
!`grep -A2 "^### TD-" docs/arc42/11-risks-and-technical-debt.md | head -30`
```

After the discussion, present the final agreed task list for confirmation:

> "Sprint {NEXT_SPRINT_NUM} final scope:
> {agreed task list}
>
> Approve? (y/n)"

### 4c. Update Sprint Plan

If the user approved changes to the sprint plan, update `docs/sprints/SPRINT{NEXT_SPRINT_NUM}.md` to reflect the agreed scope.

### 4d. Create Sprint Branch

```bash
git fetch origin main
git checkout -b feature/sprint-${NEXT_SPRINT_NUM} origin/main
git push -u origin feature/sprint-${NEXT_SPRINT_NUM}
```

### 4e. Final Commit

Stage any sprint plan updates and commit to the new branch:

```bash
git add docs/sprints/SPRINT${NEXT_SPRINT_NUM}.md
git commit -m "docs: kick off sprint ${NEXT_SPRINT_NUM} — {sprint title}"
git push
```

---

## Phase 5: Summary Report

Present a final status report:

```
╔══════════════════════════════════════════════════════╗
║           Sprint {SPRINT_NUM} Close Summary          ║
╠══════════════════════════════════════════════════════╣
║ Release v{VERSION}                                   ║
║   GitLab pipeline:    ✅/❌                          ║
║   Tag pipeline:       ✅/❌                          ║
║   PushToGithub:       ✅/❌                          ║
║   GitHub mirror:      ✅/❌                          ║
╠══════════════════════════════════════════════════════╣
║ Housekeeping                                         ║
║   Sprint plan archived:     ✅/❌                    ║
║   docs/plans/ cleaned:      ✅/❌  (N files removed) ║
║   ROADMAP.md updated:       ✅/❌                    ║
║   CODE_IMPROVEMENTS cleaned:✅/❌  (N issues removed)║
║   Version consistency:      ✅/❌                    ║
╠══════════════════════════════════════════════════════╣
║ Sprint {NEXT_SPRINT_NUM} Ready                       ║
║   Branch: feature/sprint-{NEXT_SPRINT_NUM}           ║
║   Tasks agreed: N tasks, ~X hours                    ║
╚══════════════════════════════════════════════════════╝
```

If any ❌ items remain, list the specific actions needed to resolve them.
