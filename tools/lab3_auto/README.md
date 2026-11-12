# Lab 3 automatic local runner

Run from PowerShell:

```powershell
& './tools/lab3_auto/Run-Lab3Auto.ps1'
```

Preflight: add `-CheckOnly`. Pilot: add `-Tasks cp2_slice,cp4_curb`.
The installed Docker image is reused, avoiding downloads on normal reruns.
Add `-Rebuild` only when intentionally rebuilding the image (requires network).
After the course releases reference data, `-Improve` compares YOLO11l+SAM and
Mask2Former COCO candidates for Medium/Hard. It generates masks from images only,
evaluates against the released reference, creates separate improved CVAT tasks,
and replaces a ZIP only if its re-exported score improves. Easy is preserved.
Backups/evaluation are Git-ignored in `.lab3-improve/`; see `IMPROVEMENT_REPORT.md`.
The runner processes only the eight non-Easy tasks. It requires the existing Ubuntu WSL,
Docker Desktop, CVAT stack and SAM function installed for this workspace. No passwords
are requested: a short-lived Python process inside `cvat_server` calls CVAT REST views
as the owner of Easy task 3. It does not create credentials or write annotations directly
to database tables. Docker socket access is restricted to the local runner container.

Outputs: genuine CVAT ZIP exports in `submissions/`, overlays in `automation_previews/`,
and the factual `AUTOMATION_RUN_REPORT.md`. No automatic GitHub/LMS submission.
Private resumable state, predictions and backups live in `.lab3-auto/` (Git-ignored).
Model downloads are cached in `/home/ad/sam-setup/auto-cache` in Ubuntu.

Rerunning verifies completed exports and annotation hashes without duplicating tasks.
It refuses to overwrite manual edits in managed tasks or reuse changed inputs/configuration.
Keep the state directory to resume interrupted uploads safely. Do not run two copies.
If a task fails, consult the report; other independent tasks continue. Empty outputs are
not presented as completed work. Existing ZIPs are backed up before replacement.

YOLO11n/COCO detects supported things; SAM ViT-B generates their masks. SegFormer-B0
Cityscapes provides semantic classes. Automatic predictions are fallible, especially
on small/occluded objects, thin structures and functional boundaries. Void pixels are
not forced into a class. These outputs do not establish that the student's required
manual annotation and review steps have happened.
