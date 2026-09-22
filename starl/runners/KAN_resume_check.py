# =============================================================================
# STARL v2 — KAN resume check:  "do I RESUME, or move to the next task?"
# =============================================================================
# Paste this as the FIRST cell of a fresh notebook, with the PREVIOUS KAN output
# zip attached as a dataset. It reads the saved resume state and prints exactly
# what to put in CELL 2 of KAN_runner.py. Costs seconds; no GPU needed.

import glob, os, zipfile, torch

CAND = sorted(glob.glob("/kaggle/input/**/resume_state.pt", recursive=True))
if not CAND:                                   # dataset kept the zip un-extracted
    zips = sorted(glob.glob("/kaggle/input/**/*_kan_outputs.zip", recursive=True))
    assert zips, "No resume_state.pt or *_kan_outputs.zip under /kaggle/input — attach the KAN output dataset."
    with zipfile.ZipFile(zips[0]) as z:
        member = next((n for n in z.namelist() if n.endswith("resume_state.pt")), None)
        assert member, f"{zips[0]} contains no resume_state.pt"
        z.extract(member, "/kaggle/working/_resume_check")
    CAND = [os.path.join("/kaggle/working/_resume_check", member)]

# weights_only=False: our own file, and it stores numpy RNG state that torch>=2.6
# would otherwise refuse to unpickle.
s = torch.load(CAND[0], map_location="cpu", weights_only=False)

CL_PLAN   = ["naive", "rehearsal", "lwf", "ewc", "frozen_probe", "independent"]
BASE_PLAN = ["base", "frozen_probe"]
plan = BASE_PLAN if s["task"] == "T1_APTOS" else CL_PLAN
nxt  = plan[s["stage_idx"]] if s["stage_idx"] < len(plan) else "-"

print("resume file :", CAND[0])
print("task        :", s["task"])
print("task_done   :", s["task_done"])
print("stages done :", s["completed"], f"({s['stage_idx']}/{len(plan)})")
print("next stage  :", nxt, "| interrupted mid-stage:", s["strategy"], "at epoch", s["epoch"])
print("metric rows :", len(s["final_rows"]))
print()
if s["task_done"]:
    print(f">>> {s['task']} IS COMPLETE — move to the NEXT task.")
    print(f"    CELL 2:  TASK = '<next task>'   RESUME = False")
    print(f"    Attach this zip (holds expert_{s['task']}_kan.pth) + the new task's image dataset.")
else:
    print(f">>> {s['task']} IS NOT FINISHED — resume it.")
    print(f"    CELL 2:  TASK = '{s['task']}'   RESUME = True")
    print(f"    Attach this same zip; it picks up at stage '{nxt}', epoch {s['epoch'] + 1}.")
    print(f"    WARNING: expert_{s['task']}_kan.pth may already exist (rehearsal writes it")
    print( "             early, on each val improvement) — that is NOT proof the task is done.")
    print( "             Moving on now would use a half-trained expert AND lose the remaining baselines.")
