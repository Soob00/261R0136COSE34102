"""30초마다 PROGRESS.md를 갱신하는 스크립트 (auto_checkpoint.sh에서 호출)"""
import json, re
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
STATUS_FILE  = BASE / "status.json"
RESULT_FILE  = BASE / "results.json"
LOG_DIR      = BASE / "logs"
OUTPUT_FILE  = BASE / "PROGRESS.md"

EXPERIMENTS = ["Baseline", "Masking Cons Reg", "Naive Swap", "Validity-Gated",
               "VG_lam=0.05", "VG_lam=0.2"]
TOTAL_EXP = len(EXPERIMENTS)
TOTAL_SEEDS = 3

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except:
        return {}

def latest_log_lines(n=15):
    tqdm_line = ""
    meaningful = []

    # tmux pane에서 직접 읽기 (로그 파일 없을 때 대체)
    try:
        import subprocess
        pane_out = subprocess.check_output(
            ["tmux", "capture-pane", "-t", "validity_gated:train.0", "-p"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8", errors="replace")
        for line in reversed(pane_out.splitlines()):
            line = line.strip()
            if line.startswith("train:") and ("s/it" in line or "it/s" in line):
                tqdm_line = line
                break
        meaningful = [l.strip() for l in pane_out.splitlines()
                      if l.strip() and any(k in l for k in ["ep", "test F1", "flip=", "Experiment:", "seed=", "Loading K-HATERS"])]
        meaningful = meaningful[-n:]
    except Exception:
        pass

    # 로그 파일도 있으면 병합
    logs = sorted(LOG_DIR.glob("run_*.log"), key=lambda p: p.stat().st_mtime)
    if logs and logs[-1].stat().st_size > 0:
        raw_bytes = logs[-1].read_bytes()
        if not tqdm_line:
            chunks = [c.decode("utf-8", errors="replace").strip() for c in raw_bytes.split(b"\r")]
            for chunk in reversed(chunks):
                if chunk.startswith("train:") and ("s/it" in chunk or "it/s" in chunk):
                    tqdm_line = chunk
                    break
        text = raw_bytes.decode("utf-8", errors="replace").replace("\r", "\n")
        file_lines = [l.strip() for l in text.splitlines()
                      if l.strip() and any(k in l for k in ["ep", "test F1", "flip=", "Results saved", "Experiment:", "seed=", "Loading K-HATERS"])]
        meaningful = (meaningful + file_lines)[-n:]

    return meaningful, tqdm_line

def fmt_elapsed(start_iso):
    try:
        start = datetime.fromisoformat(start_iso)
        delta = datetime.now() - start
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    except:
        return "?"

def make_progress_bar(done, total, width=20):
    filled = int(width * done / total) if total else 0
    return f"[{'█'*filled}{'░'*(width-filled)}] {done}/{total}"

def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status  = read_json(STATUS_FILE)
    results = read_json(RESULT_FILE)

    cur_exp    = status.get("current_exp", "?")
    cur_seed   = status.get("current_seed", "?")
    progress   = status.get("progress", "?")       # e.g. "2/3 seeds"
    st         = status.get("status", "?")
    start_time = status.get("start_time", "")
    logs       = status.get("logs", [])

    # results를 {tag: {...}} dict로 정규화
    if isinstance(results, list):
        norm = {}
        for r in results:
            tag = r.get("tag", "?")
            seeds = r.get("seeds", [])
            norm[tag] = {
                "f1":         [s["test_f1"]   for s in seeds if "test_f1"   in s],
                "flip_rate":  [s["flip_rate"] for s in seeds if "flip_rate" in s],
                "logit_gap":  [s["logit_gap"] for s in seeds if "logit_gap" in s],
                "fpr_gap":    [s["fpr_gap"]   for s in seeds if "fpr_gap"   in s],
            }
        results = norm

    # 완료된 실험 수 추정
    if results:
        done_exp = len(results)
    elif cur_exp in EXPERIMENTS:
        done_exp = EXPERIMENTS.index(cur_exp)
    else:
        done_exp = 0

    seed_done = 0
    if progress:
        m = re.match(r"(\d+)/(\d+)", progress)
        if m:
            seed_done = int(m.group(1)) - 1  # 현재 seed는 진행 중이므로 -1

    total_seeds_done = done_exp * TOTAL_SEEDS + seed_done

    lines = []
    lines.append(f"# Validity-Gated Experiment Progress")
    lines.append(f"")
    lines.append(f"> 갱신: {now}  |  경과: {fmt_elapsed(start_time)}")
    lines.append(f"")

    # 전체 진행률
    lines.append(f"## 전체 진행률")
    lines.append(f"")
    overall_bar = make_progress_bar(done_exp, TOTAL_EXP)
    lines.append(f"- 실험: {overall_bar}")
    seeds_bar = make_progress_bar(total_seeds_done, TOTAL_EXP * TOTAL_SEEDS)
    lines.append(f"- Seed:  {seeds_bar}")
    lines.append(f"")

    # 현재 상태
    status_emoji = {"running": "🟢 실행 중", "completed": "✅ 완료", "initializing": "🔵 초기화 중"}.get(st, st)
    lines.append(f"## 현재 상태: {status_emoji}")
    lines.append(f"")
    if st != "completed":
        lines.append(f"| 항목 | 값 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 실험 | `{cur_exp}` |")
        lines.append(f"| 진행 | {progress} |")
        lines.append(f"| Seed | {cur_seed} |")
    lines.append(f"")

    # 실험 목록
    lines.append(f"## 실험 목록")
    lines.append(f"")
    for exp in EXPERIMENTS:
        if exp in results:
            r = results[exp]
            f1_vals = r.get("f1", [])
            flip_vals = r.get("flip_rate", [])
            f1_str   = f"{sum(f1_vals)/len(f1_vals):.4f}" if f1_vals else "?"
            flip_str = f"{sum(flip_vals)/len(flip_vals):.4f}" if flip_vals else "?"
            lines.append(f"- ✅ `{exp}` — F1: **{f1_str}**, Flip: **{flip_str}**")
        elif exp == cur_exp:
            lines.append(f"- 🟢 `{exp}` — 진행 중 ({progress})")
        else:
            lines.append(f"- ⬜ `{exp}`")
    lines.append(f"")

    # 최근 로그
    recent, tqdm_line = latest_log_lines(15)
    lines.append(f"## 학습 진행")
    lines.append(f"")
    if tqdm_line:
        # "train:  20%|█▉  | 535/2690 [09:52<41:52, 1.17s/it]" 파싱
        m = re.search(r'(\d+)/(\d+)\s+\[([^\]]+)\]', tqdm_line)
        if m:
            step, total_steps, timing = m.group(1), m.group(2), m.group(3)
            elapsed, remaining = (timing.split("<") + ["?"])[:2]
            pct = int(step) * 100 // int(total_steps)
            bar = make_progress_bar(int(step), int(total_steps), width=30)
            lines.append(f"**현재 epoch 진행률**")
            lines.append(f"")
            lines.append(f"`{bar}`")
            lines.append(f"")
            lines.append(f"- step: {step} / {total_steps} ({pct}%)")
            lines.append(f"- 경과: {elapsed.strip()} / 남은 시간: {remaining.strip()}")
        else:
            lines.append(f"```\n{tqdm_line}\n```")
    lines.append(f"")
    lines.append(f"## 최근 로그")
    lines.append(f"")
    lines.append(f"```")
    if recent:
        lines.extend(recent)
    elif logs:
        lines.extend(logs[-10:])
    else:
        lines.append("(로그 없음 — 학습 시작 전)")
    lines.append(f"```")
    lines.append(f"")

    # 완료된 결과 테이블
    if results:
        lines.append(f"## 결과 (완료된 실험)")
        lines.append(f"")
        lines.append(f"| 실험 | F1 | Flip Rate | Logit Gap | FPR Gap |")
        lines.append(f"|------|----|-----------|-----------|---------|")
        for name, r in results.items():
            def fmt(v):
                return f"{sum(v)/len(v):.4f}±{(sum((x-sum(v)/len(v))**2 for x in v)/len(v))**0.5:.4f}" if v else "N/A"
            lines.append(f"| `{name}` | {fmt(r.get('f1',[]))} | {fmt(r.get('flip_rate',[]))} | {fmt(r.get('logit_gap',[]))} | {fmt(r.get('fpr_gap',[]))} |")
        lines.append(f"")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
