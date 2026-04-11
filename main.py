import datetime
import json
import os
import sys
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env if present
_env_file = pathlib.Path(__file__).with_name(".env")
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8-sig").splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import (
    EndTrialRequest,
    SubmitRunRequest,
    EvalPolicy,
    StartTrialRequest,
    GetBenchmarkRequest,
    StatusRequest,
    StartRunRequest,
)
from connectrpc.errors import ConnectError

from agent import run_agent

BITGN_URL = os.getenv("BITGN_HOST") or "https://api.bitgn.com"
BITGN_API_KEY = os.getenv("BITGN_API_KEY") or ""
BENCH_ID = os.getenv("BENCH_ID") or "bitgn/pac1-dev"
MODEL_ID = os.getenv("MODEL_ID") or "gpt-5.4"
WORKERS = int(os.getenv("WORKERS") or "8")

USE_COLOR = sys.stdout.isatty()
CLI_RED = "\x1B[31m" if USE_COLOR else ""
CLI_GREEN = "\x1B[32m" if USE_COLOR else ""
CLI_CLR = "\x1B[0m" if USE_COLOR else ""

_print_lock = threading.Lock()


def _console(worker_num: int, msg: str) -> None:
    with _print_lock:
        print(f"[worker-{worker_num}] {msg}", flush=True)


class LogFile:
    """Thread-safe append-only JSONL log file. Survives mid-run interruptions."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Write header line so the file exists immediately
        self._append({"event": "start", "ts": _ts(), "bench": BENCH_ID, "model": MODEL_ID})

    def write_trial(self, entry: dict) -> None:
        self._append({"event": "trial", "ts": _ts(), **entry})

    def write_summary(self, scores: list[tuple[str, float]]) -> None:
        total = sum(s for _, s in scores) / len(scores) * 100.0 if scores else 0.0
        self._append({"event": "summary", "ts": _ts(), "final_pct": round(total, 2),
                      "scores": {tid: s for tid, s in sorted(scores)}})

    def _append(self, obj: dict) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def run_trial(
    trial_id: str,
    client: HarnessServiceClientSync,
    task_filter: list[str],
    worker_num: int,
    log_file: LogFile,
) -> tuple[str, float, list[str]] | None:
    trial = client.start_trial(StartTrialRequest(trial_id=trial_id))

    if task_filter and trial.task_id not in task_filter:
        return None

    _console(worker_num, f"started  {trial.task_id}")

    sink: list[str] = []
    try:
        run_agent(MODEL_ID, trial.harness_url, trial.instruction, sink=sink)
    except Exception as exc:
        sink.append(f"[agent error] {exc}\n")
        _console(worker_num, f"{CLI_RED}error on {trial.task_id}: {exc}{CLI_CLR}")

    result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
    score = result.score if result.score >= 0 else 0.0
    score_detail = list(result.score_detail)

    style = CLI_GREEN if score == 1 else CLI_RED
    _console(worker_num, f"finished {trial.task_id}  score={style}{score:.2f}{CLI_CLR}")

    # Write to log immediately — survives Ctrl+C after this point
    log_file.write_trial({
        "task_id": trial.task_id,
        "trial_id": trial_id,
        "worker": worker_num,
        "score": score,
        "score_detail": score_detail,
        "log": "".join(sink),
    })

    return trial.task_id, score, score_detail


def main() -> None:
    task_filter = sys.argv[1:]
    scores: list[tuple[str, float]] = []

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = pathlib.Path(f"run_{ts}.jsonl")
    log_file = LogFile(log_path)
    print(f"Logging to {log_path}")

    try:
        client = HarnessServiceClientSync(BITGN_URL)

        print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks."
        )

        run = client.start_run(StartRunRequest(
            name="SGR NextStep Sample",
            benchmark_id=BENCH_ID,
            api_key=BITGN_API_KEY,
        ))

        trial_ids = list(run.trial_ids)
        workers = min(WORKERS, len(trial_ids)) if trial_ids else 1
        print(f"Dispatching {len(trial_ids)} trials across {workers} workers...")

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run_trial, tid, client, task_filter, i % workers + 1, log_file): tid
                    for i, tid in enumerate(trial_ids)
                }
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is None:
                        continue
                    task_id, score, _ = result
                    scores.append((task_id, score))
        finally:
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))

    except ConnectError as exc:
        print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        print(f"\n{CLI_RED}Interrupted — partial results saved to {log_path}{CLI_CLR}")

    # Write summary (even if interrupted — scores has whatever finished)
    if scores:
        log_file.write_summary(scores)
        scores.sort(key=lambda x: x[0])
        failed = [tid for tid, s in scores if s < 1]
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"{task_id}: {style}{score:.2f}{CLI_CLR}")
        if failed:
            print(f"Failed: {', '.join(failed)}")
        total = sum(s for _, s in scores) / len(scores) * 100.0
        print(f"FINAL: {total:.2f}%")

    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
