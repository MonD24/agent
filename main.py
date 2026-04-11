import datetime
import json
import os
import sys
import pathlib
import textwrap
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
WORKERS = int(os.getenv("WORKERS") or "4")

USE_COLOR = sys.stdout.isatty()
CLI_RED = "\x1B[31m" if USE_COLOR else ""
CLI_GREEN = "\x1B[32m" if USE_COLOR else ""
CLI_CLR = "\x1B[0m" if USE_COLOR else ""
CLI_BLUE = "\x1B[34m" if USE_COLOR else ""

_print_lock = threading.Lock()


def _console(worker_num: int, msg: str) -> None:
    with _print_lock:
        print(f"[worker-{worker_num}] {msg}", flush=True)


def run_trial(
    trial_id: str,
    client: HarnessServiceClientSync,
    task_filter: list[str],
    worker_num: int,
    run_log: list[dict],
) -> tuple[str, float, list[str]] | None:
    trial = client.start_trial(StartTrialRequest(trial_id=trial_id))

    if task_filter and trial.task_id not in task_filter:
        return None

    _console(worker_num, f"started task {trial.task_id}")

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
    _console(worker_num, f"finished {trial.task_id} — score: {style}{score:.2f}{CLI_CLR}")

    with _print_lock:
        run_log.append({
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
    run_log: list[dict] = []
    scores = []

    try:
        client = HarnessServiceClientSync(BITGN_URL)

        print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
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
                    pool.submit(run_trial, tid, client, task_filter, i % workers + 1, run_log): tid
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
        print(f"{CLI_RED}Interrupted{CLI_CLR}")

    # --- Write structured log file ---
    if run_log:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = pathlib.Path(f"run_{ts}.log.json")
        run_log.sort(key=lambda x: x["task_id"])
        log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDetailed log -> {log_path}")

    # --- Final scores ---
    if scores:
        scores.sort(key=lambda x: x[0])
        failed = [(tid, s) for tid, s in scores if s < 1]
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            print(f"{task_id}: {style}{score:.2f}{CLI_CLR}")
        if failed:
            print(f"\nFailed: {', '.join(tid for tid, _ in failed)}")
        total = sum(s for _, s in scores) / len(scores) * 100.0
        print(f"FINAL: {total:.2f}%")


if __name__ == "__main__":
    main()
