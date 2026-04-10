import os
import sys
import pathlib
import textwrap
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

_print_lock = __import__("threading").Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def run_trial(trial_id: str, client: HarnessServiceClientSync) -> tuple[str, float, list[str]] | None:
    """Run a single trial and return (task_id, score, score_detail)."""
    trial = client.start_trial(StartTrialRequest(trial_id=trial_id))
    _safe_print(f"{'=' * 30} Starting task: {trial.task_id} {'=' * 30}")
    _safe_print(f"{CLI_BLUE}{trial.instruction}{CLI_CLR}\n{'-' * 80}")

    try:
        run_agent(MODEL_ID, trial.harness_url, trial.instruction)
    except Exception as exc:
        _safe_print(f"{CLI_RED}Agent error on {trial.task_id}: {exc}{CLI_CLR}")

    result = client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
    if result.score >= 0:
        return trial.task_id, result.score, list(result.score_detail)
    return None


def main() -> None:
    task_filter = sys.argv[1:]

    scores = []
    try:
        client = HarnessServiceClientSync(BITGN_URL)

        _safe_print("Connecting to BitGN", client.status(StatusRequest()))
        res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=BENCH_ID))
        _safe_print(
            f"{EvalPolicy.Name(res.policy)} benchmark: {res.benchmark_id} "
            f"with {len(res.tasks)} tasks.\n{CLI_GREEN}{res.description}{CLI_CLR}"
        )

        run = client.start_run(StartRunRequest(
            name="SGR NextStep Sample",
            benchmark_id=BENCH_ID,
            api_key=BITGN_API_KEY,
        ))

        trial_ids = [
            tid for tid in run.trial_ids
            if not task_filter or any(f in tid for f in task_filter)
        ]

        try:
            workers = min(WORKERS, len(trial_ids)) if trial_ids else 1
            _safe_print(f"Running {len(trial_ids)} trials with {workers} parallel workers...")

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_trial, tid, client): tid for tid in trial_ids}
                for fut in as_completed(futures):
                    result = fut.result()
                    if result is None:
                        continue
                    task_id, score, score_detail = result
                    scores.append((task_id, score))
                    style = CLI_GREEN if score == 1 else CLI_RED
                    explain = textwrap.indent("\n".join(score_detail), "  ")
                    _safe_print(f"\n{style}Score: {score:0.2f}\n{explain}\n{CLI_CLR}")

        finally:
            client.submit_run(SubmitRunRequest(run_id=run.run_id, force=True))

    except ConnectError as exc:
        _safe_print(f"{exc.code}: {exc.message}")
    except KeyboardInterrupt:
        _safe_print(f"{CLI_RED}Interrupted{CLI_CLR}")

    if scores:
        scores.sort(key=lambda x: x[0])  # sort by task_id for readable output
        for task_id, score in scores:
            style = CLI_GREEN if score == 1 else CLI_RED
            _safe_print(f"{task_id}: {style}{score:0.2f}{CLI_CLR}")

        total = sum(score for _, score in scores) / len(scores) * 100.0
        _safe_print(f"FINAL: {total:0.2f}%")


if __name__ == "__main__":
    main()
