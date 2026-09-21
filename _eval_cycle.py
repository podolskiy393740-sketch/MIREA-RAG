"""
Один цикл: деплой изменённых файлов → перезапуск бота → eval на VPS → результаты.

Использование:
    python _eval_cycle.py                      # деплой + eval
    python _eval_cycle.py --eval-only          # только eval (без деплоя)
    python _eval_cycle.py --deploy-only        # только деплой
    python _eval_cycle.py --csv data/eval/curated_15.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import paramiko

HOST = "185.23.35.5"
USER = "root"
PASS = "Raspberry123!"
REMOTE_ROOT = "/opt/MIREA-RAG-ACCELERATOR"
REMOTE_EVAL_CSV = f"{REMOTE_ROOT}/data/eval/curated_15.csv"
REMOTE_RESULTS = "/tmp/eval_results.json"

DEPLOY_FILES = [
    "src/infrastructure/retrieval/rrf.py",
    "src/infrastructure/llm/prompt_templates.py",
    "src/application/use_cases.py",
    "src/infrastructure/eval/llm_judge.py",
    "src/infrastructure/eval/pipeline.py",
    "src/infrastructure/eval/report.py",
    "scripts/run_eval.py",
]

RESEED_DELTA_CSVS = [
    "data/external/qa_delta_v1.csv",
    "data/external/qa_delta_v2.csv",
]


def connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=20,
               look_for_keys=False, allow_agent=False)
    return c


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 30) -> str:
    _, out, _ = client.exec_command(cmd, timeout=timeout)
    out.channel.recv_exit_status()
    return out.read().decode("utf-8", errors="replace").strip()


def deploy(client: paramiko.SSHClient, csv_path: str) -> None:
    sftp = client.open_sftp()

    # Создаём папку eval на сервере если нет
    run(client, f"mkdir -p {REMOTE_ROOT}/data/eval")

    for rel in DEPLOY_FILES:
        local = rel.replace("/", "\\")
        remote = f"{REMOTE_ROOT}/{rel}"
        sftp.put(local, remote)
        print(f"  up: {rel}")

    # Загружаем eval CSV
    sftp.put(csv_path.replace("/", "\\"), REMOTE_EVAL_CSV)
    print(f"  up: {csv_path}")
    sftp.close()

    # Перезапуск бота
    print("Restarting bot...")
    result = run(client, "systemctl restart mirea-rag-bot 2>/dev/null; echo $?", timeout=15)
    if result.strip() != "0":
        # Fallback: убить и запустить напрямую
        run(client, f"pkill -f 'telegram_bot.bot' 2>/dev/null; sleep 2")
        run(client, (
            f"cd {REMOTE_ROOT} && "
            "nohup .venv/bin/python -m src.presentation.telegram_bot.bot "
            "> /var/log/mirea-rag-bot.log 2>&1 &"
        ))
    time.sleep(4)

    pid = run(client, "pgrep -f 'telegram_bot.bot' | head -1")
    print(f"Bot PID: {pid or 'not found!'}")


def reseed_delta(client: paramiko.SSHClient) -> None:
    sftp = client.open_sftp()
    run(client, f"mkdir -p {REMOTE_ROOT}/data/external")
    for local_csv in RESEED_DELTA_CSVS:
        remote_csv = f"{REMOTE_ROOT}/{local_csv}"
        sftp.put(local_csv.replace("/", "\\"), remote_csv)
        print(f"  up: {local_csv}")
    sftp.close()

    for i, local_csv in enumerate(RESEED_DELTA_CSVS):
        remote_csv = f"{REMOTE_ROOT}/{local_csv}"
        log_file = f"/tmp/reseed_delta_{i}.log"
        print(f"Running seeder for {local_csv}...")
        seed_cmd = (
            f"cd {REMOTE_ROOT} && "
            f".venv/bin/python -m scripts.seed_from_qa_csv "
            f"--csv {remote_csv} "
            f"> {log_file} 2>&1"
        )
        transport = client.get_transport()
        chan = transport.open_session()
        chan.exec_command(f"nohup bash -c '{seed_cmd}' &")
        time.sleep(2)
        chan.close()

        print("Waiting", end="", flush=True)
        for _ in range(60):
            time.sleep(30)
            alive = run(client, "pgrep -f 'seed_from_qa_csv' | head -1", timeout=10)
            if not alive:
                log = run(client, f"tail -5 {log_file}", timeout=10)
                print(f"\nSeeder log:\n{log}")
                break
            print(".", end="", flush=True)
        print()


def run_eval(client: paramiko.SSHClient, csv_path: str, limit: int | None) -> None:
    limit_arg = f"--limit {limit}" if limit else ""

    # Убираем старые результаты чтобы polling не поймал прошлый прогон
    run(client, f"rm -f {REMOTE_RESULTS} /tmp/eval_run.log")

    cmd = (
        f"cd {REMOTE_ROOT} && "
        f".venv/bin/python -m scripts.run_eval "
        f"--csv {REMOTE_EVAL_CSV} {limit_arg} "
        f"--output {REMOTE_RESULTS} "
        f"> /tmp/eval_run.log 2>&1"
    )

    # Fire-and-forget: open_session без ожидания завершения
    transport = client.get_transport()
    chan = transport.open_session()
    chan.exec_command(f"nohup bash -c '{cmd}' &")
    time.sleep(2)
    chan.close()

    print("Waiting for eval to finish", end="", flush=True)
    for _ in range(120):  # max 60 минут
        time.sleep(30)
        ok = run(client, f"test -f {REMOTE_RESULTS} && echo yes || echo no", timeout=10)
        if ok == "yes":
            break
        # Проверяем что процесс ещё жив
        alive = run(client, "pgrep -f 'run_eval' | head -1", timeout=10)
        if not alive:
            log = run(client, "tail -30 /tmp/eval_run.log", timeout=15)
            print(f"\nEval process died. Log:\n{log}")
            sys.exit(1)
        print(".", end="", flush=True)
    print()

    ok = run(client, f"test -f {REMOTE_RESULTS} && echo yes || echo no")
    if ok != "yes":
        log = run(client, "tail -30 /tmp/eval_run.log")
        print(f"Eval did not produce results. Log:\n{log}")
        sys.exit(1)


def download_results(client: paramiko.SSHClient, local_path: str) -> dict:
    sftp = client.open_sftp()
    sftp.get(REMOTE_RESULTS, local_path)
    sftp.close()
    with open(local_path, encoding="utf-8") as f:
        return json.load(f)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    fallbacks = sum(1 for r in results if r.get("needs_human_fallback"))
    errors = sum(1 for r in results if r.get("error"))

    judge_scores = [r["judge_score"] for r in results if r.get("judge_score") is not None]
    faithfulness = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    answer_rel = [r["answer_relevance"] for r in results if r.get("answer_relevance") is not None]
    ctx_recall = [r["context_recall"] for r in results if r.get("context_recall") is not None]

    def mean(xs): return sum(xs) / len(xs) if xs else None
    def fmt(v): return f"{v:.2f}" if v is not None else "n/a"

    print(f"\n{'='*55}")
    print(f"EVAL SUMMARY  ({total} cases)")
    print("=" * 55)
    print(f"  Fallback rate:    {fallbacks}/{total} ({fallbacks/total*100:.0f}%)")
    print(f"  Errors:           {errors}")
    ge4 = sum(1 for s in judge_scores if s >= 4)
    print(f"  Judge>=4:         {ge4}/{len(judge_scores)} ({ge4/len(judge_scores)*100:.0f}%)" if judge_scores else "  Judge:  n/a")
    print(f"  Judge mean:       {fmt(mean(judge_scores))}/5")
    print(f"  Faithfulness:     {fmt(mean(faithfulness))}")
    print(f"  Answer Relevance: {fmt(mean(answer_rel))}")
    print(f"  Context Recall:   {fmt(mean(ctx_recall))}")

    print("\n" + "-" * 55)
    print("FAILURES:")
    for r in results:
        if r.get("needs_human_fallback") or (r.get("judge_score") or 5) <= 2:
            j = r.get("judge_score", "?")
            print(f"  [{j}] fallback={r['needs_human_fallback']}  {r['question'][:65]}")
            if r.get("judge_reason"):
                print(f"       reason: {r['judge_reason'][:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--deploy-only", action="store_true")
    parser.add_argument("--reseed", action="store_true", help="Upload and run delta seeder before eval")
    parser.add_argument("--reseed-only", action="store_true", help="Only run delta seeder, no eval")
    parser.add_argument("--csv", default="data/eval/curated_15.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/eval/last_results.json")
    args = parser.parse_args()

    client = connect()
    try:
        if not args.eval_only and not args.reseed_only:
            print("=== DEPLOY ===")
            deploy(client, args.csv)

        if args.reseed or args.reseed_only:
            print("\n=== RESEED DELTA ===")
            reseed_delta(client)

        if not args.deploy_only and not args.reseed_only:
            print("\n=== EVAL ===")
            run_eval(client, args.csv, args.limit)
            results = download_results(client, args.output)
            print_summary(results)
            print(f"\nFull results: {args.output}")
    finally:
        client.close()
