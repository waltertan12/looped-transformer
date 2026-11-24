#!/usr/bin/env python
"""Record what's running on this machine, so other sessions and tracks don't collide.

The record lives in `.locks/running.json` (git-ignored: it describes this machine, not the
project). Entries whose process has died are dropped whenever the file is read.

    scripts/lock.py show                                   # what's running now (check before launching)
    scripts/lock.py add --name h16-wide --pid 1234 --track arith --gpu \\
        --log results/toy/wide-pause.queue.log --note "D-015, 9 runs, pauses at 40k"
    scripts/lock.py remove --name h16-wide

Queue scripts can call `add` when they start and `remove` when they finish. The house rule is at
most 2 GPU trainings at a time across every track.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / ".locks" / "running.json"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load() -> list[dict]:
    if not LOCK.exists():
        return []
    entries = json.loads(LOCK.read_text())
    live = [e for e in entries if _alive(int(e["pid"]))]
    if len(live) != len(entries):
        save(live)
    return live


def save(entries: list[dict]) -> None:
    LOCK.parent.mkdir(exist_ok=True)
    tmp = LOCK.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=1))
    tmp.replace(LOCK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    a_add = sub.add_parser("add")
    a_add.add_argument("--name", required=True)
    a_add.add_argument("--pid", type=int, required=True, help="the queue's top process; the entry lives while it does")
    a_add.add_argument("--track", default="arith", help="arith, protein, ...")
    a_add.add_argument("--gpu", action="store_true", help="uses the GPU")
    a_add.add_argument("--log", default="")
    a_add.add_argument("--note", default="")
    a_rm = sub.add_parser("remove")
    a_rm.add_argument("--name", required=True)
    a = ap.parse_args()

    entries = load()
    if a.cmd == "show":
        if not entries:
            print("Nothing running.")
        for e in entries:
            print(f"{e['name']:<14} pid {e['pid']:<8} track {e['track']:<8} {'GPU' if e['gpu'] else 'CPU'}  "
                  f"since {e['started']}\n    {e['note']}\n    log: {e['log']}")
    elif a.cmd == "add":
        entries = [e for e in entries if e["name"] != a.name]
        entries.append({"name": a.name, "pid": a.pid, "track": a.track, "gpu": a.gpu, "log": a.log,
                        "note": a.note, "started": time.strftime("%Y-%m-%d %H:%M")})
        save(entries)
        print(f"added {a.name}")
    else:
        save([e for e in entries if e["name"] != a.name])
        print(f"removed {a.name}")


if __name__ == "__main__":
    main()
