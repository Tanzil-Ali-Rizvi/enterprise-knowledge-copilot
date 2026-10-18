"""Command line: `kcopilot ingest`, `kcopilot ask "question" --user priya`."""

from __future__ import annotations

import argparse
import json
import logging

from .pipeline import Copilot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kcopilot", description="Enterprise Knowledge Copilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(Re)build the index from KC_DATA_DIR")
    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--user", default="priya", help="user_id from data/users.json")
    ask.add_argument("--mode", choices=["hybrid", "vector", "bm25"])
    ask.add_argument("--json", action="store_true", help="print the full JSON response")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    cp = Copilot()
    if args.cmd == "ingest":
        print(json.dumps(cp.ingest(cp.settings.data_dir), indent=2))
        return

    ans = cp.ask(args.question, args.user, mode=args.mode)
    if args.json:
        print(ans.model_dump_json(indent=2))
        return
    print(f"\n{ans.answer}\n")
    for c in ans.citations:
        where = f"{c.title} > {c.section}" if c.section else c.title
        print(f"  [{c.index}] {where}  ({c.source})")
    print(f"\nconfidence={ans.confidence}  total={ans.timings_ms['total_ms']}ms")


if __name__ == "__main__":
    main()
