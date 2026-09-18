#!/usr/bin/env python3
"""The whole loop in one command.

    python pipeline.py                 one full cycle: generate a corpus round, fine-tune, benchmark
    python pipeline.py --cycles 3      three cycles, each answering with the model the last one produced
    python pipeline.py --from-stage benchmark   pick up in the middle

Each stage is one of the three scripts, run as a child process so its own resume logic and its own dashboard
apply. If a stage is interrupted, the pipeline stops there; running the same command again continues from
that stage, because every stage is itself resumable.
"""
import argparse, os, subprocess, sys, time

from common import Log, Registry, STATE, Stop, log, read_json, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["corpus", "finetune", "benchmark"]


def run_stage(name, argv):
    log("stage", "starting " + name + ":  python " + " ".join(argv), "bold")
    p = subprocess.Popen([sys.executable] + [os.path.join(HERE, argv[0])] + argv[1:])
    try:
        rc = p.wait()
    except KeyboardInterrupt:
        Stop.set()
        try:
            p.wait(timeout=90)      # the child installs its own handler and saves before exiting
        except Exception:
            p.kill()
        rc = 130
    if rc == 130 or Stop.is_set():
        raise KeyboardInterrupt()
    if rc != 0:
        sys.exit(name + " failed with exit code " + str(rc))
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="default")
    ap.add_argument("--config", default="", help="config file to pass to every stage (default: config.json)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--questions", type=int, default=0,
                    help="exercises per corpus round (0 = whatever config.json says)")
    ap.add_argument("--from-stage", default="corpus", choices=STAGES)
    ap.add_argument("--generator", default="", help="who answers in the first cycle (default: the newest model)")
    ap.add_argument("--stack", action="store_true",
                    help="each cycle keeps training the previous version's weights on the new round only, "
                         "instead of retraining from the base model on the whole corpus")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    a, rest = ap.parse_known_args()
    Log.setup(True, False)
    Stop.install()

    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    flags = ((["--no-browser"] if a.no_browser else []) + (["--no-dash"] if a.no_dash else []) +
             (["--config", a.config] if a.config else []))
    state_path = os.path.join(STATE, a.run, "pipeline.json")
    st = read_json(state_path, {"cycles": []}) or {"cycles": []}

    try:
        for c in range(a.cycles):
            gen = a.generator or ("latest" if Registry.load().get("versions") else "base")
            cycle = {"n": len(st["cycles"]) + 1, "started": time.time(), "generator": gen}
            st["cycles"].append(cycle)
            write_json(state_path, st)
            log("cycle", "cycle " + str(cycle["n"]) + " of " + str(a.cycles) + "  (answers written by " +
                gen + ", judges always on the base model, " +
                ("stacking on the previous version" if a.stack else
                 "each version retrained from base on the whole corpus") + ")", "bold")

            start = STAGES.index(a.from_stage) if c == 0 else 0
            if start <= 0 and "corpus" not in skip:
                run_stage("corpus", ["run.py", "--run", a.run, "--generator", gen, "--new"] +
                          (["--questions", str(a.questions)] if a.questions else []) + flags + rest)
            if start <= 1 and "finetune" not in skip:
                ft = ["finetune.py", "--run", a.run]
                if a.stack and Registry.load().get("versions"):
                    ft += ["--from-model", "latest"]
                run_stage("finetune", ft + flags)
            if start <= 2 and "benchmark" not in skip:
                run_stage("benchmark", ["benchmark.py", "--run", a.run] + flags)
            cycle["finished"] = time.time()
            cycle["produced"] = (Registry.latest() or {}).get("id")
            write_json(state_path, st)
            a.generator = ""     # later cycles always answer with the newest model
        log("done", "pipeline finished. Results: state/" + a.run + "/bench/comparison.json", "green")
    except KeyboardInterrupt:
        write_json(state_path, st)
        log("stop", "pipeline stopped. Every stage saved its own progress - run the same command to continue.",
            "yellow")


if __name__ == "__main__":
    main()
