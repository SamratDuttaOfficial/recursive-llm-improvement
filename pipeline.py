#!/usr/bin/env python3
"""The whole loop in one command.

    python pipeline.py                 one full cycle: build a corpus, fine-tune, benchmark
    python pipeline.py --rounds 85     the same, writing 85 rounds of exercises before it trains
    python pipeline.py --cycles 3      three cycles, each answering with the model the last one produced
    python pipeline.py --from-stage benchmark   pick up in the middle

Each stage is one of the three scripts, run as a child process so its own resume logic and its own dashboard
apply. If a stage is interrupted, the pipeline stops there, and running the same command again carries on from
that stage of that cycle: the corpus stage writes only the rounds still missing, with the model the cycle began
with, training continues from its last checkpoint and the benchmark from the problems it has not answered yet.
Only once every cycle asked for has finished does the same command start a new one; `--new-cycle` starts one
straight away instead.
"""
import argparse, os, re, subprocess, sys, time

import config as conf
from common import Log, Registry, STATE, Stop, log, read_json, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["corpus", "finetune", "benchmark"]


def run_stage(name, argv):
    if Stop.is_set():                   # a Ctrl-C between two stages must not start the next one
        raise KeyboardInterrupt()
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


def newest_round(run):
    """The number of the newest exercise set on disk, counted the way run.py counts them; 0 before the first."""
    d = os.path.join(STATE, run, "rounds")
    names = os.listdir(d) if os.path.isdir(d) else []
    return max([int(m.group(1)) for m in (re.match(r"round_(\d+)$", n) for n in names)
                if m and os.path.exists(os.path.join(d, m.group(0), "questions.json"))] or [0])


def version_ids():
    return [v["id"] for v in Registry.load().get("versions", [])]


def open_cycle(st):
    """The cycle a stopped run left unfinished, which the same command goes back to."""
    last = st["cycles"][-1] if st["cycles"] else None
    if st.get("job") and last and last.get("job") == st["job"]["n"] and not last.get("finished"):
        return last
    return None


def corpus(a, cyc, save, flags, rest):
    """Write this cycle's rounds. The target is a round number, fixed when the cycle first gets here - the
    newest round then plus --rounds - so a run stopped halfway and started again writes only what is missing,
    rather than --rounds more."""
    if "from_round" not in cyc:
        cyc["from_round"] = newest_round(a.run)
        save()
    extra = ["--rounds", "0"]           # the target below decides when to stop, not a count per run
    if a.rounds:
        target = cyc["from_round"] + a.rounds
        extra += ["--until-round", str(target)]
        have = max(0, newest_round(a.run) - cyc["from_round"])
        log("corpus", "rounds " + str(cyc["from_round"] + 1) + " to " + str(target) + " for this cycle" +
            (" - " + str(have) + " already written, carrying on" if have else ""))
    run_stage("corpus", ["run.py", "--run", a.run, "--generator", cyc["generator"], "--new"] +
              (["--questions", str(a.questions)] if a.questions else []) + extra + flags + rest)
    if a.rounds and newest_round(a.run) < target:
        log("warn", "run.py stopped by itself at round " + str(newest_round(a.run)) + " of " + str(target) +
            " (its log says why); training on what there is", "yellow")


def finetune(a, cyc, save, flags):
    """Train this cycle's version. Registering it is the last thing finetune.py does, so a version that was
    not there when this cycle's training began is that training's result - also when the pipeline was stopped
    before it could note that the stage had finished, where training again would only be refused as a
    duplicate of it."""
    if "versions_before" not in cyc:
        cyc["versions_before"] = version_ids()
        save()
    before = cyc["versions_before"]
    made = [v for v in version_ids() if v not in before]
    if made:
        log("stage", "finetune had already finished: it produced " + made[-1])
    else:
        ft = ["finetune.py", "--run", a.run]
        if cyc.get("stack") and before:
            ft += ["--from-model", before[-1]]
        run_stage("finetune", ft + flags)
        made = [v for v in version_ids() if v not in before]
    cyc["produced"] = made[-1] if made else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="default")
    ap.add_argument("--config", default="", help="config file to pass to every stage (default: config.json)")
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=None,
                    help="rounds of exercises each cycle writes before it trains (default: corpus.rounds in "
                         "config.json, where 0 means until Ctrl-C)")
    ap.add_argument("--questions", type=int, default=0,
                    help="exercises per corpus round (0 = whatever config.json says)")
    ap.add_argument("--from-stage", default=None, choices=STAGES,
                    help="start at this stage (default: wherever the last run stopped)")
    ap.add_argument("--new-cycle", action="store_true",
                    help="start a new cycle even though the last one did not finish")
    ap.add_argument("--generator", default="", help="who answers in a new cycle (default: the newest model)")
    ap.add_argument("--stack", action="store_true",
                    help="each cycle keeps training the previous version's weights on the new round only, "
                         "instead of retraining from the base model on the whole corpus")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    a, rest = ap.parse_known_args()
    Log.setup(True, False)
    Stop.install()
    if a.rounds is None:
        a.rounds = int(conf.load(a.config or None, quiet=True).get("corpus.rounds") or 0)

    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    flags = ((["--no-browser"] if a.no_browser else []) + (["--no-dash"] if a.no_dash else []) +
             (["--config", a.config] if a.config else []))
    state_path = os.path.join(STATE, a.run, "pipeline.json")
    st = read_json(state_path, {}) or {}
    st.setdefault("cycles", [])
    save = lambda: write_json(state_path, st)
    if a.new_cycle and open_cycle(st):
        log("plan", "--new-cycle: cycle " + str(open_cycle(st)["n"]) + " is left unfinished", "yellow")
        st.pop("job", None)
    if not st.get("job"):               # one run of this command, however many times it is started
        st["jobs"] = st.get("jobs", 0) + 1
        st["job"] = {"n": st["jobs"], "done": 0}
    job = st["job"]

    try:
        first = True
        while True:
            cyc = open_cycle(st)
            resumed = cyc is not None
            if not resumed:
                if job["done"] >= a.cycles:
                    break
                # Who answers is fixed when the cycle starts - "latest" becomes the version it names then - so a
                # resumed cycle goes on with the same model even if another version was registered meanwhile.
                gen = (a.generator if first and a.generator not in ("", "latest") else
                       (Registry.latest() or {}).get("id") or "base")
                cyc = {"n": len(st["cycles"]) + 1, "job": job["n"], "started": time.time(), "generator": gen,
                       "stack": a.stack, "done": []}
                st["cycles"].append(cyc)
            elif (a.generator not in ("", "latest", cyc["generator"])) or a.stack != bool(cyc.get("stack")):
                log("warn", "cycle " + str(cyc["n"]) + " was started answering with " + cyc["generator"] +
                    (", stacking" if cyc.get("stack") else "") + ", and it finishes that way "
                    "(--new-cycle starts a new one instead)", "yellow")
            if first and a.from_stage:
                cyc["done"] = STAGES[:STAGES.index(a.from_stage)]
            first = False
            save()
            todo = [s for s in STAGES if s not in cyc["done"] and s not in skip]
            log("cycle", ("resuming" if resumed else "starting") + " cycle " + str(cyc["n"]) + " (" +
                str(job["done"] + 1) + " of " + str(max(a.cycles, job["done"] + 1)) + ")" +
                (" at " + todo[0] if resumed and todo else "") + "  (answers written by " + cyc["generator"] +
                ", judges always on the base model, " +
                ("stacking on the previous version" if cyc.get("stack") else
                 "each version retrained from base on the whole corpus") + ")", "bold")

            for stage in STAGES:
                if stage in cyc["done"]:
                    continue
                if stage not in skip:
                    if stage == "corpus":
                        corpus(a, cyc, save, flags, rest)
                    elif stage == "finetune":
                        finetune(a, cyc, save, flags)
                    else:
                        run_stage("benchmark", ["benchmark.py", "--run", a.run] + flags)
                cyc["done"].append(stage)
                save()
            cyc["finished"] = time.time()
            cyc["produced"] = cyc.get("produced") or (Registry.latest() or {}).get("id")
            job["done"] += 1
            save()
        st.pop("job", None)
        save()
        log("done", "pipeline finished. Results: state/" + a.run + "/bench/comparison.json", "green")
    except KeyboardInterrupt:
        save()
        log("stop", "pipeline stopped. Every stage saved its own progress - run the same command to continue.",
            "yellow")


if __name__ == "__main__":
    main()
