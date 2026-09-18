#!/usr/bin/env python3
"""Phase 2 of the loop: fine-tune the model on the answers it produced.

    python finetune.py

One command again. It builds the training set out of the accepted answers, creates a project virtual
environment, installs the right stack for this machine (PyTorch + PEFT on NVIDIA/CPU, MLX on Apple silicon),
downloads the base weights in Hugging Face format, trains a LoRA, merges it, converts the result to GGUF and
registers it as the next model version, which `run.py --generator latest` and `benchmark.py` then pick up.

Seven phases, each recorded the moment it finishes:
    dataset -> deps -> base weights -> train -> merge -> gguf -> register
Re-running skips everything already done; an interrupted training resumes from its last checkpoint.

The recipe - epochs, LoRA rank, learning rate, sequence length, what a corpus answer has to score to be
trained on - lives in the `finetune` block of `config.json`, and a flag overrides it for one run.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time

from common import (DATA, Dash, Interrupted, LOGS, Log, MODELS, Registry, STATE, Stop, Venv, WIN, ARM_MAC,
                    common_state, download, fmt_t, gpu_status, log, phase, read_json, read_jsonl, write_json)
import config as conf
import prompts

CONVERT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"
PHASES = ["dataset", "deps", "base", "train", "merge", "gguf", "register"]
LIVE_TRAIN = {"steps": [], "state": {}, "log": []}

# Which config key backs each flag. A flag the user actually passes wins; anything else comes from the file.
CONFIG_MAP = {
    "from_model": "finetune.from_model", "hf_base": "model.hf_repo",
    "epochs": "finetune.epochs", "batch": "finetune.batch", "accum": "finetune.accum",
    "seq_len": "finetune.seq_len", "lr": "finetune.lr", "rank": "finetune.rank",
    "alpha": "finetune.alpha", "save_steps": "finetune.save_steps",
    "min_score": "finetune.min_score", "min_examples": "finetune.min_examples",
    "dash_port": "ui.ports.finetune",
}


# ---------------------------------------------------------------- the training set
def chatml(question, answer):
    return ("<|im_start|>system\n" + prompts.SFT_SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + question.strip() + "<|im_end|>\n"
            "<|im_start|>assistant\n" + answer.strip() + "<|im_end|>")


def corpus_rounds(run):
    """Which corpus rounds exist on disk."""
    d = os.path.join(STATE, run, "corpus")
    if not os.path.isdir(d):
        return []
    return sorted(int(re.findall(r"\d+", f)[0]) for f in os.listdir(d) if re.match(r"round_\d+\.jsonl$", f))


def build_dataset(a, out_dir):
    """Every accepted answer from the requested rounds, newest rounds last, de-duplicated by exercise."""
    corpus_dir = os.path.join(STATE, a.run, "corpus")
    if not os.path.isdir(corpus_dir):
        sys.exit("no corpus yet in " + corpus_dir + " - run `python run.py` first")
    files = sorted(f for f in os.listdir(corpus_dir) if re.match(r"round_\d+\.jsonl$", f))
    if a.rounds:
        want = {int(x) for x in a.rounds.split(",") if x.strip()}
        files = [f for f in files if int(re.findall(r"\d+", f)[0]) in want]
    rows, seen, per_round = [], {}, {}
    for f in files:
        n = int(re.findall(r"\d+", f)[0])
        per_round.setdefault(n, 0)
        for it in read_jsonl(os.path.join(corpus_dir, f)):
            if it.get("check_score", 0) < a.min_score:
                continue
            key = re.sub(r"\W+", "", (it.get("title") or it.get("qid", "")).lower())
            seen[key] = it          # a later round's answer to the same exercise replaces the earlier one
    for it in seen.values():
        per_round[it["round"]] = per_round.get(it["round"], 0) + 1
        text = chatml(it["question"], it["answer"])
        rows.append({"text": text, "qid": it["qid"], "round": it["round"],
                     "score": it.get("check_score"), "chars": len(text)})
    rows.sort(key=lambda r: (r["round"], r["qid"]))
    if len(rows) < a.min_examples:
        sys.exit("only " + str(len(rows)) + " accepted answers (need at least " + str(a.min_examples) +
                 "). Generate more with:  python run.py --new")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fingerprint = hashlib.sha1(
        "|".join(r["qid"] + ":" + str(r["score"]) for r in rows).encode()).hexdigest()[:16]
    stats = {"examples": len(rows), "rounds": per_round, "chars": sum(r["chars"] for r in rows),
             "avg_chars": round(sum(r["chars"] for r in rows) / len(rows)), "fingerprint": fingerprint}
    log("dataset", str(len(rows)) + " training examples from rounds " +
        ", ".join(str(k) for k in sorted(per_round)) + " (avg " + str(stats["avg_chars"]) + " chars)")
    return path, stats


def plan_parent(a):
    """Decide what this version is trained *from*, and on which rounds.

    Two ways to make the next version, and they are not the same experiment:

      --from-model base    (default) retrain from the untouched base model on the whole corpus so far.
                           Each version is one clean training run, so v3 is not v1's mistakes compounded
                           three times, and every version stays comparable to the base.
      --from-model latest  stack: keep training the newest version's own weights. Cheaper each time and it
                           accumulates, but it also accumulates drift, and a stacked version has seen its
                           parent's data already - so by default only the rounds the parent never trained
                           on are used, which is what makes stacking meaningful rather than three epochs
                           of the same data.
    """
    if a.from_model == "base":
        return None, a.rounds
    parent = Registry.resolve(a.from_model)
    if not parent or parent.get("id") in ("base", "base_f16"):
        log("warn", "no fine-tuned version to stack on yet; training from the base model", "yellow")
        return None, a.rounds
    if not (parent.get("merged") and os.path.exists(os.path.join(parent["merged"], "config.json"))):
        sys.exit("cannot stack on " + parent["id"] + ": its merged weights are gone (" +
                 str(parent.get("merged")) + "). Train from the base model instead, or re-run "
                 "`python finetune.py --version " + parent["id"] + " --restart`.")
    rounds = a.rounds
    if not rounds:
        seen = {int(x) for x in (parent.get("rounds") or [])}
        fresh = [r for r in corpus_rounds(a.run) if r not in seen]
        if not fresh:
            sys.exit("stacking on " + parent["id"] + " would re-train it on exactly the rounds it has "
                     "already learned (" + ", ".join(str(x) for x in sorted(seen)) + "). Generate a new "
                     "round first:  python run.py --generator latest --new\n"
                     "Or pass --rounds explicitly if repeating them is what you want.")
        rounds = ",".join(str(r) for r in fresh)
        log("plan", "stacking on " + parent["id"] + ": training on the rounds it has not seen (" +
            rounds + ")")
    return parent, rounds


def check_duplicate(a, vid, parent, stats):
    """Refuse to spend hours producing a version identical to one that already exists: same training data,
    same parent, same recipe."""
    if a.force:
        return
    pid = parent["id"] if parent else "base"
    for v in Registry.load().get("versions", []):
        if v["id"] == vid:
            continue
        if v.get("fingerprint") == stats.get("fingerprint") and v.get("parent", v.get("from")) == pid \
                and (v.get("recipe") or {}).get("epochs") == a.epochs \
                and (v.get("recipe") or {}).get("rank") == a.rank:
            sys.exit(v["id"] + " was already trained from " + pid + " on exactly this data (" +
                     str(stats["examples"]) + " examples, fingerprint " + str(stats["fingerprint"]) +
                     ") with the same recipe, so " + vid + " would be a copy of it.\n"
                     "Generate more answers first:   python run.py --generator latest --new\n"
                     "Or change the recipe (--epochs/--rank/--lr), or pass --force to train it anyway.")


# ---------------------------------------------------------------- dependencies for this machine
def torch_index():
    """The right PyTorch wheel index: CUDA where there is an NVIDIA GPU, plain CPU otherwise."""
    g = gpu_status()
    if g and g["kind"] == "cuda":
        return "https://download.pytorch.org/whl/cu124"
    if g and g["kind"] == "rocm" and not WIN:
        return "https://download.pytorch.org/whl/rocm6.2"
    if WIN or (g is None):
        return "https://download.pytorch.org/whl/cpu"
    return None


def install_deps(a):
    Venv.ensure()
    if ARM_MAC and not a.force_torch:
        Venv.need(["mlx", "mlx_lm"], ["mlx", "mlx-lm"], note="MLX (Apple silicon GPU training)")
        Venv.need(["transformers"], ["transformers>=4.51", "sentencepiece", "protobuf"],
                  note="transformers (tokenizer + GGUF conversion)")
        Venv.need(["huggingface_hub"], ["huggingface_hub"], note="huggingface_hub (weights download)")
        return "mlx"
    Venv.need(["torch"], ["torch"], index=torch_index(), note="PyTorch (" + str(torch_index()).split("/")[-1] + ")")
    Venv.need(["transformers", "peft", "datasets", "accelerate"],
              ["transformers>=4.51", "peft>=0.13", "datasets", "accelerate", "sentencepiece", "protobuf"],
              note="transformers + PEFT + datasets")
    Venv.need(["huggingface_hub"], ["huggingface_hub"], note="huggingface_hub (weights download)")
    if a.load_4bit:
        Venv.need(["bitsandbytes"], ["bitsandbytes"], note="bitsandbytes (4-bit base weights)")
    return "torch"


# ---------------------------------------------------------------- base weights in HF format
def hf_repo_exists(repo):
    try:
        from common import http
        http("https://huggingface.co/api/models/" + repo, timeout=20)
        return True
    except Exception:
        return False


def fetch_base_hf(a):
    dest = os.path.join(MODELS, "base_hf")
    if os.path.exists(os.path.join(dest, "config.json")):
        return dest
    repo = a.hf_base
    if not repo:
        for cand in a.hf_candidates:
            if hf_repo_exists(cand):
                repo = cand
                break
    if not repo:
        sys.exit("could not find the base model on Hugging Face. Set model.hf_repo in " + a.config +
                 " (or add the repo to model.hf_candidates), or pass --hf-base <repo>.")
    log("weights", "downloading " + repo + " in Hugging Face format (needed for training)")
    code = ("import sys;from huggingface_hub import snapshot_download;"
            "p=snapshot_download(sys.argv[1], local_dir=sys.argv[2], "
            "allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py'],"
            "max_workers=4);print(p)")
    r = Venv.run(["-c", code, repo, dest])
    if r.returncode != 0 or not os.path.exists(os.path.join(dest, "config.json")):
        sys.exit("downloading " + repo + " failed")
    write_json(os.path.join(dest, "_source.json"), {"repo": repo, "when": time.time()})
    log("weights", "base weights ready in " + dest)
    return dest


# ---------------------------------------------------------------- training
def run_training(a, base_hf, data, out_dir, prog, save):
    """Start train_worker.py in the venv and follow its PROGRESS lines. Ctrl-C stops it and keeps the last
    checkpoint, so the next run continues from there. An out-of-memory failure retries with smaller settings."""
    ladder = [{"seq_len": a.seq_len, "batch": a.batch, "load_4bit": a.load_4bit},
              {"seq_len": max(512, a.seq_len // 2), "batch": 1, "load_4bit": a.load_4bit},
              {"seq_len": max(512, a.seq_len // 2), "batch": 1, "load_4bit": True}]
    start = prog.get("train_attempt", 0)
    for attempt in range(start, len(ladder)):
        if Stop.is_set():
            raise Interrupted()
        cfg = ladder[attempt]
        prog["train_attempt"] = attempt
        prog["train_cfg"] = cfg
        save()
        args = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_worker.py"),
                "--base", base_hf, "--data", data, "--out", out_dir,
                "--epochs", str(a.epochs), "--batch", str(cfg["batch"]), "--accum", str(a.accum),
                "--seq-len", str(cfg["seq_len"]), "--lr", str(a.lr), "--rank", str(a.rank),
                "--alpha", str(a.alpha), "--save-steps", str(a.save_steps), "--merge"]
        if cfg["load_4bit"]:
            args.append("--load-4bit")
        log("train", "starting" + (" (attempt " + str(attempt + 1) + ")" if attempt else "") +
            ": seq " + str(cfg["seq_len"]) + ", batch " + str(cfg["batch"]) + " x accum " + str(a.accum) +
            ", lr " + str(a.lr) + ", rank " + str(a.rank) + (", 4-bit base" if cfg["load_4bit"] else ""), "bold")
        logf = open(os.path.join(LOGS, "train.log"), "a", encoding="utf-8")
        logf.write("\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " attempt " + str(attempt + 1) + " =====\n")
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
        p = subprocess.Popen([Venv.python()] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, errors="replace", **kw)
        from common import kill_tree
        Stop.on_exit(lambda: kill_tree(p))
        oom, t_last, t0 = False, time.time(), time.time()
        for line in p.stdout:
            logf.write(line)
            line = line.rstrip()
            if Stop.is_set():
                kill_tree(p)
                break
            if line.startswith("PROGRESS "):
                ev = json.loads(line[9:])
                handle_progress(ev, prog, save, t0)
            else:
                LIVE_TRAIN["log"].append(line[:300])
                del LIVE_TRAIN["log"][:-200]
                if re.search(r"out of memory|CUDA out of memory|MPS backend out of memory", line, re.I):
                    oom = True
                if time.time() - t_last > 30:
                    t_last = time.time()
                    log("train", line[:150])
        rc = p.wait()
        logf.close()
        if Stop.is_set():
            raise Interrupted()
        if rc == 0:
            prog["train_done"] = True
            save()
            return True
        if oom and attempt + 1 < len(ladder):
            log("warn", "out of memory - retrying with smaller settings", "yellow")
            continue
        sys.exit("training failed (exit " + str(rc) + "); the full output is in logs/train.log")
    return False


def handle_progress(ev, prog, save, t0):
    kind = ev.get("kind")
    if kind == "step":
        LIVE_TRAIN["steps"].append({"step": ev.get("step"), "loss": ev.get("loss"),
                                    "lr": ev.get("lr"), "t": ev.get("t")})
        del LIVE_TRAIN["steps"][:-4000]
        LIVE_TRAIN["state"].update(step=ev.get("step"), max_steps=ev.get("max_steps"),
                                   loss=ev.get("loss"), epoch=ev.get("epoch"))
        prog["step"], prog["max_steps"] = ev.get("step"), ev.get("max_steps")
        st, mx = ev.get("step") or 0, ev.get("max_steps") or 0
        if st and st % 10 == 0:
            el = time.time() - t0
            eta = el / st * (mx - st) if mx and st else 0
            log("train", "step " + str(st) + "/" + str(mx) + "  loss " +
                (format(ev["loss"], ".4f") if ev.get("loss") is not None else "-") +
                "  epoch " + str(ev.get("epoch", "?")) + "  eta " + fmt_t(eta))
            save()
    elif kind == "checkpoint":
        prog["last_checkpoint"] = ev.get("step")
        save()
        log("train", "checkpoint at step " + str(ev.get("step")) + " (a Ctrl-C here loses nothing)")
    elif kind == "setup":
        LIVE_TRAIN["state"].update(device=ev.get("device"), gpu=ev.get("gpu"))
        log("train", "device " + str(ev.get("device")) + " - " + str(ev.get("gpu", "")))
    elif kind == "model":
        LIVE_TRAIN["state"].update(trainable=ev.get("trainable"), total=ev.get("total"))
        log("train", "LoRA on " + ", ".join(ev.get("targets", [])) + ": " +
            format((ev.get("trainable") or 0) / 1e6, ".1f") + "M trainable of " +
            format((ev.get("total") or 0) / 1e6, ".0f") + "M")
    elif kind == "data":
        LIVE_TRAIN["state"].update(examples=ev.get("examples"))
    elif kind == "resume":
        log("train", "resuming from " + str(ev.get("checkpoint")), "green")
    elif kind in ("merging", "merged"):
        log("train", kind + " " + str(ev.get("to") or ev.get("path")))
    elif kind == "error":
        log("warn", "trainer error: " + str(ev.get("error"))[:200], "yellow")


# ---------------------------------------------------------------- GGUF conversion
def convert_gguf(a, hf_dir, out_gguf, label):
    if os.path.exists(out_gguf):
        return out_gguf
    script = os.path.join(DATA, "convert_hf_to_gguf.py")
    if not os.path.exists(script):
        download(CONVERT_URL, script)
    Venv.need(["gguf"], ["gguf"], note="gguf (GGUF writer)")
    Venv.need(["torch"], ["torch"], index=torch_index(), note="PyTorch")
    log("gguf", "converting " + label + " to GGUF f16 (this is what llama-server loads)")
    os.makedirs(os.path.dirname(out_gguf), exist_ok=True)
    r = Venv.run([script, hf_dir, "--outfile", out_gguf, "--outtype", "f16"])
    if r.returncode != 0 or not os.path.exists(out_gguf):
        log("warn", "GGUF conversion failed for " + label + "; the merged Hugging Face model is still in " +
            hf_dir, "yellow")
        return None
    log("gguf", label + " -> " + out_gguf + " (" +
        format(os.path.getsize(out_gguf) / 1e9, ".2f") + " GB)", "green")
    return out_gguf


# ---------------------------------------------------------------- dashboard
def make_api(a, prog_holder):
    def api(what, q):
        if what == "state":
            return common_state(None, {
                "kind": "finetune", "run": a.run, "progress": prog_holder[0],
                "train": {"state": LIVE_TRAIN["state"], "steps": LIVE_TRAIN["steps"][-1500:],
                          "log": LIVE_TRAIN["log"][-60:]},
                "registry": Registry.load(),
                "config": {k: v for k, v in vars(a).items() if not k.startswith("_")}})
        return None
    return api


# ---------------------------------------------------------------- main
def main():
    O = conf.opt()      # a flag with no default: when it is not passed, config.json decides
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    conf.add_args(ap)
    ap.add_argument("--run", default="default")
    ap.add_argument("--rounds", default="", help="which corpus rounds to train on (default: all)")
    ap.add_argument("--from-model", default=O,
                    help="what to fine-tune: 'base' (default: a clean run from the untouched model on the "
                         "whole corpus) or 'latest'/'v1' (stack: keep training that version's own weights "
                         "on the rounds it has not seen)")
    ap.add_argument("--hf-base", default=O, help="Hugging Face repo of the base weights (auto-detected)")
    ap.add_argument("--version", default="", help="version id to produce (default: the next free vN)")
    ap.add_argument("--epochs", type=float, default=O)
    ap.add_argument("--batch", type=int, default=O)
    ap.add_argument("--accum", type=int, default=O)
    ap.add_argument("--seq-len", type=int, default=O)
    ap.add_argument("--lr", type=float, default=O)
    ap.add_argument("--rank", type=int, default=O)
    ap.add_argument("--alpha", type=int, default=O)
    ap.add_argument("--save-steps", type=int, default=O)
    ap.add_argument("--min-score", type=int, default=O, help="checker score an answer needs to be trained on")
    ap.add_argument("--min-examples", type=int, default=O)
    ap.add_argument("--load-4bit", action="store_true", help="load the base in 4-bit (tight GPUs)")
    ap.add_argument("--force-torch", action="store_true", help="use PyTorch even on Apple silicon")
    ap.add_argument("--no-gguf", action="store_true", help="stop after merging, do not convert")
    ap.add_argument("--restart", action="store_true", help="throw away this version's progress and start over")
    ap.add_argument("--force", action="store_true",
                    help="train even when an existing version already used exactly this data and recipe")
    ap.add_argument("--dash-port", type=int, default=O)
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    Log.setup(not a.no_color, a.verbose)
    if conf.handle_meta(a):
        return
    cfg = conf.load(a.config or None)
    conf.apply(a, cfg, CONFIG_MAP)
    a.config = cfg.path
    a.hf_candidates = list(cfg.get("model.hf_candidates") or [])
    a.load_4bit = a.load_4bit or bool(cfg.get("finetune.load_4bit"))
    a.force_torch = a.force_torch or bool(cfg.get("finetune.force_torch"))
    if not cfg.get("finetune.gguf", True):
        a.no_gguf = True
    if not cfg.get("ui.dashboard", True):
        a.no_dash = True
    if not cfg.get("ui.open_browser", True):
        a.no_browser = True
    Stop.install()

    n = int(re.sub(r"\D", "", a.version) or 0) or Registry.next_n()
    vid = a.version or ("v" + str(n))
    out_dir = os.path.join(STATE, a.run, "ft", vid)
    os.makedirs(out_dir, exist_ok=True)
    prog_path = os.path.join(out_dir, "progress.json")
    if a.restart and os.path.exists(out_dir):
        log("reset", "discarding previous progress for " + vid, "yellow")
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    prog = read_json(prog_path, {"version": vid, "n": n, "started": time.time(), "phase": None})
    holder = [prog]
    save = lambda: write_json(prog_path, prog)

    parent, a.rounds = plan_parent(a)
    prog["parent"] = parent["id"] if parent else "base"

    g = gpu_status()
    Log.block(["", Log.paint("  Recursive self-improvement - phase 2: fine-tuning " + vid, "bold"),
               "  config     " + os.path.basename(a.config),
               "  from       " + (parent["id"] + " (stacking: its weights are trained further)" if parent
                                  else "the base model (a clean run, not compounded on an earlier version)"),
               "  corpus     state/" + a.run + "/corpus" + (" rounds " + a.rounds if a.rounds else " (all rounds)"),
               "  recipe     LoRA r=" + str(a.rank) + " alpha=" + str(a.alpha) + ", " + str(a.epochs) +
               " epochs, lr " + str(a.lr) + ", seq " + str(a.seq_len) +
               ", effective batch " + str(a.batch * a.accum),
               "  machine    " + (g["name"] + " " + str(g["total"]) + " MB " +
                                  ("unified memory" if g.get("unified") else "VRAM") if g else "CPU only") +
               ("  (MLX)" if ARM_MAC and not a.force_torch else "  (PyTorch)"),
               "  output     " + out_dir, ""])

    if not a.no_dash:
        Dash(make_api(a, holder), "finetune").start(a.dash_port, not a.no_browser)

    try:
        # ---- 1. dataset
        phase("dataset", "collecting accepted answers")
        data = os.path.join(out_dir, "dataset.jsonl")
        if not prog.get("dataset") or not os.path.exists(data):
            data, stats = build_dataset(a, out_dir)
            check_duplicate(a, vid, parent, stats)
            prog["dataset"] = stats
            save()
        else:
            log("dataset", str(prog["dataset"]["examples"]) + " training examples (already built)")

        # ---- 2. dependencies
        phase("deps", "preparing the python environment")
        backend = install_deps(a)
        prog["backend"] = backend
        save()

        # ---- 3. base weights
        phase("weights", "base model in Hugging Face format")
        base_hf = fetch_base_hf(a)          # always needed: the fair-comparison base GGUF comes from it
        if parent:
            base_hf = parent["merged"]
            log("weights", "stacking on top of " + parent["id"] + " (" + base_hf + ")")
        prog["base_hf"] = base_hf
        save()

        # ---- 4 + 5. train and merge
        phase("train", vid)
        if not prog.get("train_done"):
            run_training(a, base_hf, data, out_dir, prog, save)
        else:
            log("train", "training for " + vid + " was already finished")
        merged = os.path.join(out_dir, "merged")
        if not os.path.exists(os.path.join(merged, "config.json")):
            sys.exit("the merged model is missing in " + merged + " - re-run to continue training")

        # ---- 6. gguf (the fine-tuned model, and the base in the same precision so the comparison is fair)
        gguf = None
        if not a.no_gguf:
            phase("gguf", "converting to GGUF")
            gguf = convert_gguf(a, merged, os.path.join(MODELS, vid, vid + "-f16.gguf"), vid)
            reg = Registry.load()
            if not (reg.get("base_f16") or {}).get("gguf"):
                bg = convert_gguf(a, os.path.join(MODELS, "base_hf"),
                                  os.path.join(MODELS, "base_f16", "base-f16.gguf"), "base model")
                if bg:
                    reg["base_f16"] = {"id": "base_f16", "name": "base-f16", "gguf": bg, "params": {},
                                       "note": "the untouched base model in the same precision as the "
                                               "fine-tuned versions, so benchmark comparisons are like for like"}
                    Registry.save(reg)
            prog["gguf"] = gguf
            save()

        # ---- 7. register
        phase("register", vid)
        # `rounds` is the union of what this version saw and what its parent had already seen, so a chain of
        # stacked versions knows its whole history and never re-trains on the same round twice.
        mine = sorted(int(k) for k in (prog.get("dataset", {}).get("rounds") or {}))
        inherited = sorted(int(x) for x in ((parent or {}).get("rounds") or []))
        info = {"id": vid, "n": n, "name": vid, "gguf": gguf, "merged": merged,
                "adapter": os.path.join(out_dir, "adapter"),
                "parent": parent["id"] if parent else "base", "from": a.from_model,
                "created": time.time(), "run": a.run,
                "examples": prog.get("dataset", {}).get("examples"),
                "fingerprint": prog.get("dataset", {}).get("fingerprint"),
                "trained_on": mine, "rounds": sorted(set(mine + inherited)),
                "recipe": {"rank": a.rank, "alpha": a.alpha, "epochs": a.epochs, "lr": a.lr,
                           "seq_len": a.seq_len, "batch": a.batch, "accum": a.accum, "backend": backend},
                "steps": prog.get("step"), "params": {}}
        Registry.add_version(info)
        prog["phase"] = "done"
        prog["finished"] = time.time()
        save()
        size = 0
        for root_, _, fs in os.walk(out_dir):
            size += sum(os.path.getsize(os.path.join(root_, f)) for f in fs
                        if os.path.exists(os.path.join(root_, f)))
        if gguf and os.path.exists(gguf):
            size += os.path.getsize(gguf)
        log("done", vid + " is ready and registered" + (" -> " + gguf if gguf else " (Hugging Face format only)"),
            "green")
        log("disk", vid + " keeps " + format(size / 1e9, ".1f") + " GB (checkpoints, merged weights, GGUF). "
            "Delete " + os.path.join(out_dir, "ckpt") + " once you are happy with it.")
        log("next", "measure it:  python benchmark.py --models base," + vid +
            "    then keep going:  python run.py --generator " + vid + " --new", "bold")
    except Interrupted:
        log("stop", "stopped cleanly. Progress is in " + prog_path +
            "; re-run the same command to continue from the last checkpoint.", "yellow")
    except KeyboardInterrupt:
        Stop.set()
        log("stop", "interrupted; re-run to resume", "yellow")
    finally:
        Stop.run_cleanup()


if __name__ == "__main__":
    main()
