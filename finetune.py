#!/usr/bin/env python3
"""Phase 2 of the loop: fine-tune the model on the answers it produced.

    python finetune.py

One command again. It builds the training set out of the whole corpus, creates a project virtual
environment, installs the right stack for this machine (PyTorch + PEFT on NVIDIA/CPU, MLX on Apple silicon),
downloads the base weights in Hugging Face format, trains a LoRA, merges it, converts the result to GGUF and
registers it as the next model version, which `run.py --generator latest` and `benchmark.py` then pick up.

Seven phases, each recorded the moment it finishes:
    dataset -> deps -> base weights -> train -> merge -> gguf -> register
Re-running skips everything already done; an interrupted training resumes from its last checkpoint.

The recipe - epochs, LoRA rank, learning rate, sequence length, whether only the correct answers are trained
on - lives in the `finetune` block of `config.json`, and a flag overrides it for one run.

`python finetune.py --remove v1` deletes a version, so the next run trains it again from scratch.
"""
import argparse, ast, concurrent.futures as cf, hashlib, json, os, re, shutil, subprocess, sys, tarfile, time

from common import (DATA, Dash, Interrupted, LOGS, Log, MODELS, Registry, STATE, Stop, Venv, WIN, ARM_MAC,
                    chatml_prompt, common_state, download, fmt_t, gpu_status, log, phase, read_json, read_jsonl,
                    write_json)
import codecheck
import config as conf
import prompts

# llama.cpp's converter is convert_hf_to_gguf.py plus the `conversion` package and the `gguf-py` that sit beside
# it, all from the same revision: the script on its own is a shell that fails on `import conversion`. One source
# archive of llama.cpp has all three.
CONVERTER_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master"
CONVERTER_PARTS = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/")
PHASES = ["dataset", "deps", "base", "train", "merge", "gguf", "register"]
LIVE_TRAIN = {"steps": [], "state": {}, "log": []}
# How running out of memory reads from PyTorch (CUDA, MPS) and from MLX, whose Metal driver says "Insufficient
# Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".
OOM = re.compile(r"out of memory|outofmemory|insufficient memory|metal::malloc", re.I)
# Metal dropping the work for any other reason: "Discarded (victim of GPU error/recovery)" when the GPU resets
# over something anywhere on the Mac, timeouts, hangs. The settings are not at fault, so the same ones go again
# from the last checkpoint.
GPU_FAULT = re.compile(r"command buffer execution failed|kIOGPUCommandBufferCallbackError|GPU error/recovery", re.I)
RETRIES = 3         # failed attempts in a row with no new checkpoint between them, and training gives up
DATA_FORMAT = "chatml, empty reasoning block, loss on the answer"      # how a corpus row becomes training text

# Which config key backs each flag. A flag the user actually passes wins; anything else comes from the file.
CONFIG_MAP = {
    "from_model": "finetune.from_model", "hf_base": "model.hf_repo",
    "epochs": "finetune.epochs", "batch": "finetune.batch", "accum": "finetune.accum",
    "seq_len": "finetune.seq_len", "lr": "finetune.lr", "rank": "finetune.rank",
    "alpha": "finetune.alpha", "save_steps": "finetune.save_steps",
    "min_examples": "finetune.min_examples", "only_correct": "finetune.only_correct",
    "exec_timeout": "corpus.exec_timeout",
    "dash_port": "ui.ports.finetune",
}


# ---------------------------------------------------------------- the training set
def chatml(question, answer):
    """One training example: the prompt exactly as a model is later asked with it (common.chatml_prompt), the
    answer after it, and how many characters of it are prompt. The loss is taken on the answer only."""
    prompt = chatml_prompt(prompts.SFT_SYSTEM, question.strip())
    return prompt + answer.strip() + "<|im_end|>", len(prompt)


def asserts_at_top(code):
    """True when the code has assert statements of its own at module level, which ran when the checker ran it."""
    try:
        return any(isinstance(node, ast.Assert) for node in ast.parse(code).body)
    except SyntaxError:
        return False


def correctness(answer, timeout):
    """None when an answer is correct as far as it can be checked here, otherwise why it is not. Correct means:
    the code is complete, the checker finds no errors in it, it runs, and every check it carries holds - its own
    assert examples, doctests and test_ functions - of which it has at least one, since an answer with nothing to
    check it against is not known to be correct. The checker is the one run.py scores answers with."""
    rep, code = codecheck.check(answer, run=True, timeout=timeout)
    ex = rep.get("exec") or {}
    doc, tests, own = ex.get("doctest") or {}, ex.get("tests") or [], ex.get("examples")
    if not code:
        return "no code"
    if rep.get("truncated"):
        return "cut off"
    if (rep.get("counts") or {}).get("error"):
        return "checker errors"
    if ex.get("timeout"):
        return "times out"
    if not ex.get("import_ok"):
        return "crashes"
    if (own and not own.get("ok")) or doc.get("failed") or doc.get("error") or any(not t.get("ok") for t in tests):
        return "fails its own asserts"
    if not ((own and own.get("ok")) or doc.get("attempted") or tests or asserts_at_top(code)):
        return "has no asserts to check"
    return None


def answer_key(it):
    return hashlib.sha1(str(it.get("answer") or "").encode("utf-8", "replace")).hexdigest()


def verify(items, a, out_dir):
    """correctness() for every answer, several at a time. The verdicts are kept in out_dir/verdicts.json as they
    come in, so a stopped run does not check the same answers twice - unless the checker has changed since, when
    they are all checked again. Returns {answer_key: reason or None}."""
    path = os.path.join(out_dir, "verdicts.json")
    kept = read_json(path, {}) or {}
    cache = kept.get("verdicts", {}) if kept.get("checker") == codecheck.VERSION else {}
    save = lambda: write_json(path, {"checker": codecheck.VERSION, "verdicts": cache})
    todo = list({answer_key(it): it for it in items if answer_key(it) not in cache}.items())
    if todo:
        phase("dataset", "checking which answers are correct")
        log("dataset", "checking " + str(len(todo)) + " answers: does each one run and pass its own asserts?")
        codecheck.ruff_available()          # installed once here, not by every worker at the same moment
        done, t0 = 0, time.time()
        pool = cf.ThreadPoolExecutor(max_workers=max(2, min(16, os.cpu_count() or 4)))
        futs = {pool.submit(correctness, it["answer"], a.exec_timeout): key for key, it in todo}
        try:
            for f in cf.as_completed(futs):
                if Stop.is_set():
                    raise Interrupted()
                try:
                    cache[futs[f]] = f.result() or ""
                except Exception as e:
                    cache[futs[f]] = "check failed (" + type(e).__name__ + ")"
                done += 1
                if done % 100 == 0:
                    save()
                    log("dataset", "  " + str(done) + "/" + str(len(todo)) + " checked, eta " +
                        fmt_t((time.time() - t0) / done * (len(todo) - done)))
        finally:
            for f in futs:
                f.cancel()
            pool.shutdown(wait=not Stop.is_set())
            save()
    return {k: (v or None) for k, v in cache.items()}


def corpus_rounds(run):
    """Which corpus rounds exist on disk."""
    d = os.path.join(STATE, run, "corpus")
    if not os.path.isdir(d):
        return []
    return sorted(int(re.findall(r"\d+", f)[0]) for f in os.listdir(d) if re.match(r"round_\d+\.jsonl$", f))


def rnd(it):
    """The round a corpus row belongs to. 0 when the row predates the field or carries junk in it."""
    try:
        return int(it.get("round") or 0)
    except (TypeError, ValueError):
        return 0


def short_path(path):
    try:
        return os.path.relpath(path, STATE)
    except ValueError:                      # a different drive on Windows
        return path


def corpus_files(a):
    """Every corpus file to train on: this machine's, plus whatever has been merged in.

    A corpus built on another machine is combined by dropping its files in beside the local ones. The
    search is recursive, so `state/<run>/corpus/from-laptop/round_001.jsonl` is picked up and does not
    collide with the local `round_001.jsonl` - which matters, because every machine starts at round 1 and
    names its files the same way. `--corpus PATH` reads a folder or a file in place, without copying.
    Answers are de-duplicated by exercise afterwards, so the same exercise solved on two machines
    contributes once.
    """
    roots = [os.path.join(STATE, a.run, "corpus")] + [str(p) for p in (getattr(a, "corpus", None) or [])]
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(os.path.abspath(root))
        elif os.path.isdir(root):
            for dirpath, _dirs, names in os.walk(root):
                found += [os.path.abspath(os.path.join(dirpath, n))
                          for n in sorted(names) if n.endswith(".jsonl")]
        elif root not in roots[:1]:
            sys.exit("nothing to read at " + root + " (--corpus wants a .jsonl file or a folder of them)")
    out, dedup = [], set()
    for p in found:                         # the same file named twice is read once
        k = os.path.normcase(p)
        if k not in dedup:
            dedup.add(k)
            out.append(p)
    return out


def build_dataset(a, out_dir):
    """The corpus answers from the requested rounds, newest rounds last, de-duplicated by exercise.

    With `finetune.only_correct` (the default) only the correct answers are trained on - those that run, pass
    their own asserts and have no checker errors (see correctness()). run.py still saves an answer for every
    exercise; this is where the wrong ones are left out, and the log says how many and why. With it off, every
    answer is trained on. The checker score rides along on each row so the report can show the spread.

    A row is read for what it has. `host`, `run`, `kept` and `clean` were added to the format later and a
    corpus written before that has none of them - it still trains, it just reports its machine as unknown.
    The only fields a row cannot do without are the question and the answer.
    """
    files = corpus_files(a)
    if not files:
        sys.exit("no corpus yet in " + os.path.join(STATE, a.run, "corpus") +
                 " - run `python run.py` first")
    want = {int(x) for x in a.rounds.split(",") if x.strip()} if a.rounds else None
    everything, per_file, skipped = [], [], 0
    for path in files:
        got = []
        for it in read_jsonl(path):
            if not isinstance(it, dict) or not str(it.get("question") or "").strip() \
                    or not str(it.get("answer") or "").strip():
                skipped += 1
                continue
            if want is not None and rnd(it) not in want:
                continue
            got.append(it)
        everything += got
        per_file.append((path, len(got)))
    if skipped:
        log("warn", str(skipped) + " corpus row(s) had no question or no answer and were left out", "yellow")
    # Newest rounds last, so that when the same exercise appears twice the later answer is the one kept.
    # Between two machines' round 1 the order is by host name, which is arbitrary but at least stable.
    everything.sort(key=lambda it: (rnd(it), str(it.get("host") or ""), str(it.get("qid") or "")))
    dropped = {}
    if a.only_correct:          # before de-duplicating, so a wrong later answer cannot push out a correct one
        verdicts = verify(everything, a, out_dir)
        right = [it for it in everything if not verdicts.get(answer_key(it))]
        for it in everything:
            why = verdicts.get(answer_key(it))
            if why:
                dropped[why] = dropped.get(why, 0) + 1
        log("dataset", str(len(right)) + " of " + str(len(everything)) + " answers are correct (they run, pass "
            "their own asserts and have no checker errors) - training on those only" +
            ("; left out: " + ", ".join(str(n) + " " + why for why, n in
                                        sorted(dropped.items(), key=lambda x: -x[1])) if dropped else ""), "bold")
        everything = right
    rows, seen, per_round, hosts = [], {}, {}, {}
    for it in everything:
        key = re.sub(r"\W+", "", (it.get("title") or it.get("qid") or it.get("question", ""))[:120].lower())
        seen[key] = it
    if len(files) > 1:
        log("corpus", str(len(files)) + " corpus files, " + str(len(everything)) + " answers, " +
            str(len(seen)) + " after de-duplicating by exercise")
        for path, n in per_file:
            log("corpus", "    " + str(n).rjust(5) + "  " + short_path(path), "dim")
    for key, it in seen.items():
        n, host = rnd(it), str(it.get("host") or "?")
        per_round[n] = per_round.get(n, 0) + 1
        hosts[host] = hosts.get(host, 0) + 1
        text, prompt_chars = chatml(it["question"], it["answer"])
        rows.append({"text": text, "prompt_chars": prompt_chars, "qid": it.get("qid") or key[:24], "round": n,
                     "host": it.get("host"), "score": it.get("check_score"), "chars": len(text)})
    rows.sort(key=lambda r: (r["round"], str(r["host"] or ""), r["qid"]))
    if len(rows) < a.min_examples:
        sys.exit("only " + str(len(rows)) + (" correct" if a.only_correct else "") + " answers in the corpus "
                 "(need at least " + str(a.min_examples) + ")" +
                 (" - " + str(sum(dropped.values())) + " more were left out as not correct, and "
                  "`--all-answers` trains on them too" if dropped else "") +
                 ". Generate more with:  python run.py --new")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dataset.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # What the data is, for the duplicate check: which answers, and how they were turned into training text - a
    # version trained before the text had the reasoning block, or before the wrong answers were left out, is not a
    # copy of one trained now on the same rounds.
    fingerprint = hashlib.sha1((DATA_FORMAT + ("|correct only|" if a.only_correct else "|") + "|".join(
        r["qid"] + ":" + str(r["score"]) for r in rows)).encode()).hexdigest()[:16]
    stats = {"examples": len(rows), "rounds": per_round, "hosts": hosts, "sources": len(files),
             "chars": sum(r["chars"] for r in rows),
             "avg_chars": round(sum(r["chars"] for r in rows) / len(rows)), "fingerprint": fingerprint,
             "only_correct": bool(a.only_correct), "left_out": dropped,
             "checker": codecheck.VERSION if a.only_correct else None}
    named = [h for h in hosts if h != "?"]      # a corpus written before `host` existed reports as "?"
    note = ([("across " + str(len(named)) + " machines")] if len(named) > 1 else []) + \
           ([(str(hosts["?"]) + " with no machine stamp")] if hosts.get("?") and named else [])
    log("dataset", str(len(rows)) + " training examples from rounds " +
        ", ".join(str(k) for k in sorted(per_round)) +
        (" (" + ", ".join(note) + ")" if note else "") +
        ", avg " + str(stats["avg_chars"]) + " chars")
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
    checkpoint, so the next run continues from there. A failure restarts it from that checkpoint: with smaller
    settings after running out of memory, with the same ones after a GPU fault - for as long as the attempts
    keep saving checkpoints, and RETRIES times in a row when they do not."""
    ladder = [{"seq_len": a.seq_len, "batch": a.batch, "load_4bit": a.load_4bit},
              {"seq_len": max(512, a.seq_len // 2), "batch": 1, "load_4bit": a.load_4bit},
              {"seq_len": max(512, a.seq_len // 2), "batch": 1, "load_4bit": True}]
    mlx = prog.get("backend") == "mlx"
    if mlx:
        ladder = ladder[:2]     # MLX training has no 4-bit mode, so a third rung would repeat the second
    # The MLX worker measures what fits in the memory free when it starts and shortens the sequences itself, so
    # each run starts from the configured length; PyTorch starts from the settings that last worked.
    attempt = 0 if mlx else min(prog.get("train_attempt", 0), len(ladder) - 1)
    runs = stalled = 0
    while True:
        if Stop.is_set():
            raise Interrupted()
        cfg = ladder[attempt]
        prog["train_attempt"] = attempt
        prog["train_cfg"] = cfg
        save()
        runs += 1
        args = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_worker.py"),
                "--base", base_hf, "--data", data, "--out", out_dir,
                "--epochs", str(a.epochs), "--batch", str(cfg["batch"]), "--accum", str(a.accum),
                "--seq-len", str(cfg["seq_len"]), "--lr", str(a.lr), "--rank", str(a.rank),
                "--alpha", str(a.alpha), "--save-steps", str(a.save_steps), "--merge"]
        if cfg["load_4bit"]:
            args.append("--load-4bit")
        log("train", ("starting" if runs == 1 else "restarting from the last checkpoint") +
            (" (attempt " + str(attempt + 1) + ")" if attempt else "") +
            ": seq " + str(cfg["seq_len"]) + ", batch " + str(cfg["batch"]) + " x accum " + str(a.accum) +
            ", lr " + str(a.lr) + ", rank " + str(a.rank) + (", 4-bit base" if cfg["load_4bit"] else ""), "bold")
        logf = open(os.path.join(LOGS, "train.log"), "a", encoding="utf-8")
        logf.write("\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " attempt " + str(attempt + 1) +
                   (", run " + str(runs) if runs > 1 else "") + " =====\n")
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
        p = subprocess.Popen([Venv.python()] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, errors="replace", **kw)
        from common import kill_tree
        Stop.on_exit(lambda: kill_tree(p))
        oom = fault = False
        t_last, clock, before = time.time(), {}, prog.get("last_checkpoint")
        for line in p.stdout:
            logf.write(line)
            line = line.rstrip()
            if Stop.is_set():
                kill_tree(p)
                break
            if OOM.search(line):        # raw or inside a PROGRESS event: the worker reports errors both ways
                oom = True
            elif GPU_FAULT.search(line):
                fault = True
            if line.startswith("PROGRESS "):
                ev = json.loads(line[9:])
                handle_progress(ev, prog, save, clock)
            else:
                LIVE_TRAIN["log"].append(line[:300])
                del LIVE_TRAIN["log"][:-200]
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
        progress = prog.get("last_checkpoint") != before
        stalled = 0 if progress else stalled + 1
        if oom and mlx and progress:
            again = "out of memory, after training for a while: less is free now than when it started"
        elif oom and attempt + 1 < len(ladder):
            log("warn", "out of memory - retrying with smaller settings", "yellow")
            attempt += 1
            continue
        elif fault and not oom and stalled < RETRIES:
            again = "the GPU dropped the training's work (a GPU reset or fault, not memory)"
        else:
            why = (", out of memory at seq " + str(cfg["seq_len"]) if oom else
                   ", " + str(stalled) + " GPU faults in a row with no checkpoint between them" if fault else "")
            sys.exit("training failed (exit " + str(rc) + why + "); the full output is in logs/train.log")
        log("warn", again + " - restarting from the last checkpoint in 30 s", "yellow")
        if Stop.winding_down() or Stop.sleep(30):
            raise Interrupted()


def handle_progress(ev, prog, save, clock):
    kind = ev.get("kind")
    if kind == "step":
        LIVE_TRAIN["steps"].append({"step": ev.get("step"), "loss": ev.get("loss"),
                                    "lr": ev.get("lr"), "t": ev.get("t")})
        del LIVE_TRAIN["steps"][:-4000]
        LIVE_TRAIN["state"].update(step=ev.get("step"), max_steps=ev.get("max_steps"),
                                   loss=ev.get("loss"), epoch=ev.get("epoch"))
        prog["step"], prog["max_steps"] = ev.get("step"), ev.get("max_steps")
        st, mx = ev.get("step") or 0, ev.get("max_steps") or 0
        if "step" not in clock:         # this run's first step: a resumed run starts part-way through
            clock.update(step=st, t=time.time())
        if st and st % 10 == 0:
            done, el = st - clock["step"], time.time() - clock["t"]
            log("train", "step " + str(st) + "/" + str(mx) + "  loss " +
                (format(ev["loss"], ".4f") if ev.get("loss") is not None else "-") +
                "  epoch " + str(ev.get("epoch", "?")) +
                "  eta " + (fmt_t(el / done * (mx - st)) if mx and done > 0 else "?") +
                ("  peak mem " + format(ev["peak_gb"], ".1f") + " GB" if ev.get("peak_gb") else ""))
            save()
    elif kind == "val":
        if ev.get("loss") is not None:
            log("train", "validation loss " + format(ev["loss"], ".4f") + " at step " + str(ev.get("step")))
    elif kind == "trained":
        if ev.get("kept_step") is not None:
            prog["kept"] = {"step": ev["kept_step"], "val_loss": ev.get("val_loss")}
            save()
            log("train", "keeping the adapter from step " + str(ev["kept_step"]) + ", where the validation loss was "
                "lowest" + (" (" + format(ev["val_loss"], ".4f") + ")" if ev.get("val_loss") is not None else ""),
                "green")
    elif kind == "warn":
        log("warn", str(ev.get("msg"))[:300], "yellow")
    elif kind == "checkpoint":
        prog["last_checkpoint"] = ev.get("step")
        save()
        log("train", "checkpoint at step " + str(ev.get("step")) + " (a Ctrl-C here loses nothing)")
    elif kind == "setup":
        LIVE_TRAIN["state"].update(device=ev.get("device"), gpu=ev.get("gpu"))
        log("train", "device " + str(ev.get("device")) + " - " + str(ev.get("gpu", "")) +
            ("; sequences up to " + str(ev["max_len"]) + " tokens" if ev.get("max_len") else ""))
        if ev.get("recurrence"):
            log("train", "linear attention: " + ev["recurrence"])
        if ev.get("loss"):
            log("train", "loss: " + ev["loss"])
    elif kind == "memory":
        steps = ", ".join(str(gb) + " GB at " + n + " tokens" for n, gb in (ev.get("measured") or {}).items())
        log("train", "memory: " + str(ev.get("free_gb")) + " GB free of " + str(ev.get("ram_gb")) + " GB, budget " +
            str(ev.get("budget_gb")) + " GB" + ("; a training step takes " + steps if steps else "") +
            " -> training at " + str(ev.get("max_len")) + " tokens")
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
def converter():
    """llama.cpp's Hugging Face -> GGUF converter, unpacked once into data/llama.cpp-convert/: the script, its
    `conversion` package and the gguf-py of the same revision, which the script puts ahead of any installed gguf
    by itself. It is unpacked into a temporary folder first, so an interrupted run never leaves half of it."""
    home = os.path.join(DATA, "llama.cpp-convert")
    script = os.path.join(home, "convert_hf_to_gguf.py")
    if os.path.exists(script):
        return script
    archive = download(CONVERTER_URL, os.path.join(DATA, "llama.cpp-source.tar.gz"))
    tmp = home + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""     # below the archive's top folder
                parts = rel.split("/")
                if m.isfile() and rel.startswith(CONVERTER_PARTS) and ".." not in parts:
                    out = os.path.join(tmp, *parts)
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with t.extractfile(m) as src, open(out, "wb") as f:
                        shutil.copyfileobj(src, f)
        why = None if os.path.exists(os.path.join(tmp, "convert_hf_to_gguf.py")) else "no convert_hf_to_gguf.py in it"
    except Exception as e:          # a download cut short, most likely
        why = repr(e)[:160]
    os.remove(archive)
    if why:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit("could not unpack llama.cpp's converter from its source archive (" + why + "); re-run to "
                 "download it again")
    shutil.rmtree(home, ignore_errors=True)
    os.replace(tmp, home)
    return script


def convert_gguf(a, hf_dir, out_gguf, label):
    """A Hugging Face model folder -> one f16 GGUF, the file llama-server loads. The converter's own output
    goes to logs/gguf.log, and when it fails, the end of it is shown. Returns the GGUF, or None."""
    if os.path.exists(out_gguf):
        return out_gguf
    script = converter()
    # the converter needs torch and transformers; gguf-py comes with it and needs the other four
    Venv.need(["numpy", "yaml", "tqdm", "requests"], ["numpy", "pyyaml", "tqdm", "requests"],
              note="what gguf-py needs")
    Venv.need(["transformers"], ["transformers"], note="transformers (the tokenizer)")
    Venv.need(["torch"], ["torch"], index=torch_index(), note="PyTorch")
    log("gguf", "converting " + label + " to GGUF f16 (this is what llama-server loads)")
    os.makedirs(os.path.dirname(out_gguf), exist_ok=True)
    tmp = out_gguf + ".partial"         # renamed when complete, so a cut-off file is never taken for a GGUF
    env = {k: v for k, v in os.environ.items() if k != "NO_LOCAL_GGUF"}    # always the gguf-py beside it
    path = os.path.join(LOGS, "gguf.log")
    with open(path, "a", encoding="utf-8") as lf:
        lf.write("\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " " + label + ": " + hf_dir + " =====\n")
        lf.flush()
        r = Venv.run([script, hf_dir, "--outfile", tmp, "--outtype", "f16"], stdout=lf, stderr=subprocess.STDOUT,
                     env=env)
    if r.returncode != 0 or not os.path.exists(tmp):
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4096))
            tail = [l.strip() for l in re.split(r"[\r\n]+", f.read().decode("utf-8", "replace")) if l.strip()]
        log("warn", "GGUF conversion failed for " + label + " (exit " + str(r.returncode) + "): " +
            (tail[-1][:240] if tail else "no output") + " - the whole output is in logs/gguf.log", "yellow")
        return None
    os.replace(tmp, out_gguf)
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
def unconverted(a):
    """The newest version when it was registered without its GGUF - its conversion failed, back when a failed
    conversion did not stop the run - and so can be neither benchmarked nor served: the next run finishes it
    rather than train a new version on the same data. Its id, or None."""
    last = (Registry.load().get("versions") or [None])[-1]
    if a.version or a.no_gguf or not last or last.get("gguf") or last.get("run", a.run) != a.run:
        return None
    return last["id"] if os.path.exists(os.path.join(str(last.get("merged")), "config.json")) else None


def _inside(path, root):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:                      # a different drive on Windows
        return False


def _du(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs
               if os.path.isfile(os.path.join(d, f)))


def remove_version(vid):
    """Delete a fine-tuned version for good, so the next fine-tune makes it again under the same name from
    scratch: its registry entry, its training folder (dataset, adapter, merged weights), its GGUF, and its
    benchmark results in every run - the benchmark keeps answers by model name, so a new model of the same name
    would otherwise inherit the old one's. The rounds it wrote as the generator are moved to
    state/<run>/removed/<version>-<time>/ rather than deleted, and a pipeline cycle still answering with it is
    closed, so the pipeline starts a new one."""
    here = os.path.dirname(os.path.abspath(__file__))
    rel = lambda p: os.path.relpath(p, here) if _inside(p, here) else p
    reg = Registry.load()
    entry = next((v for v in reg.get("versions", []) if v.get("id") == vid), None)
    if not entry:
        sys.exit("there is no version '" + vid + "' to remove (registered: " +
                 (", ".join(v["id"] for v in reg.get("versions", [])) or "none") + ")")
    kids = [v["id"] for v in reg["versions"] if v.get("parent") == vid]
    if kids:
        sys.exit(vid + " cannot be removed on its own: " + ", ".join(kids) + " " +
                 ("were" if len(kids) > 1 else "was") + " trained on top of it. Remove " +
                 ("those" if len(kids) > 1 else "it") + " first.")
    runs = sorted(d for d in os.listdir(STATE) if os.path.isdir(os.path.join(STATE, d))) if os.path.isdir(STATE) else []
    ft = os.path.dirname(str(entry.get("merged") or ""))
    if not (os.path.basename(ft) == vid and _inside(ft, STATE)):
        ft = os.path.join(STATE, str(entry.get("run") or "default"), "ft", vid)
    gg = os.path.dirname(str(entry.get("gguf") or ""))
    if not (os.path.basename(gg) == vid and _inside(gg, MODELS)):
        gg = os.path.join(MODELS, vid)
    freed = 0
    for d in [ft, gg] + [os.path.join(STATE, r, "bench", vid) for r in runs]:
        if os.path.isdir(d):
            size = _du(d)
            shutil.rmtree(d)
            freed += size
            log("remove", "deleted " + rel(d) + " (" + format(size / 1e9, ".2f") + " GB)")
    stamp_ = time.strftime("%Y%m%d-%H%M%S")
    for r in runs:
        rounds_dir, corpus_dir = os.path.join(STATE, r, "rounds"), os.path.join(STATE, r, "corpus")
        mine = sorted(n for n in (os.listdir(rounds_dir) if os.path.isdir(rounds_dir) else [])
                      if re.match(r"round_\d+$", n) and
                      (read_json(os.path.join(rounds_dir, n, "info.json"), {}) or {}).get("generator") == vid)
        if not mine:
            continue
        away = os.path.join(STATE, r, "removed", vid + "-" + stamp_)
        os.makedirs(os.path.join(away, "rounds"), exist_ok=True)
        os.makedirs(os.path.join(away, "corpus"), exist_ok=True)
        for n in mine:
            os.replace(os.path.join(rounds_dir, n), os.path.join(away, "rounds", n))
            if os.path.exists(os.path.join(corpus_dir, n + ".jsonl")):
                os.replace(os.path.join(corpus_dir, n + ".jsonl"), os.path.join(away, "corpus", n + ".jsonl"))
        nums = {int(n.split("_")[1]) for n in mine}
        meta_path = os.path.join(STATE, r, "meta.json")
        meta = read_json(meta_path, None)
        if isinstance(meta, dict) and meta.get("rounds"):
            meta["rounds"] = [x for x in meta["rounds"] if not (x.get("n") in nums and x.get("generator") == vid)]
            write_json(meta_path, meta)
        log("remove", "moved the " + str(len(mine)) + " round(s) " + vid + " wrote in run '" + r + "' (" +
            ", ".join(str(x) for x in sorted(nums)) + ") to " + rel(away) + " - delete that folder if you do "
            "not want them back")
    for r in runs:
        pj = os.path.join(STATE, r, "pipeline.json")
        st = read_json(pj, None)
        last = (st or {}).get("cycles", [None])[-1] if isinstance(st, dict) and st.get("cycles") else None
        if not (st and st.get("job") and last and not last.get("finished")):
            continue
        if last.get("generator") == vid:
            last["abandoned"] = vid + " was removed"
            st.pop("job", None)
            write_json(pj, st)
            log("remove", "closed pipeline cycle " + str(last.get("n")) + " of run '" + r + "', which was answering "
                "with " + vid + ": the next pipeline run starts a new cycle")
        elif last.get("produced") == vid:        # stopped after training it: the same command trains it again
            last["done"] = [s for s in last.get("done", []) if s == "corpus"]
            last.pop("produced", None)
            last.pop("versions_before", None)
            write_json(pj, st)
            log("remove", "pipeline cycle " + str(last.get("n")) + " of run '" + r + "' had trained " + vid +
                ": running the pipeline again trains it anew")
    reg["versions"] = [v for v in reg["versions"] if v.get("id") != vid]
    Registry.save(reg)
    log("done", vid + " is removed (" + format(freed / 1e9, ".1f") + " GB freed). The next `python finetune.py` "
        "trains " + "v" + str(Registry.next_n()) + " from scratch.", "green")


def main():
    O = conf.opt()      # a flag with no default: when it is not passed, config.json decides
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    conf.add_args(ap)
    ap.add_argument("--run", default="default")
    ap.add_argument("--rounds", default="", help="which corpus rounds to train on (default: all)")
    ap.add_argument("--corpus", action="append", default=[], metavar="PATH",
                    help="another machine's corpus to train on as well - a .jsonl file or a folder of "
                         "them, repeatable. Copying those files into state/<run>/corpus/ does the same "
                         "thing, at any depth, so the filenames cannot collide")
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
    ap.add_argument("--min-examples", type=int, default=O)
    ap.add_argument("--only-correct", dest="only_correct", action="store_true", default=O,
                    help="train only on answers that run, pass their own asserts and have no checker errors "
                         "(the default: finetune.only_correct in config.json)")
    ap.add_argument("--all-answers", dest="only_correct", action="store_false", default=O,
                    help="train on every answer in the corpus, correct or not")
    ap.add_argument("--remove", default="", metavar="VERSION",
                    help="delete a fine-tuned version - its registry entry, training files, GGUF and benchmark "
                         "results - so the next run trains it again from scratch; the rounds it wrote are "
                         "moved to state/<run>/removed/")
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
    if a.remove:
        remove_version(a.remove.strip())
        return
    a.exec_timeout = int(a.exec_timeout or 25)
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

    todo = unconverted(a)
    if todo:
        a.version = todo
        log("resume", todo + " is trained but has no GGUF, because its conversion did not finish; this run "
                             "converts it instead of training a new version", "yellow")

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
               "  corpus     state/" + a.run + "/corpus" + (" rounds " + a.rounds if a.rounds else " (all rounds)") +
               (", correct answers only" if a.only_correct else ", every answer"),
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
        phase("dataset", "collecting the corpus answers")
        data = os.path.join(out_dir, "dataset.jsonl")
        # A training set built with the other setting, before there was one, or by an older checker is built again
        # while training has not finished - the worker then starts over, since its data changed. A finished
        # training keeps its own.
        built = prog.get("dataset") or {}
        why = ("before only the correct answers could be chosen" if "only_correct" not in built else
               "with the other --only-correct setting" if built["only_correct"] != bool(a.only_correct) else
               "by an older checker, which took correct code for wrong" if a.only_correct and
               built.get("checker") != codecheck.VERSION else "")
        stale = bool(built) and not prog.get("train_done") and bool(why)
        if stale:
            log("dataset", "the training set was built " + why + "; building it again")
        if not prog.get("dataset") or not os.path.exists(data) or stale:
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
            if not gguf:
                sys.exit(vid + " is trained and merged, but converting it to GGUF failed (why: the end of "
                         "logs/gguf.log), so it is not registered - the benchmark and run.py load GGUFs. Re-run to "
                         "try the conversion again; the training is kept.")
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
                           "seq_len": a.seq_len, "batch": a.batch, "accum": a.accum, "backend": backend,
                           "only_correct": bool((prog.get("dataset") or {}).get("only_correct")),
                           "data": DATA_FORMAT},
                "kept": prog.get("kept"), "steps": prog.get("step"), "params": {}}
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
