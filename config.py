#!/usr/bin/env python3
"""One config file for the whole project.

`config.json` in this folder decides which model is used, how many solvers and how many judges there are,
what each of them is told, how much runs in parallel, which benchmarks are measured and where they are
downloaded from. Nothing here has to be edited in code: change the file, re-run, and the model or the
benchmark named in it is fetched automatically on first use.

    python run.py --print-config          # show what is in effect right now
    python run.py --config other.json     # use a different file
    python run.py --init-config           # rewrite config.json with the defaults

The file is read with comments (`//` and `#`) and trailing commas allowed, and any long piece of text may be
written as a list of lines instead of one string with `\\n` in it - which is what makes the prompts readable.
A command-line flag always wins over the file; the file always wins over the built-in default.
"""
import argparse, json, os, re, sys

import codecheck
import prompts
from common import LlamaServer, ROOT, log

CONFIG_PATH = os.path.join(ROOT, "config.json")

# Text lives under "prompts", and under the "prompt" key of one solver or judge. Only those are written as
# lists of lines and joined back together on load; every other list stays a real list.
TEXT_KEYS = ("prompt", "note", "warn", "title")


# ---------------------------------------------------------------- the defaults, and what each one means
def _sampler(name, prompt, temperature, top_k, top_p, min_p=None):
    d = {"name": name, "temperature": temperature, "top_k": top_k, "top_p": top_p}
    if min_p is not None:
        d["min_p"] = min_p
    d["prompt"] = prompt
    return d


DEFAULTS = {
    "language": "Python",

    "model": {
        "base": "qwen3.5:0.8b",
        "generator": "base",
        "judge": "base",
        "hf_repo": "",
        "hf_candidates": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B-Instruct", "Qwen/Qwen3-0.6B"],
    },

    "parallel": {
        "workers": 0,
        "slots": 0,
        "max_slots": 32,
        "backend": "auto",
    },

    "corpus": {
        "questions_per_round": 24,
        "rounds": 1,
        "answer_tokens": 20000,
        "judge_tokens": 3000,
        "question_tokens": 4000,
        "summary_tokens": 700,
        "question_temp": 1.0,
        "refine_temp": 0.35,
        "judge_weight": 0.7,
        "min_score": 55,
        "recall_full": 60,
        "recall_chars": 60000,
        "exec_timeout": 25,
        "run_code": True,
        "ctx": 0,
        "difficulty_mix": {"easy": 0.30, "hard": 0.25},
        "short_share": 0.6,
        "axis_weights": {"correctness": 0.45, "robustness": 0.2, "efficiency": 0.15, "style": 0.2},
    },

    "solvers": [
        _sampler("careful", prompts.SOLVER_PERSONAS["careful"], 0.25, 20, 0.85, 0.05),
        _sampler("efficient", prompts.SOLVER_PERSONAS["efficient"], 0.70, 50, 0.95, 0.02),
        _sampler("pythonic", prompts.SOLVER_PERSONAS["pythonic"], 1.00, 80, 0.98, 0.00),
    ],

    "judges": [
        _sampler("bug-hunter", prompts.JUDGE_PERSONAS["bug-hunter"], 0.10, 15, 0.80),
        _sampler("architect", prompts.JUDGE_PERSONAS["architect"], 0.45, 40, 0.92),
        _sampler("pragmatist", prompts.JUDGE_PERSONAS["pragmatist"], 0.75, 60, 0.96),
    ],

    "checker": {
        "ruff_rules": "E,F,W,B,C4,UP,SIM,RET,ARG,PIE",
        "ruff_ignore": "W291,W292,W293",
        "penalties": {"error": 34, "warning": 4, "info": 1, "timeout": 25, "import_error": 30,
                      "doctest_failure": 8, "test_failure": 10, "examples_fail": 22},
        "bonuses": {"has_tests": 4, "has_type_hints": 2},
    },

    "finetune": {
        "from_model": "base",
        "epochs": 3.0,
        "batch": 1,
        "accum": 16,
        "seq_len": 2048,
        "lr": 1e-4,
        "rank": 32,
        "alpha": 64,
        "save_steps": 25,
        "min_score": 60,
        "min_examples": 20,
        "load_4bit": False,
        "force_torch": False,
        "gguf": True,
    },

    "benchmarks": {
        "use": ["lbpp", "evoeval"],
        "solve_tokens": 2200,
        "temp": 0.0,
        "test_timeout": 30,
        "limit": 0,
        "self_check": True,
        "probe": {"enabled": True, "tokens": 320, "overlap": 0.45, "run": 28},
        "catalog": {
            "lbpp": {
                "style": "lbpp",
                "harness": "lbpp",
                "packed": True,
                "source": {"kind": "hf-parquet", "repo": "CohereLabs/lbpp", "path": "python/test.parquet"},
                "fields": {"task_id": "task_id", "title": "title", "prompt": "instruction",
                           "signature": "signature", "test_file": "test_file", "test_setup": "test_setup",
                           "test_list": "test_list", "reference": "completion"},
                "note": "Less Basic Python Problems (Cohere, 162). Published with every field zlib+base64 "
                        "encoded so crawlers never ingest it as plain text. Harder than HumanEval.",
            },
            "evoeval": {
                "style": "evalplus",
                "harness": "evalplus",
                "source": {"kind": "hf-jsonl", "repo": "evoeval/EvoEval_{split}", "path": "test.jsonl",
                           "splits": ["subtle", "creative"],
                           "all_splits": ["subtle", "creative", "difficult", "combine", "tool_use"]},
                "fields": {"task_id": "task_id", "prompt": "prompt", "entry_point": "entry_point",
                           "test": "test", "reference": "canonical_solution"},
                "reference_join": True,
                "note": "Every HumanEval task rewritten so the memorised HumanEval answer is the wrong one. "
                        "Same difficulty as HumanEval, but nothing can be answered from memory.",
            },
            "humanevalplus": {
                "style": "evalplus",
                "harness": "evalplus",
                "source": {"kind": "hf-parquet", "repo": "evalplus/humanevalplus",
                           "path": "data/test-00000-of-00001-5973903632b82d40.parquet"},
                "fields": {"task_id": "task_id", "prompt": "prompt", "entry_point": "entry_point",
                           "test": "test", "reference": "canonical_solution"},
                "reference_join": True,
                "note": "Public since 2021, so contaminated for any recent model. Off by default.",
            },
            "mbppplus": {
                "style": "mbpp",
                "harness": "evalplus",
                "source": {"kind": "hf-parquet", "repo": "evalplus/mbppplus",
                           "path": "data/test-00000-of-00001-d5781c9c51e02795.parquet"},
                "fields": {"task_id": "task_id", "prompt": "prompt", "entry_point": "entry_point",
                           "test": "test", "test_list": "test_list", "reference": "code"},
                "id_prefix": "Mbpp/",
                "note": "Public since 2021, so contaminated for any recent model. Off by default.",
            },
            "lcb": {
                "style": "lcb",
                "harness": "lcb",
                "packed": True,
                "self_check": False,
                "source": {"kind": "hf-jsonl", "repo": "livecodebench/code_generation_lite",
                           "paths": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl",
                                     "test5.jsonl", "test6.jsonl"]},
                "newest": 150,
                "after": "2024-08-01",
                "max_tests": 18,
                "warn": "LiveCodeBench ships its test cases inline: the download is several GB, and "
                        "competitive-programming problems are far above a 0.8B model.",
            },
        },
    },

    "ui": {
        "dashboard": True,
        "open_browser": True,
        "report_every": 60,
        "ports": {"corpus": 8777, "finetune": 8778, "benchmark": 8779},
    },

    "server": {"port_corpus": 11500, "port_benchmark": 11600},

    "prompts": {
        "question_system": prompts.QUESTION_SYSTEM,
        "question_user": prompts.QUESTION_USER,
        "avoid_none": prompts.AVOID_NONE,
        "avoid_some": prompts.AVOID_SOME,
        "solver_base": prompts.SOLVER_BASE,
        "solver_user": prompts.SOLVER_USER,
        "judge_base": prompts.JUDGE_BASE,
        "judge_user": prompts.JUDGE_USER,
        "judge_answer_block": prompts.JUDGE_ANSWER_BLOCK,
        "lint_summary_system": prompts.LINT_SUMMARY_SYSTEM,
        "lint_summary_user": prompts.LINT_SUMMARY_USER,
        "refine_system": prompts.REFINE_SYSTEM,
        "refine_user": prompts.REFINE_USER,
        "bench_system": prompts.BENCH_SYSTEM,
        "bench_user_func": prompts.BENCH_USER_FUNC,
        "bench_user_stdin": prompts.BENCH_USER_STDIN,
        "bench_complete_stub": prompts.BENCH_COMPLETE_STUB,
        "probe_system": prompts.PROBE_SYSTEM,
        "probe_user": prompts.PROBE_USER,
        "sft_system": prompts.SFT_SYSTEM,
    },
}

# A comment written above the key when config.json is generated. Dotted paths; "*" matches a list element.
COMMENTS = {
    "language": "The one language the loop works in. Qwen's code training is overwhelmingly Python and\n"
                "every coding benchmark its authors report is Python, so that is where it has something\n"
                "to improve on. Changing this alone does not translate the prompts - edit those too.",
    "model": "Which weights everything runs on. `base` is an Ollama name and is pulled automatically the\n"
             "first time it is needed, so putting a different model here is all that is required to\n"
             "switch. `generator` is who writes the questions and the answers (base | latest | v2 | a\n"
             "path to a .gguf); `judge` is deliberately kept on the untouched base model, which is what\n"
             "makes each round's scores comparable to the last. `hf_repo` is only needed for training -\n"
             "leave it empty and the candidates below are tried in order.",
    "parallel": "How much runs at once. 0 means: measure the GPU and decide. `workers` is exercises in\n"
                "flight, `slots` is llama-server's parallel decoding slots, `backend` is auto | llama |\n"
                "ollama. Every solver and every judge of one exercise already runs concurrently.",
    "corpus": "Phase 1 (run.py). `answer_tokens` is the ceiling for one answer - the judges read all of\n"
              "them whole, never truncated, so this also decides how big the judge context has to be.\n"
              "`judge_weight` splits the winner decision between the judges and the programmatic\n"
              "checker (1 = judges alone, 0 = checker alone). `recall_full` / `recall_chars` decide how\n"
              "much of the previous questionnaire is shown when writing the next one.",
    "corpus.difficulty_mix": "Fractions of each round that are easy and hard; the rest are medium.",
    "corpus.short_share": "Fraction of each round that is a short exercise; the rest are long.",
    "corpus.axis_weights": "How the four judge axes combine into one number when ranking answers.",
    "solvers": "The agents that answer each exercise, one entry each - add or remove entries to change\n"
               "how many there are. They all run on the same model and differ only in their sampling\n"
               "and in the `prompt` below, which is appended to the shared solver instructions.",
    "judges": "The agents that score the answers. Same model again, different sampling, different bias.\n"
              "Each judge sees every answer in full and in its own order, so position cannot decide the\n"
              "winner. Add or remove entries to change how many judges there are.",
    "checker": "The programmatic check every answer goes through: compile, an AST pass, ruff, and an\n"
               "actual run of the code. `penalties` and `bonuses` are points off and on a 0-100 score.",
    "finetune": "Phase 2 (finetune.py). `from_model` is `base` for a clean run from the untouched model\n"
                "on the whole corpus, or `latest`/`v1` to keep training that version's own weights on\n"
                "the rounds it has not seen yet.",
    "benchmarks": "Phase 3 (benchmark.py). `use` picks from `catalog` below; anything named there is\n"
                  "downloaded on first use. `probe` is the memorization screen - every problem is shown\n"
                  "half-finished to the base model, and one it can reproduce verbatim is flagged, so\n"
                  "every score is reported twice: over all problems, and over the unseen ones only.",
    "benchmarks.catalog": "Every benchmark this project knows how to fetch and run. To add one, copy the\n"
                          "closest entry and change `source` and `fields`.\n"
                          "  style    lbpp | evalplus | mbpp | lcb - how a downloaded row becomes a problem\n"
                          "  harness  lbpp | evalplus | lcb - how a candidate answer is executed\n"
                          "  source   kind hf-parquet | hf-jsonl | url-jsonl | url-parquet, plus repo/path\n"
                          "           (`{split}` in a repo name expands over `splits`)\n"
                          "  packed   true when the dataset base64+zlib encodes its fields\n"
                          "  fields   which column of the download holds each part of a problem",
    "ui": "The live dashboard each phase serves on 127.0.0.1.",
    "server": "Where the model servers listen. Only worth changing if something else owns these ports.",
    "prompts": "Every prompt, in full. `{...}` markers are filled in by the code and must stay; anything\n"
               "else is yours to rewrite. In the judge prompts `{count}`, `{slots}`, `{schema}` and\n"
               "`{best_slots}` expand to match however many judges and solvers are configured above.",
}


# ---------------------------------------------------------------- reading a file with comments
def strip_jsonc(text):
    """Remove // and # comments and trailing commas, leaving the strings alone."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _join_text(obj, key=None, parent=None):
    """A prompt may be written as a list of lines. Join those back, and leave every other list alone."""
    if isinstance(obj, dict):
        return {k: _join_text(v, k, obj if parent is None else parent) for k, v in obj.items()}
    if isinstance(obj, list):
        if key in TEXT_KEYS and all(isinstance(x, str) for x in obj):
            return "\n".join(obj)
        return [_join_text(x, key) for x in obj]
    return obj


def _join_prompts(data):
    """Everything under "prompts" is text, plus the "prompt" of each solver and judge."""
    out = dict(data)
    if isinstance(out.get("prompts"), dict):
        out["prompts"] = {k: ("\n".join(v) if isinstance(v, list) else v) for k, v in out["prompts"].items()}
    for role in ("solvers", "judges"):
        if isinstance(out.get(role), list):
            out[role] = [_join_text(x) if isinstance(x, dict) else x for x in out[role]]
    return _join_text(out)


def deep_merge(base, over):
    """`over` wins, key by key. A list in `over` replaces the list in `base` outright: three judges in the
    file means three judges, not three added to the defaults."""
    if not isinstance(over, dict) or not isinstance(base, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(base[k], v) if k in base and isinstance(base[k], dict) else v
    return out


# ---------------------------------------------------------------- writing a readable file back out
def _is_text(key):
    return key in TEXT_KEYS


def _dump(obj, indent, key=None, path=""):
    pad = "  " * indent
    if isinstance(obj, str) and ("\n" in obj) and (_is_text(key) or path.startswith("prompts.")):
        lines = obj.split("\n")
        inner = ",\n".join(pad + "  " + json.dumps(l, ensure_ascii=False) for l in lines)
        return "[\n" + inner + "\n" + pad + "]"
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        parts = []
        for k, v in obj.items():
            sub = path + ("." if path else "") + str(k)
            note = COMMENTS.get(sub)
            line = pad + "  " + json.dumps(str(k)) + ": " + _dump(v, indent + 1, k, sub)
            if note:
                # the comment belongs to the key below it, so it has to sit inside the same entry -
                # otherwise the separating comma lands at the end of the comment instead of the value
                head = "\n".join(pad + "  // " + l for l in note.split("\n"))
                line = ("" if not parts else "\n") + head + "\n" + line
            parts.append(line)
        return "{\n" + ",\n".join(parts) + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in obj) and len(json.dumps(obj)) < 96:
            return json.dumps(obj, ensure_ascii=False)
        inner = ",\n".join(pad + "  " + _dump(x, indent + 1, key, path + ".*") for x in obj)
        return "[\n" + inner + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def dump_jsonc(data):
    header = ("// RecursiveLLMImprovement - the one file that decides what the loop does.\n"
              "//\n"
              "// Comments (// and #) and trailing commas are allowed, and any long piece of text may be\n"
              "// written as a list of lines. A command-line flag overrides whatever is set here.\n"
              "// Delete this file and re-run to get a fresh copy of the defaults.\n")
    return header + _dump(data, 0) + "\n"


def write_default(path=CONFIG_PATH, data=None):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dump_jsonc(data or DEFAULTS))
    return path


# ---------------------------------------------------------------- the config object
class Config:
    def __init__(s, data, path):
        s.data, s.path = data, path

    def get(s, dotted, default=None):
        cur = s.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def __getitem__(s, k):
        return s.data[k]

    @property
    def solvers(s):
        return s.data["solvers"]

    @property
    def judges(s):
        return s.data["judges"]

    def bench(s, name):
        return (s.get("benchmarks.catalog") or {}).get(name)

    def as_json(s):
        return json.dumps(s.data, ensure_ascii=False, indent=1)


def validate(data, path):
    """Fail early and say exactly which line of the file is wrong, rather than half way through a run."""
    bad = []
    for role in ("solvers", "judges"):
        items = data.get(role)
        if not isinstance(items, list) or not items:
            bad.append('"' + role + '" must be a list with at least one entry')
            continue
        if len(items) > 26:
            bad.append('"' + role + '" has ' + str(len(items)) + " entries; the answers are labelled A-Z, "
                       "so 26 is the most there can be")
        for i, it in enumerate(items):
            if not isinstance(it, dict) or not str(it.get("name") or "").strip():
                bad.append('"' + role + '"[' + str(i) + '] needs a "name"')
            elif not str(it.get("prompt") or "").strip():
                bad.append('"' + role + '"[' + str(i) + '] (' + str(it.get("name")) + ') needs a "prompt"')
    names = [str(x.get("name")) for x in data.get("judges", []) if isinstance(x, dict)]
    if len(set(names)) != len(names):
        bad.append("two judges share a name; each one needs its own")
    catalog = (data.get("benchmarks") or {}).get("catalog") or {}
    for name in (data.get("benchmarks") or {}).get("use") or []:
        if name not in catalog:
            bad.append('benchmarks.use names "' + name + '", which is not in benchmarks.catalog (have: ' +
                       ", ".join(sorted(catalog)) + ")")
    for name, spec in catalog.items():
        if spec.get("style") not in ("lbpp", "evalplus", "mbpp", "lcb"):
            bad.append('benchmark "' + name + '" has style ' + repr(spec.get("style")) +
                       "; it must be one of lbpp, evalplus, mbpp, lcb")
        if spec.get("harness") not in ("lbpp", "evalplus", "lcb"):
            bad.append('benchmark "' + name + '" has harness ' + repr(spec.get("harness")) +
                       "; it must be one of lbpp, evalplus, lcb")
        if not (spec.get("source") or {}).get("kind"):
            bad.append('benchmark "' + name + '" has no source.kind')
    jw = (data.get("corpus") or {}).get("judge_weight")
    if jw is None or not 0 <= float(jw) <= 1:
        bad.append("corpus.judge_weight must be between 0 and 1")
    if bad:
        sys.exit("config error in " + path + ":\n  - " + "\n  - ".join(bad))
    return data


def unknown_keys(user, known=None, path=""):
    """Keys the code never reads. A misspelled `judge_wieght` would otherwise sit in the file looking
    effective and do nothing at all, which is the worst way for a config file to fail."""
    known = DEFAULTS if known is None else known
    # free-form by design: the benchmarks the user invents, and the fields of one solver or judge
    if path in ("benchmarks.catalog", "solvers", "judges") or path.startswith("benchmarks.catalog."):
        return []
    out = []
    for k, v in (user or {}).items():
        sub = path + ("." if path else "") + str(k)
        if k not in known:
            out.append(sub)
        elif isinstance(v, dict) and isinstance(known[k], dict):
            out += unknown_keys(v, known[k], sub)
    return out


_ACTIVE = {"cfg": None}


def load(path=None, quiet=False):
    """Read config.json (creating it from the defaults the first time), merge it over the defaults, and
    push the prompts and the agent lists into `prompts` so every script sees the same thing."""
    path = path or os.environ.get("RLI_CONFIG") or CONFIG_PATH
    if not os.path.exists(path):
        write_default(path)
        if not quiet:
            log("config", "wrote " + os.path.basename(path) + " - the model, the agents, the prompts and "
                "the benchmarks all live there now")
    try:
        raw = open(path, "r", encoding="utf-8").read()
    except OSError as e:
        sys.exit("could not read " + path + ": " + str(e))
    try:
        user = json.loads(strip_jsonc(raw) or "{}")
    except ValueError as e:
        sys.exit("config error in " + path + ": " + str(e) + "\n"
                 "(comments and trailing commas are fine; a missing comma or quote is not)")
    if not isinstance(user, dict):
        sys.exit("config error in " + path + ": the file must hold one JSON object")
    stray = unknown_keys(user)
    if stray and not quiet:
        log("warn", os.path.basename(path) + ": nothing reads " + ", ".join(stray[:6]) +
            (" (+" + str(len(stray) - 6) + " more)" if len(stray) > 6 else "") +
            " - a typo? Run with --init-config elsewhere to see every key.", "yellow")
    data = validate(deep_merge(DEFAULTS, _join_prompts(user)), path)
    cfg = Config(data, path)
    _ACTIVE["cfg"] = cfg
    prompts.bind(cfg.data)          # every prompt, and how many solvers and judges there are
    codecheck.bind(cfg.data)        # the lint rules and how the checker scores an answer
    LlamaServer.MAX_SLOTS = int(cfg.get("parallel.max_slots") or LlamaServer.MAX_SLOTS)
    return cfg


def active():
    return _ACTIVE["cfg"] or load(quiet=True)


# ---------------------------------------------------------------- wiring it to the command line
def add_args(ap):
    """The three flags every script shares. Everything else is per-script and listed in its own MAP."""
    ap.add_argument("--config", default="", metavar="PATH",
                    help="which config file to use (default: config.json next to these scripts)")
    ap.add_argument("--init-config", action="store_true",
                    help="write a fresh config.json with the defaults and exit")
    ap.add_argument("--print-config", action="store_true",
                    help="print the configuration actually in effect and exit")
    return ap


def opt(default=argparse.SUPPRESS):
    """Argparse default for a flag that is backed by the config file: when it is not passed, it is simply
    absent from the namespace, and `apply` fills it in from the file."""
    return default


def apply(a, cfg, mapping):
    """Fill every config-backed flag that the user did not pass on the command line."""
    for attr, dotted in mapping.items():
        if not hasattr(a, attr):
            setattr(a, attr, cfg.get(dotted))
    return a


def handle_meta(a, cfg=None):
    """--init-config / --print-config, which both end the run."""
    if getattr(a, "init_config", False):
        p = write_default(getattr(a, "config", "") or CONFIG_PATH)
        print("wrote " + p)
        return True
    if getattr(a, "print_config", False):
        c = cfg or load(getattr(a, "config", "") or None, quiet=True)
        print("# " + c.path)
        print(c.as_json())
        return True
    return False
