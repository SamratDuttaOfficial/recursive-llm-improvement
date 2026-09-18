#!/usr/bin/env python3
"""Phase 3 of the loop: measure whether the fine-tuning actually helped.

    python benchmark.py

Downloads the benchmarks, runs every registered model that has no results yet, executes the real unit tests
in a separate process with a timeout, and writes a comparison table. The base model is measured once and
then never again - later runs only measure the new fine-tuned versions.

Which benchmarks run is `benchmarks.use` in `config.json`, and every one it can fetch is described in
`benchmarks.catalog` there - repository, file, how its columns map onto a problem, and which harness
executes a candidate answer. Nothing below knows the name of any particular benchmark, so adding one is an
edit to that file, not to this code, and it downloads itself the first time it is named.

Two Python benchmarks are on by default:

  lbpp           Less Basic Python Problems (Cohere, 162 problems). Written for the paper "On Leakage of
                 Code Generation Evaluation Datasets" as a benchmark that was fresh and unleaked at
                 creation, and published with every field zlib+base64 encoded so that crawlers never ingest
                 the problems or their solutions as plain text. Harder than HumanEval and MBPP.

  evoeval        EvoEval (200 problems): every HumanEval task rewritten so that the memorised HumanEval
                 solution is the *wrong* answer - `subtle` changes the requirement in ways that are easy to
                 miss, `creative` restates it as an unrelated story. Same difficulty as HumanEval, which is
                 where a 0.8B model can still score, but nothing can be answered from memory.

Also in the catalog but off by default: `humanevalplus` and `mbppplus` (public since 2021, so contaminated
for any recent model) and `lcb` (LiveCodeBench; its test cases ship inline, so the download runs to several
GB, and its competitive-programming problems are far above a 0.8B model).

Neither default benchmark can be answered from memory by construction, but that is not taken on trust:
every problem is also put through a *memorization probe* against the base model - the guided-prompting
test, where the model is shown the first part of the statement and asked to reproduce the rest verbatim.
Problems it can reproduce are flagged, and every score is reported twice: over all problems, and over the
subset the model demonstrably has not memorised.
"""
import argparse, base64, concurrent.futures as cf, io, json, os, pickle, re, subprocess, sys, tempfile
import time, zlib

import config as conf
import prompts
from codecheck import extract_code
from common import (DATA, Dash, Gen, Interrupted, Log, Registry, STATE, Stop, Backend, Venv,
                    check_stop, common_state, download, fmt_t, hf_url, log, phase,
                    read_json, read_jsonl, write_json)

# Which config key backs each flag. A flag the user actually passes wins; anything else comes from the file.
CONFIG_MAP = {
    "model": "model.base", "solve_tokens": "benchmarks.solve_tokens", "temp": "benchmarks.temp",
    "test_timeout": "benchmarks.test_timeout", "limit": "benchmarks.limit",
    "probe_tokens": "benchmarks.probe.tokens", "probe_overlap": "benchmarks.probe.overlap",
    "probe_run": "benchmarks.probe.run",
    "slots": "parallel.slots", "backend": "parallel.backend",
    "port": "server.port_benchmark", "dash_port": "ui.ports.benchmark",
}


# ---------------------------------------------------------------- safe decoding of packed dataset fields
class SafeUnpickler(pickle.Unpickler):
    """These datasets pack plain strings inside pickles. Refuse every global, so nothing can be constructed
    and no code from the download can run - only the built-in types survive."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError("refused to load " + module + "." + name + " from a dataset field")


B64 = re.compile(r"^[A-Za-z0-9+/=\s]{24,}$")


def unpack(value):
    """These datasets pack their fields to keep them out of web crawls. Peel the layers off in the order they
    were applied - base64, then zlib, then either UTF-8 or a pickle - and finally parse the JSON document
    that is usually inside. A field that is already plain text comes back unchanged."""
    if value is None or isinstance(value, (list, dict)):
        return value
    raw = value
    if isinstance(raw, str):
        if not B64.match(raw):
            return value                     # plain text, e.g. the instruction and the signature
        try:
            raw = base64.b64decode(raw, validate=False)
        except Exception:
            return value
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    obj = raw
    if isinstance(raw, bytes):
        try:
            obj = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                obj = SafeUnpickler(io.BytesIO(raw)).load()   # a pickled str, never a constructed object
            except Exception:
                return None
    if isinstance(obj, (bytes, bytearray)):
        try:
            obj = obj.decode("utf-8")
        except Exception:
            return None
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return obj
    return obj


# ---------------------------------------------------------------- dataset preparation
def reference_path(bench):
    return os.path.join(DATA, bench + "_reference.json")


def join_solution(prompt, solution):
    """Rebuild a complete reference program from a stub plus its body. Some prompts end with the closing
    `\"\"\"` of the docstring and no newline, others already end with one, so plain concatenation produces a
    syntax error for a large share of the problems - which would then look like the harness failing them."""
    prompt, solution = prompt or "", solution or ""
    for candidate in (prompt + solution, prompt.rstrip("\n") + "\n" + solution, solution):
        try:
            compile(candidate, "<ref>", "exec")
            return candidate
        except SyntaxError:
            continue
    return prompt + "\n" + solution


def parquet_rows(pq_path):
    """Turn a downloaded parquet file into rows with the venv's pyarrow (the orchestrator stays stdlib)."""
    Venv.need(["pyarrow"], ["pyarrow"], note="pyarrow (reading the benchmark parquet)")
    raw_path = pq_path + ".jsonl"
    code = ("import json,sys,pyarrow.parquet as pq;"
            "t=pq.read_table(sys.argv[1]).to_pylist();"
            "open(sys.argv[2],'w',encoding='utf-8').write("
            "'\\n'.join(json.dumps(r,default=str) for r in t))")
    if Venv.run(["-c", code, pq_path, raw_path]).returncode != 0:
        sys.exit("could not read " + pq_path)
    rows = read_jsonl(raw_path)
    try:
        os.remove(raw_path)
    except OSError:
        pass
    return rows


def field(row, fields, key, default=""):
    """Read one part of a problem out of a downloaded row, using the column names the config gives."""
    name = fields.get(key)
    if not name:
        return default
    v = row.get(name)
    return default if v is None else v


def fetch_rows(name, source, split=""):
    """Download whatever a benchmark's `source` block points at and return its rows.

       hf-parquet / url-parquet   a parquet file, read through the venv's pyarrow
       hf-jsonl   / url-jsonl     one JSON object per line

    `{split}` in a repo name or a path expands, so one entry can cover a family of splits."""
    kind = str(source.get("kind") or "hf-jsonl")
    repo = str(source.get("repo") or "").replace("{split}", split)
    paths = [str(p) for p in (source.get("paths") or [source.get("path") or ""])]
    rows = []
    for path in paths:
        path = path.replace("{split}", split)
        url = path if kind.startswith("url") else hf_url(repo, path, source.get("hf_kind", "datasets"))
        stem = re.sub(r"[^\w.-]", "_", "_".join(x for x in (name, split, os.path.basename(path)) if x))
        local = os.path.join(DATA, stem)
        try:
            download(url, local)
        except Interrupted:
            raise
        except Exception as e:
            log("warn", name + ": could not fetch " + url + " (" + repr(e)[:90] + ")", "yellow")
            continue
        rows += parquet_rows(local) if "parquet" in kind else read_jsonl(local)
    if not rows:
        sys.exit("nothing could be downloaded for benchmark '" + name + "'. Check its `source` block in "
                 "config.json (repo/path), or remove it from benchmarks.use.")
    return rows


def build_lbpp(rows, spec, name, split=""):
    """A benchmark whose problems ship a whole pytest file. LBPP also packs every field to keep it out of
    web crawls, which is what `packed` turns on."""
    f, dec = spec.get("fields") or {}, (unpack if spec.get("packed") else (lambda x: x))
    items, refs = [], {}
    for r in rows:
        tid = str(field(r, f, "task_id"))
        tests = dec(field(r, f, "test_file"))
        setup = dec(field(r, f, "test_setup")) or ""
        tl = dec(field(r, f, "test_list")) or []
        if not isinstance(tests, str) or not tests.strip():
            if isinstance(tl, list) and tl:
                tests = setup + "\n" + "\n".join(str(x) for x in tl)
            else:
                continue
        items.append({"task_id": tid, "bench": name, "title": str(field(r, f, "title")),
                      "prompt": dec(field(r, f, "prompt")) or "",
                      "signature": dec(field(r, f, "signature")) or "",
                      "test_file": tests, "test_setup": setup,
                      "n_tests": len(tl) if isinstance(tl, list) else 1,
                      "categories": r.get("categories"), "difficulty": split or name, "date": None,
                      "kind": spec.get("harness", "lbpp")})
        refs[tid] = dec(field(r, f, "reference")) or ""
    return items, refs


def build_evalplus(rows, spec, name, split=""):
    """A benchmark that gives a function stub and a self-contained test program: HumanEval+ and everything
    shaped like it, including EvoEval's rewrites. The stub is shown to the model as the thing to complete."""
    f = spec.get("fields") or {}
    items, refs = [], {}
    for r in rows:
        raw_id = str(field(r, f, "task_id"))
        tid = (split + "/" + raw_id.split("/")[-1]) if split else (spec.get("id_prefix", "") + raw_id)
        stub = str(field(r, f, "prompt")).rstrip()
        entry = str(field(r, f, "entry_point"))
        ref = field(r, f, "reference") or ""
        items.append({"task_id": tid, "bench": name, "title": entry or tid,
                      "prompt": prompts.fill(prompts.BENCH_COMPLETE_STUB, stub=stub),
                      "signature": "", "entry_point": entry, "test": field(r, f, "test") or "",
                      "test_imports": field(r, f, "test_imports", []) or [], "n_tests": 1,
                      "difficulty": split or name, "date": None,
                      "kind": spec.get("harness", "evalplus")})
        refs[tid] = join_solution(stub, ref) if spec.get("reference_join") else ref
    return items, refs


def build_mbpp(rows, spec, name, split=""):
    """A benchmark whose problem is prose plus a handful of asserts: the first assert doubles as the
    signature the model has to match."""
    f = spec.get("fields") or {}
    items, refs = [], {}
    for r in rows:
        tid = spec.get("id_prefix", "") + str(field(r, f, "task_id"))
        asserts = field(r, f, "test_list", []) or []
        if isinstance(asserts, str):
            try:
                asserts = json.loads(asserts.replace("'", '"'))
            except Exception:
                asserts = [asserts]
        text = str(field(r, f, "prompt"))
        if asserts:
            text += "\n\nYour function must satisfy:\n" + "\n".join(str(x) for x in asserts[:3])
        items.append({"task_id": tid, "bench": name, "title": str(field(r, f, "entry_point")) or tid,
                      "prompt": text, "signature": str(asserts[0])[:200] if asserts else "",
                      "entry_point": str(field(r, f, "entry_point")), "test": field(r, f, "test") or "",
                      "test_imports": field(r, f, "test_imports", []) or [], "n_tests": 1,
                      "difficulty": split or name, "date": None,
                      "kind": spec.get("harness", "evalplus")})
        refs[tid] = field(r, f, "reference") or ""
    return items, refs


def build_lcb(rows, spec, name, split=""):
    """A competitive-programming benchmark: the test cases ship inline with the problem, which is why the
    download is so large. Problems carry a date, so the newest ones can be kept."""
    f, cap = spec.get("fields") or {}, int(spec.get("max_tests") or 18)
    dec = unpack if spec.get("packed") else (lambda x: x)
    items = []
    for r in rows:
        pub = dec(r.get("public_test_cases")) or []
        priv = dec(r.get("private_test_cases")) or []
        tests = [t for t in (pub if isinstance(pub, list) else []) +
                 (priv if isinstance(priv, list) else []) if isinstance(t, dict)]
        if not tests:
            continue
        meta = dec(r.get("metadata")) or {}
        items.append({"task_id": name + "/" + str(r.get("question_id")), "bench": name,
                      "title": r.get("question_title", ""), "prompt": r.get("question_content", ""),
                      "signature": r.get("starter_code", "") or "",
                      "tests": tests[:cap], "n_tests": min(cap, len(tests)),
                      "func_name": (meta or {}).get("func_name", ""),
                      "difficulty": r.get("difficulty", ""), "platform": r.get("platform", ""),
                      "date": str(r.get("contest_date", ""))[:10],
                      "kind": spec.get("harness", "lcb")})
    uniq = {i["task_id"]: i for i in items}
    return sorted(uniq.values(), key=lambda x: x["date"] or ""), {}


BUILDERS = {"lbpp": build_lbpp, "evalplus": build_evalplus, "mbpp": build_mbpp, "lcb": build_lcb}


def ensure_bench(name, spec):
    """Download a benchmark the first time it is named in the config, turn it into this project's own
    problem format, and cache both that and its reference solutions. Nothing here knows the name of any
    particular benchmark: it all comes out of `benchmarks.catalog`."""
    source = spec.get("source") or {}
    splits = [str(x) for x in (source.get("splits") or [])]
    key = name + ("_" + "_".join(splits) if splits else "")
    out = os.path.join(DATA, key + ".jsonl")
    if os.path.exists(out):
        return read_jsonl(out)
    build = BUILDERS[spec["style"]]
    items, refs = [], {}
    for sp in (splits or [""]):
        rows = fetch_rows(name, source, sp)
        got, ref = build(rows, spec, name, sp)
        items += got
        refs.update(ref)
    if refs:
        # only ever used to check that this harness can run the benchmark's own answers, never in a prompt
        write_json(reference_path(name), refs)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    log("data", name + ": " + str(len(items)) + " problems ready" +
        (" (" + ", ".join(splits) + ")" if splits else ""))
    return items


def self_check(bench, items, a):
    """Run every problem's own reference solution through this harness once and keep only the problems that
    pass. A problem whose official solution fails here cannot tell us anything about the model - it means the
    environment is missing something or the test does not run headless - so scoring it would be noise.
    The verdict is cached, so this costs a minute on the very first run and nothing afterwards."""
    refs = read_json(reference_path(bench), {}) or {}
    if not refs:
        return items
    cache_path = os.path.join(DATA, bench + "_runnable.json")
    cache = read_json(cache_path, {}) or {}
    todo = [it for it in items if it["task_id"] not in cache and refs.get(it["task_id"])]
    if todo:
        # the benchmarks' own solutions and tests use numpy/pandas here and there, and pytest in a few places
        Venv.need(["numpy"], ["numpy"], note="numpy (some benchmark problems need it)")
        Venv.need(["pandas"], ["pandas"], note="pandas (some benchmark problems need it)")
        Venv.need(["pytest"], ["pytest"], note="pytest (some benchmark tests use it)")
        log("check", bench + ": validating the harness against " + str(len(todo)) +
            " reference solutions (once only)")
        with cf.ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
            futs = {ex.submit(evaluate, refs[it["task_id"]], it, max(a.test_timeout, 30)): it for it in todo}
            for f in cf.as_completed(futs):
                it = futs[f]
                try:
                    v = f.result()
                except Exception as e:
                    v = {"ok": False, "why": repr(e)[:80]}
                cache[it["task_id"]] = {"ok": bool(v["ok"]), "why": v.get("why", "")}
        write_json(cache_path, cache)
    good = [it for it in items if (cache.get(it["task_id"]) or {"ok": True})["ok"]]
    dropped = len(items) - len(good)
    if dropped:
        why = {}
        for it in items:
            c = cache.get(it["task_id"])
            if c and not c["ok"]:
                why[c["why"][:40]] = why.get(c["why"][:40], 0) + 1
        log("check", bench + ": using " + str(len(good)) + " of " + str(len(items)) + " problems; " +
            str(dropped) + " dropped because their own reference solution does not run here (" +
            ", ".join(k + " x" + str(v) for k, v in sorted(why.items(), key=lambda x: -x[1])[:3]) + ")")
    return good


def load_bench(name, a):
    """One benchmark, ready to run: downloaded if this is the first time, screened against its own
    reference solutions, and cut to whatever limits the config sets."""
    spec = (a.catalog or {}).get(name)
    if not spec:
        sys.exit("unknown benchmark '" + name + "'. Add it to benchmarks.catalog in " + a.config +
                 ", or pick one of: " + ", ".join(sorted(a.catalog or {})))
    if spec.get("warn"):
        log("warn", str(spec["warn"]) + " Use it only if you mean to.", "yellow")
    items = ensure_bench(name, spec)
    if spec.get("after"):
        items = [i for i in items if (i.get("date") or "") >= str(spec["after"])]
    if spec.get("newest"):
        items = items[-int(spec["newest"]):]   # the newest problems, the least likely to be memorised
    if not a.no_self_check and spec.get("self_check", True):
        items = self_check(name, items, a)
    limit = int(spec.get("limit") or a.limit or 0)
    return items[:limit] if limit else items


# ---------------------------------------------------------------- running one candidate solution
LBPP_HARNESS = """import json, sys, traceback
sys.setrecursionlimit(20000)
try:
    exec(compile(open('tests.py', encoding='utf-8').read(), 'tests.py', 'exec'), {'__name__': '__main__'})
except BaseException:
    traceback.print_exc()
    sys.exit(1)
print('__PASS__')
"""

LCB_FUNC_HARNESS = r'''import json, sys, traceback
sys.setrecursionlimit(20000)
spec = json.load(open('spec.json', encoding='utf-8'))
ns = {}
try:
    exec(compile(open('code.py', encoding='utf-8').read(), 'code.py', 'exec'), ns)
except BaseException:
    traceback.print_exc(); print('__FAIL__ import'); sys.exit(1)
fn = None
if spec['func'] and spec['func'] in ns:
    fn = ns[spec['func']]
elif 'Solution' in ns:
    obj = ns['Solution']()
    fn = getattr(obj, spec['func'], None) or next(
        (getattr(obj, m) for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))), None)
if fn is None:
    print('__FAIL__ no function'); sys.exit(1)
ok = 0
for t in spec['tests']:
    try:
        args = [json.loads(l) for l in str(t['input']).split('\n') if l.strip() != '']
        want = json.loads(t['output']) if str(t['output']).strip() else None
        got = fn(*args)
        if got == want or (isinstance(got, tuple) and list(got) == want) or str(got) == str(want):
            ok += 1
        else:
            print('__FAIL__ wrong answer'); sys.exit(1)
    except BaseException:
        traceback.print_exc(); print('__FAIL__ raised'); sys.exit(1)
print('__PASS__', ok)
'''


def run_lbpp(code, item, timeout):
    with tempfile.TemporaryDirectory(prefix="rli_b_") as d:
        open(os.path.join(d, "code.py"), "w", encoding="utf-8").write(code)
        open(os.path.join(d, "tests.py"), "w", encoding="utf-8").write(item["test_file"])
        open(os.path.join(d, "h.py"), "w", encoding="utf-8").write(LBPP_HARNESS)
        return _subrun([Venv.runner(), "h.py"], d, timeout)


EVALPLUS_HARNESS = r'''import sys, traceback
sys.setrecursionlimit(20000)
ns = {"__name__": "__answer__"}
try:
    exec(compile(open("code.py", encoding="utf-8").read(), "code.py", "exec"), ns)
except BaseException:
    traceback.print_exc(); print("__FAIL__ import"); sys.exit(1)
try:
    exec(compile(open("tests.py", encoding="utf-8").read(), "tests.py", "exec"), ns)
except BaseException:
    traceback.print_exc(); print("__FAIL__ wrong answer"); sys.exit(1)
entry = open("entry.txt", encoding="utf-8").read().strip() if __import__("os").path.exists("entry.txt") else ""
if entry and "check" in ns:
    fn = ns.get(entry)
    if fn is None:
        print("__FAIL__ missing " + entry); sys.exit(1)
    try:
        ns["check"](fn)
    except BaseException:
        traceback.print_exc(); print("__FAIL__ wrong answer"); sys.exit(1)
print("__PASS__")
'''


def run_evalplus(code, item, timeout):
    with tempfile.TemporaryDirectory(prefix="rli_b_") as d:
        open(os.path.join(d, "code.py"), "w", encoding="utf-8").write(code)
        imports = item.get("test_imports") or []
        if isinstance(imports, str):
            imports = [imports]
        tests = "\n".join(str(x) for x in imports) + "\n" + (item.get("test") or "")
        open(os.path.join(d, "tests.py"), "w", encoding="utf-8").write(tests)
        if item.get("entry_point"):
            open(os.path.join(d, "entry.txt"), "w", encoding="utf-8").write(item["entry_point"])
        open(os.path.join(d, "h.py"), "w", encoding="utf-8").write(EVALPLUS_HARNESS)
        return _subrun([Venv.runner(), "h.py"], d, timeout)


def run_lcb(code, item, timeout):
    tests = item.get("tests") or []
    functional = bool(item.get("signature", "").strip()) or any(
        (t.get("testtype") or "") == "functional" for t in tests)
    with tempfile.TemporaryDirectory(prefix="rli_b_") as d:
        open(os.path.join(d, "code.py"), "w", encoding="utf-8").write(code)
        if functional:
            json.dump({"func": item.get("func_name") or "", "tests": tests},
                      open(os.path.join(d, "spec.json"), "w", encoding="utf-8"))
            open(os.path.join(d, "h.py"), "w", encoding="utf-8").write(LCB_FUNC_HARNESS)
            return _subrun([Venv.runner(), "h.py"], d, timeout)
        for t in tests:
            r = _subrun([Venv.runner(), "code.py"], d, timeout, stdin=str(t.get("input", "")))
            if not r["ok"]:
                return r
            got = [l.rstrip() for l in (r["out"] or "").strip().splitlines()]
            want = [l.rstrip() for l in str(t.get("output", "")).strip().splitlines()]
            if got != want:
                return {"ok": False, "why": "wrong answer", "out": (r["out"] or "")[-300:]}
        return {"ok": True, "why": "", "out": ""}


def _subrun(cmd, cwd, timeout, stdin=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
                           input=stdin if stdin is not None else "", errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": "timeout", "out": ""}
    except Exception as e:
        return {"ok": False, "why": repr(e)[:120], "out": ""}
    out = (p.stdout or "")
    if "__PASS__" in out:
        return {"ok": True, "why": "", "out": ""}
    if p.returncode != 0 or "__FAIL__" in out:
        why = "failed"
        m = re.search(r"__FAIL__ (\w[\w ]*)", out)
        if m:
            why = m.group(1)
        else:
            tb = (p.stderr or out).strip().splitlines()
            why = tb[-1][:120] if tb else "failed"
        return {"ok": False, "why": why, "out": (p.stderr or out)[-300:]}
    return {"ok": True, "why": "", "out": out}


RUNNERS = {"lbpp": run_lbpp, "evalplus": run_evalplus, "lcb": run_lcb}


def evaluate(code, item, timeout):
    if not code.strip():
        return {"ok": False, "why": "no code in the answer", "out": ""}
    return RUNNERS[item["kind"]](code, item, timeout)


# ---------------------------------------------------------------- contamination probe
def ngrams(text, n=8):
    toks = re.findall(r"\w+", (text or "").lower())
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def overlap(a, b, n=8):
    A, B = ngrams(a, n), ngrams(b, n)
    return len(A & B) / max(1, min(len(A), len(B)))


def longest_common_run(a, b):
    ta = re.findall(r"\w+", (a or "").lower())
    tb = re.findall(r"\w+", (b or "").lower())
    if not ta or not tb:
        return 0
    prev = [0] * (len(tb) + 1)
    best = 0
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def probe_one(backend, item, a):
    """Guided prompting: show the model the first 45% of the statement and ask for the rest, word for word.
    A model that has memorised the problem reproduces it; one that has not writes something else."""
    text = re.sub(r"\s+", " ", item["prompt"]).strip()
    words = text.split(" ")
    if len(words) < 60:
        return {"skipped": True, "memorized": False, "overlap": 0.0, "run": 0}
    cut = int(len(words) * 0.45)
    first, rest = " ".join(words[:cut]), " ".join(words[cut:])
    g = Gen(tag="probe " + item["task_id"], kind="probe", max_tokens=a.probe_tokens, model="base",
            meta={"task": item["task_id"]})
    try:
        out, _ = backend.chat("judge", prompts.PROBE_SYSTEM,
                              prompts.fill(prompts.PROBE_USER, first=first), a.probe_tokens, 0.0,
                              gen=g, sampler={"top_k": 1})
        g.done()
    except Interrupted:
        g.done("interrupted")
        raise
    except Exception as e:
        g.done(repr(e)[:120])
        return {"skipped": True, "memorized": False, "overlap": 0.0, "run": 0, "error": repr(e)[:120]}
    if out.strip().upper().startswith("UNKNOWN"):
        return {"memorized": False, "overlap": 0.0, "run": 0, "said_unknown": True}
    ov = overlap(out, rest)
    run = longest_common_run(out[:4000], rest[:4000])
    memorized = ov >= a.probe_overlap or run >= a.probe_run
    return {"memorized": bool(memorized), "overlap": round(ov, 3), "run": run,
            "continuation": out[:600]}


def screen(backend, items, bench, a):
    """Run the probe over a benchmark once and cache it: it depends only on the base model."""
    path = os.path.join(DATA, "contamination_" + bench + ".json")
    cache = read_json(path, {}) or {}
    todo = [it for it in items if it["task_id"] not in cache]
    if todo and not a.no_probe:
        phase("probe", bench + ": memorization screen on " + str(len(todo)) + " problems")
        log("probe", bench + ": checking " + str(len(todo)) + " problems for memorisation by the base model")
        workers = max(2, backend.total_slots())
        done = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(probe_one, backend, it, a): it for it in todo}
            for f in cf.as_completed(futs):
                it = futs[f]
                try:
                    cache[it["task_id"]] = f.result()
                except Interrupted:
                    break
                except Exception as e:
                    cache[it["task_id"]] = {"skipped": True, "memorized": False, "error": repr(e)[:100]}
                done += 1
                if done % 25 == 0:
                    write_json(path, cache)
                    log("probe", bench + ": " + str(done) + "/" + str(len(todo)))
        write_json(path, cache)
    flagged = sum(1 for it in items if (cache.get(it["task_id"]) or {}).get("memorized"))
    log("probe", bench + ": " + str(flagged) + " of " + str(len(items)) +
        " problems look memorised by the base model; scores are reported with and without them",
        "yellow" if flagged else "green")
    return cache


# ---------------------------------------------------------------- one model on one benchmark
class Bench:
    def __init__(s, a, backend, run_dir):
        s.a, s.backend, s.dir = a, backend, run_dir
        s.active = {}
        s.live = {}

    def result_path(s, model_id, bench, task_id):
        safe = re.sub(r"[^\w.-]", "_", task_id)
        return os.path.join(s.dir, model_id, bench, safe + ".json")

    def summary_path(s, model_id, bench):
        return os.path.join(s.dir, model_id, bench + ".summary.json")

    def solve_one(s, model_id, item):
        a = s.a
        p = s.result_path(model_id, item["bench"], item["task_id"])
        got = read_json(p)
        if got:
            return got
        check_stop()
        s.active[item["task_id"]] = model_id
        note = ""
        if item.get("signature"):
            note = ("Use exactly this signature:\n\n```python\n" + item["signature"].strip() + "\n```\n\n")
        functional = item["kind"] in ("lbpp", "evalplus") or bool(item.get("signature", "").strip())
        user = (prompts.fill(prompts.BENCH_USER_FUNC, prompt=item["prompt"], signature_note=note)
                if functional else prompts.fill(prompts.BENCH_USER_STDIN, prompt=item["prompt"]))
        g = Gen(tag=model_id + " " + item["task_id"], kind="solve", max_tokens=a.solve_tokens,
                model=model_id, meta={"task": item["task_id"], "bench": item["bench"]})
        t0 = time.time()
        try:
            text, info = s.backend.chat("gen", prompts.BENCH_SYSTEM, user, a.solve_tokens, a.temp,
                                        gen=g, sampler={"top_k": 1} if a.temp == 0 else None)
            g.done()
        except Interrupted:
            g.done("interrupted")
            raise
        except Exception as e:
            g.done(repr(e)[:150])
            s.active.pop(item["task_id"], None)
            raise
        code = extract_code(text)
        verdict = evaluate(code, item, a.test_timeout)
        rec = {"task_id": item["task_id"], "bench": item["bench"], "model": model_id,
               "passed": bool(verdict["ok"]), "why": verdict["why"], "secs": round(time.time() - t0, 1),
               "tokens": info.get("out"), "code": code, "answer": text[:20000],
               "stderr": verdict.get("out", "")[:400], "difficulty": item.get("difficulty"),
               "date": item.get("date"), "when": time.time()}
        write_json(p, rec)
        s.active.pop(item["task_id"], None)
        return rec

    def run_model(s, model_id, bench, items, contam):
        a = s.a
        done = [it for it in items if os.path.exists(s.result_path(model_id, bench, it["task_id"]))]
        todo = [it for it in items if it not in done]
        log("bench", model_id + " on " + bench + ": " + str(len(todo)) + " to run, " +
            str(len(done)) + " already done", "bold")
        phase("benchmark", model_id + " on " + bench)
        workers = max(2, int(s.backend.total_slots() * 1.25))
        n_done, t0 = 0, time.time()
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(s.solve_one, model_id, it): it for it in todo}
            try:
                for f in cf.as_completed(futs):
                    it = futs[f]
                    try:
                        rec = f.result()
                    except Interrupted:
                        break
                    except Exception as e:
                        log("warn", it["task_id"] + " failed: " + repr(e)[:120], "yellow")
                        continue
                    n_done += 1
                    if n_done % 5 == 0 or a.verbose:
                        eta = (time.time() - t0) / n_done * (len(todo) - n_done)
                        log("bench", model_id + " " + bench + " " + str(n_done) + "/" + str(len(todo)) +
                            "  " + it["task_id"] + " " + ("pass" if rec["passed"] else "fail: " + rec["why"][:40]) +
                            "  eta " + fmt_t(eta))
            except (KeyboardInterrupt, Interrupted):
                Stop.set()
            if Stop.is_set():
                for f in futs:
                    f.cancel()
        return s.summarize(model_id, bench, items, contam)

    def summarize(s, model_id, bench, items, contam):
        recs = []
        for it in items:
            r = read_json(s.result_path(model_id, bench, it["task_id"]))
            if r:
                r["memorized"] = bool((contam.get(it["task_id"]) or {}).get("memorized"))
                recs.append(r)
        if not recs:
            return None
        clean = [r for r in recs if not r["memorized"]]
        by_diff = {}
        for r in recs:
            d = r.get("difficulty") or "-"
            by_diff.setdefault(d, []).append(r["passed"])
        out = {"model": model_id, "bench": bench, "n": len(recs), "passed": sum(r["passed"] for r in recs),
               "pass_rate": round(100 * sum(r["passed"] for r in recs) / len(recs), 1),
               "n_clean": len(clean), "passed_clean": sum(r["passed"] for r in clean),
               "pass_rate_clean": round(100 * sum(r["passed"] for r in clean) / len(clean), 1) if clean else None,
               "flagged": len(recs) - len(clean),
               "avg_tokens": round(sum(r.get("tokens") or 0 for r in recs) / len(recs)),
               "by_difficulty": {k: {"n": len(v), "pass": sum(v),
                                     "rate": round(100 * sum(v) / len(v), 1)} for k, v in by_diff.items()},
               "complete": len(recs) == len(items), "when": time.time()}
        write_json(s.summary_path(model_id, bench), out)
        return out


# ---------------------------------------------------------------- reporting
def table(rows):
    if not rows:
        return
    w = Log.paint
    lines = ["", w("  model        benchmark   pass@1        on unseen only     flagged   avg tok", "bold")]
    for r in rows:
        clean = (format(r["pass_rate_clean"], ".1f") + "%  (" + str(r["passed_clean"]) + "/" +
                 str(r["n_clean"]) + ")") if r.get("pass_rate_clean") is not None else "-"
        lines.append("  " + str(r["model"])[:11].ljust(11) + "  " + str(r["bench"]).ljust(10) + "  " +
                     (format(r["pass_rate"], ".1f") + "%").rjust(6) + " (" + str(r["passed"]) + "/" +
                     str(r["n"]) + ")".ljust(2) + "   " + clean.ljust(18) + "  " +
                     str(r["flagged"]).rjust(7) + "   " + str(r["avg_tokens"]).rjust(7) +
                     ("" if r["complete"] else w("   (partial)", "yellow")))
    Log.block(lines + [""])


def deltas(rows):
    """base vs each version, on the uncontaminated subset where there is one."""
    base = {r["bench"]: r for r in rows if r["model"] in ("base", "base_f16")}
    out = []
    for r in rows:
        b = base.get(r["bench"])
        if not b or r["model"] == b["model"]:
            continue
        d = {"model": r["model"], "bench": r["bench"],
             "delta": round(r["pass_rate"] - b["pass_rate"], 1),
             "delta_clean": (round(r["pass_rate_clean"] - b["pass_rate_clean"], 1)
                             if r.get("pass_rate_clean") is not None and b.get("pass_rate_clean") is not None
                             else None)}
        out.append(d)
    if out:
        w = Log.paint
        lines = ["", w("  change against the base model", "bold")]
        for d in out:
            sign = "+" if d["delta"] > 0 else ""
            cl = ("  |  unseen only " + ("+" if (d["delta_clean"] or 0) > 0 else "") + str(d["delta_clean"]) +
                  " pts") if d["delta_clean"] is not None else ""
            lines.append("  " + d["model"].ljust(8) + " " + d["bench"].ljust(6) + "  " +
                         w(sign + str(d["delta"]) + " pts", "green" if d["delta"] > 0 else
                           ("red" if d["delta"] < 0 else "dim")) + cl)
        Log.block(lines + [""])
    return out


# ---------------------------------------------------------------- main
def make_api(a, bench_obj, holder):
    def api(what, q):
        if what == "state":
            return common_state(bench_obj.backend if bench_obj else None, {
                "kind": "benchmark", "run": a.run, "rows": holder["rows"], "deltas": holder["deltas"],
                "active": dict(bench_obj.active) if bench_obj else {},
                "contamination": holder["contam"], "registry": Registry.load(),
                "plan": holder["plan"],
                # the catalog is the whole download spec of every benchmark: useful in the file, noise here
                "config": {k: v for k, v in vars(a).items()
                           if not k.startswith("_") and k != "catalog"}})
        if what == "results":
            model, bench = q.get("model"), q.get("bench")
            d = os.path.join(bench_obj.dir, model, bench)
            if not os.path.isdir(d):
                return {"results": []}
            out = []
            for f in sorted(os.listdir(d)):
                r = read_json(os.path.join(d, f)) or {}
                out.append({k: r.get(k) for k in ("task_id", "passed", "why", "secs", "tokens", "difficulty")})
            return {"model": model, "bench": bench, "results": out}
        if what == "result":
            d = os.path.join(bench_obj.dir, q.get("model", ""), q.get("bench", ""),
                             re.sub(r"[^\w.-]", "_", q.get("task", "")) + ".json")
            return read_json(d)
        return None
    return api


def base_id(run_dir, benches, reg):
    """The name the base model is measured under: whatever it was first measured as, so it is never
    measured twice under two names. Only a base that has never been measured takes the f16 identity."""
    for name in ("base", "base_f16"):
        for b in benches:
            s = read_json(os.path.join(run_dir, name, b + ".summary.json"))
            if s and s.get("complete"):
                return name
    return "base_f16" if reg.get("base_f16") else "base"


def main():
    O = conf.opt()      # a flag with no default: when it is not passed, config.json decides
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    conf.add_args(ap)
    ap.add_argument("--model", default=O, help="the base model (pulled automatically)")
    ap.add_argument("--run", default="default")
    ap.add_argument("--models", default="", help="which to measure (default: base once, then every "
                                                 "registered version that has no results)")
    ap.add_argument("--benchmarks", default="",
                    help="comma list of names from benchmarks.catalog in config.json "
                         "(default: whatever benchmarks.use lists)")
    ap.add_argument("--limit", type=int, default=O, help="cap the problems per benchmark (0 = all)")
    ap.add_argument("--solve-tokens", type=int, default=O)
    ap.add_argument("--temp", type=float, default=O, help="0 = greedy, the standard pass@1 setting")
    ap.add_argument("--test-timeout", type=int, default=O)
    ap.add_argument("--probe-tokens", type=int, default=O)
    ap.add_argument("--probe-overlap", type=float, default=O, help="8-gram overlap that counts as memorised")
    ap.add_argument("--probe-run", type=int, default=O, help="longest verbatim run of words that counts")
    ap.add_argument("--splits", default="", metavar="BENCH=a,b",
                    help="override one benchmark's splits, e.g. --splits evoeval=subtle,creative")
    ap.add_argument("--no-probe", action="store_true", help="skip the contamination screen")
    ap.add_argument("--no-self-check", action="store_true",
                    help="do not drop problems whose own reference solution fails in this environment")
    ap.add_argument("--force", action="store_true", help="re-measure models that already have results")
    ap.add_argument("--slots", type=int, default=O)
    ap.add_argument("--backend", default=O, choices=["auto", "llama", "ollama"])
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--port", type=int, default=O)
    ap.add_argument("--dash-port", type=int, default=O)
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--report", action="store_true", help="print the saved results and exit")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    Log.setup(not a.no_color, a.verbose)
    if conf.handle_meta(a):
        return
    cfg = conf.load(a.config or None)
    conf.apply(a, cfg, CONFIG_MAP)
    a.config = cfg.path
    a.catalog = dict(cfg.get("benchmarks.catalog") or {})
    if not cfg.get("benchmarks.probe.enabled", True):
        a.no_probe = True
    if not cfg.get("benchmarks.self_check", True):
        a.no_self_check = True
    if not cfg.get("ui.dashboard", True):
        a.no_dash = True
    if not cfg.get("ui.open_browser", True):
        a.no_browser = True
    Stop.install()
    a.ctx = a.ctx or (a.solve_tokens + 4096)

    # `--splits evoeval=subtle,creative` edits that benchmark's source block for this run only
    if a.splits:
        name, _, value = a.splits.partition("=")
        spec = a.catalog.get(name.strip())
        if not spec:
            sys.exit("--splits names '" + name.strip() + "', which is not in benchmarks.catalog")
        want = [x.strip() for x in value.split(",") if x.strip()]
        allowed = (spec.get("source") or {}).get("all_splits") or want
        bad = [x for x in want if x not in allowed]
        if bad:
            sys.exit("unknown split(s) for " + name + ": " + ", ".join(bad) +
                     " (have: " + ", ".join(allowed) + ")")
        spec = json.loads(json.dumps(spec))
        spec["source"]["splits"] = want
        a.catalog[name.strip()] = spec

    run_dir = os.path.join(STATE, a.run, "bench")
    os.makedirs(run_dir, exist_ok=True)
    benches = [b.strip() for b in a.benchmarks.split(",") if b.strip()] or \
        [str(b) for b in (cfg.get("benchmarks.use") or [])]
    unknown = [b for b in benches if b not in a.catalog]
    if unknown:
        sys.exit("unknown benchmark(s): " + ", ".join(unknown) + ". Add them to benchmarks.catalog in " +
                 a.config + ", or choose from: " + ", ".join(sorted(a.catalog)))
    a.benchmarks = ",".join(benches)       # so the dashboard shows what is actually being measured
    holder = {"rows": [], "deltas": [], "contam": {}, "plan": []}

    # Which models still need measuring. The base model is measured once and never again - so whichever
    # name it was first measured under ("base", or "base_f16" once a fine-tune has produced the same weights
    # at full precision) is the name it keeps. Without this, converting the base to f16 during the first
    # fine-tune would silently give it a second identity and the base would be benchmarked all over again.
    reg = Registry.load()
    want = [m.strip() for m in a.models.split(",") if m.strip()]
    if not want:
        want = [base_id(run_dir, benches, reg)] + [v["id"] for v in reg.get("versions", [])]
    plan = []
    for m in want:
        for b in benches:
            done = read_json(os.path.join(run_dir, m, b + ".summary.json"))
            if not done and m in ("base", "base_f16"):
                other = "base" if m == "base_f16" else "base_f16"
                done = read_json(os.path.join(run_dir, other, b + ".summary.json"))
                if done and done.get("complete"):
                    log("skip", "the base model was already measured on " + b + " as '" + other + "': " +
                        format(done["pass_rate"], ".1f") + "% pass@1 - it is never re-measured")
            if done and done.get("complete") and not a.force:
                if not any(r["model"] == done["model"] and r["bench"] == b for r in holder["rows"]):
                    holder["rows"].append(done)
                    if done["model"] == m:
                        log("skip", m + " on " + b + " was already measured: " +
                            format(done["pass_rate"], ".1f") + "% pass@1 - not re-running it")
                continue
            plan.append((m, b))
    holder["plan"] = [{"model": m, "bench": b} for m, b in plan]

    if a.report or not plan:
        rows = sorted(holder["rows"], key=lambda r: (r["bench"], r["model"]))
        if not rows:
            log("report", "no measurements yet for run '" + a.run + "'. Run `python benchmark.py` "
                          "(after `python finetune.py` there will be a version to compare).")
            return
        table(rows)
        holder["deltas"] = deltas(rows)
        if not a.report:
            log("done", "every requested model has already been measured; --force re-runs them", "green")
        return

    Log.block(["", Log.paint("  Recursive self-improvement - phase 3: benchmarking", "bold"),
               "  models     " + ", ".join(dict.fromkeys(m for m, _ in plan)),
               "  benchmarks " + ", ".join(benches) + "   (pass@1, temperature " + str(a.temp) + ")",
               "  clean      every problem is probed for memorisation against the base model; "
               "scores are reported with and without the flagged ones", ""])

    backend = Backend(a.ctx, a.slots, a.port, a.backend)
    Stop.on_exit(backend.stop)
    files = backend.ensure_weights(a.model)
    Registry.set_base(a.model, files["model"], files.get("params"))
    reg = Registry.load()
    base_entry = reg.get("base_f16") or reg["base"]
    backend.serve("judge", base_entry)        # the probe always uses the untouched base model

    bench_obj = Bench(a, backend, run_dir)
    if not a.no_dash:
        Dash(make_api(a, bench_obj, holder), "benchmark").start(a.dash_port, not a.no_browser)

    try:
        data = {}
        contam = {}
        for b in benches:
            data[b] = load_bench(b, a)
            log("data", b + ": " + str(len(data[b])) + " problems")
            contam[b] = screen(backend, data[b], b, a)
            holder["contam"][b] = {"flagged": sum(1 for v in contam[b].values() if v.get("memorized")),
                                   "total": len(data[b])}
        for model_id in dict.fromkeys(m for m, _ in plan):
            if Stop.is_set():
                break
            entry = Registry.resolve(model_id) or (reg.get("base_f16") if model_id == "base_f16" else None)
            if not entry or not entry.get("gguf"):
                log("warn", "no weights registered for " + model_id + "; skipping", "yellow")
                continue
            backend.serve("gen", entry)
            log("model", "measuring " + model_id + " (" + os.path.basename(entry["gguf"]) + ")", "bold")
            for m, b in plan:
                if m != model_id or Stop.is_set():
                    continue
                summ = bench_obj.run_model(model_id, b, data[b], contam[b])
                if summ:
                    holder["rows"] = [r for r in holder["rows"]
                                      if not (r["model"] == m and r["bench"] == b)] + [summ]
                    log("result", m + " " + b + ": " + format(summ["pass_rate"], ".1f") + "% pass@1 (" +
                        str(summ["passed"]) + "/" + str(summ["n"]) + ")" +
                        ("  |  unseen only " + format(summ["pass_rate_clean"], ".1f") + "%"
                         if summ["pass_rate_clean"] is not None else ""), "green")
    except Interrupted:
        pass
    except KeyboardInterrupt:
        Stop.set()
    finally:
        rows = sorted(holder["rows"], key=lambda r: (r["bench"], r["model"]))
        table(rows)
        holder["deltas"] = deltas(rows)
        write_json(os.path.join(run_dir, "comparison.json"),
                   {"rows": rows, "deltas": holder["deltas"], "contamination": holder["contam"],
                    "when": time.time(), "config": {k: v for k, v in vars(a).items()}})
        backend.stop()
    if Stop.is_set():
        log("stop", "stopped cleanly - finished problems are saved; re-run to continue", "yellow")
    else:
        log("done", "results in " + run_dir + "/comparison.json", "green")


if __name__ == "__main__":
    main()
