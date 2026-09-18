#!/usr/bin/env python3
"""Programmatic quality checks for the Python the model writes - the part of the loop that does not rely on
another model's opinion. Four layers, cheapest first:

1. extract  - pull the code out of the answer (fenced blocks, preferring the complete one)
2. compile  - real syntax errors, with line and column
3. lint     - ruff when it is available in the project venv (installed automatically, it is a single small
              wheel), plus an AST pass that always runs: undefined names, unused imports, bare except,
              mutable default arguments, shadowed builtins, missing docstring on the public function
4. execute  - run the file in a separate process with a timeout and a scratch working directory, then run
              its doctests and any `test_*` functions it defines

The findings go two ways: a score that ranks the three candidate answers, and a plain-text report that the
model itself summarises before the rewrite step.
"""
import ast, builtins, json, os, re, subprocess, sys, tempfile, textwrap, time, warnings

from common import Venv, log

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S)
BUILTINS = set(dir(builtins))
RUFF_RULES = "E,F,W,B,C4,UP,SIM,RET,ARG,PIE"
# W291-W293 are about trailing whitespace at the end of the extracted block, which is an artifact of pulling
# the code out of a fenced answer rather than anything the model did wrong.
RUFF_IGNORE = "W291,W292,W293"
_RUFF = {"checked": False, "ok": False}

# What each kind of finding costs, and what a well-formed answer earns back. `config.bind_checker` replaces
# these with the "checker" block of config.json, so the scoring can be retuned without touching this file.
PENALTIES = {"error": 34, "warning": 4, "info": 1, "timeout": 25, "import_error": 30,
             "doctest_failure": 8, "test_failure": 10, "examples_fail": 22}
BONUSES = {"has_tests": 4, "has_type_hints": 2}


def bind(cfg):
    """Take the checker's rules and its scoring from the config file."""
    global RUFF_RULES, RUFF_IGNORE
    c = cfg.get("checker") or {}
    RUFF_RULES = str(c.get("ruff_rules") or RUFF_RULES)
    RUFF_IGNORE = str(c.get("ruff_ignore") or RUFF_IGNORE)
    PENALTIES.update({k: v for k, v in (c.get("penalties") or {}).items() if k in PENALTIES})
    BONUSES.update({k: v for k, v in (c.get("bonuses") or {}).items() if k in BONUSES})


# ---------------------------------------------------------------- 1. extract
OPEN_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n")


def code_blocks(text):
    """Every fenced block in the answer, as (language, code). An answer that ran out of tokens in the middle
    of its code block has an opening fence and no closing one; that block is returned too, because losing it
    would score a merely truncated answer as having written no code at all."""
    text = text or ""
    out = [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(text)]
    end = 0
    for m in FENCE.finditer(text):
        end = m.end()
    tail = text[end:]
    m = OPEN_FENCE.search(tail)
    if m and "```" not in tail[m.end():]:
        out.append((m.group(1).lower(), tail[m.end():]))
    return out


def salvage(code):
    """Truncated code rarely parses. Drop whole trailing lines until it does, so the part the model did
    finish can still be linted and run. Gives up once a third of the block is gone."""
    lines = code.rstrip().split("\n")
    floor = max(1, int(len(lines) * 0.67))
    for cut in range(len(lines), floor - 1, -1):
        candidate = "\n".join(lines[:cut]).rstrip()
        if candidate and parses_ok(candidate):
            return candidate, cut < len(lines)
    return code, False


def extract_code(text):
    """The Python the answer is really proposing: the longest block that parses, else the longest block with
    its unfinished tail trimmed off, else the whole answer when it looks like bare code."""
    blocks = [c for lang, c in code_blocks(text) if lang in ("", "py", "python", "python3")]
    if not blocks:
        blocks = [c for _, c in code_blocks(text)]
    parses = [c for c in blocks if parses_ok(c)]
    if parses:
        return max(parses, key=len).strip()
    if blocks:
        fixed, _ = salvage(max(blocks, key=len))
        return fixed.strip()
    stripped = (text or "").strip()
    return stripped if stripped and parses_ok(stripped) else ""


def looks_truncated(text):
    """True when the answer stopped in the middle of a fenced block."""
    return (text or "").count("```") % 2 == 1


def parses_ok(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_examples(text, main):
    """The answer is asked to end with a block of `assert` examples. Those are the model's own claims about
    its code, so running them is the sharpest cheap test there is: they are appended to the solution before
    it is executed, but they are never linted as if they were part of it."""
    out = []
    for lang, block in code_blocks(text):
        if lang not in ("", "py", "python", "python3"):
            continue
        b = block.strip()
        if not b or b == main.strip() or b in main:
            continue
        if not parses_ok(b):
            continue
        if re.search(r"^\s*(assert\b|def test_)", b, re.M) or ">>>" in b:
            out.append(b)
    return "\n\n".join(out)


# ---------------------------------------------------------------- 2. compile
def syntax_errors(code):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # `is` against a literal etc. are reported as findings, not noise
            compile(code, "<answer>", "exec")
        return []
    except SyntaxError as e:
        return [{"tool": "syntax", "rule": type(e).__name__, "line": e.lineno or 0, "col": e.offset or 0,
                 "msg": str(e.msg), "severity": "error"}]
    except ValueError as e:      # e.g. a null byte
        return [{"tool": "syntax", "rule": "ValueError", "line": 0, "col": 0, "msg": str(e), "severity": "error"}]


# ---------------------------------------------------------------- 3a. lint - ruff
def ruff_available(install=True):
    if _RUFF["checked"]:
        return _RUFF["ok"]
    _RUFF["checked"] = True
    try:
        if not Venv.have("ruff"):
            if not install:
                return False
            Venv.need(["ruff"], ["ruff"], note="ruff (Python linter)")
        p = subprocess.run([Venv.python(), "-m", "ruff", "--version"], capture_output=True, text=True, timeout=60)
        _RUFF["ok"] = p.returncode == 0
    except Exception:
        _RUFF["ok"] = False
    if not _RUFF["ok"]:
        log("lint", "ruff unavailable; using the built-in AST checks only")
    return _RUFF["ok"]


def ruff_lint(code, timeout=60):
    if not ruff_available():
        return []
    with tempfile.TemporaryDirectory(prefix="rli_lint_") as d:
        f = os.path.join(d, "answer.py")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            p = subprocess.run([Venv.python(), "-m", "ruff", "check", "--output-format", "json",
                                "--isolated", "--select", RUFF_RULES, "--ignore", RUFF_IGNORE,
                                "--target-version", "py310", f],
                               capture_output=True, text=True, timeout=timeout)
            items = json.loads(p.stdout or "[]")
        except Exception:
            return []
    out = []
    for it in items:
        loc = it.get("location") or {}
        rule = it.get("code") or "RUFF"
        out.append({"tool": "ruff", "rule": rule, "line": loc.get("row", 0), "col": loc.get("column", 0),
                    "msg": it.get("message", ""),
                    "severity": "error" if rule.startswith(("E9", "F8", "F6", "F7")) else "warning"})
    return out


# ---------------------------------------------------------------- 3b. lint - always-on AST pass
class _Scan(ast.NodeVisitor):
    """A small pyflakes: names used but never bound, imports never used, and habits that make code brittle."""

    def __init__(s):
        s.imported, s.used, s.bound, s.problems = {}, set(), set(BUILTINS), []
        s.funcs, s.classes, s.has_main, s.depth = [], [], False, 0

    def add(s, node, rule, msg, severity="warning"):
        s.problems.append({"tool": "ast", "rule": rule, "line": getattr(node, "lineno", 0),
                           "col": getattr(node, "col_offset", 0), "msg": msg, "severity": severity})

    def visit_Import(s, n):
        for a in n.names:
            name = (a.asname or a.name).split(".")[0]
            s.imported[name] = n.lineno
            s.bound.add(name)
        s.generic_visit(n)

    def visit_ImportFrom(s, n):
        for a in n.names:
            if a.name == "*":
                s.add(n, "star-import", "`from " + str(n.module) + " import *` hides what the code depends on")
                continue
            name = a.asname or a.name
            s.imported[name] = n.lineno
            s.bound.add(name)
        s.generic_visit(n)

    def visit_Name(s, n):
        if isinstance(n.ctx, ast.Load):
            s.used.add(n.id)
        else:
            s.bound.add(n.id)
        s.generic_visit(n)

    def visit_Attribute(s, n):
        node = n
        while isinstance(node, ast.Attribute):
            node = node.value
        if isinstance(node, ast.Name):
            s.used.add(node.id)
        s.generic_visit(n)

    def _func(s, n):
        s.bound.add(n.name)
        if s.depth == 0:
            s.funcs.append(n.name)
        for a in list(n.args.args) + list(n.args.kwonlyargs) + list(n.args.posonlyargs):
            s.bound.add(a.arg)
        if n.args.vararg:
            s.bound.add(n.args.vararg.arg)
        if n.args.kwarg:
            s.bound.add(n.args.kwarg.arg)
        for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x]:
            if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                s.add(d, "mutable-default", "mutable default argument in `" + n.name +
                      "`: it is shared between every call", "error")
        if s.depth == 0 and not n.name.startswith("_") and not ast.get_docstring(n):
            s.add(n, "no-docstring", "public function `" + n.name + "` has no docstring", "info")
        s.depth += 1
        s.generic_visit(n)
        s.depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = _func

    def visit_ClassDef(s, n):
        s.bound.add(n.name)
        if s.depth == 0:
            s.classes.append(n.name)
            if not ast.get_docstring(n):
                s.add(n, "no-docstring", "class `" + n.name + "` has no docstring", "info")
        s.depth += 1
        s.generic_visit(n)
        s.depth -= 1

    def visit_ExceptHandler(s, n):
        if n.type is None:
            s.add(n, "bare-except", "bare `except:` also swallows KeyboardInterrupt and SystemExit")
        elif isinstance(n.type, ast.Name) and n.type.id == "Exception" and \
                len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
            s.add(n, "silent-except", "`except Exception: pass` hides every failure")
        s.generic_visit(n)

    def visit_Global(s, n):
        s.add(n, "global", "`global " + ", ".join(n.names) + "` makes the function hard to reason about", "info")
        s.generic_visit(n)

    def visit_Compare(s, n):
        for op, cmp in zip(n.ops, n.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(cmp, ast.Constant) and \
                    isinstance(cmp.value, (int, str, float)) and cmp.value is not None:
                s.add(n, "is-literal", "`is` compares identity, not value: use `==` for literals", "error")
        s.generic_visit(n)

    def visit_If(s, n):
        t = n.test
        if isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__":
            s.has_main = True
        s.generic_visit(n)


def ast_lint(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], {}
    sc = _Scan()
    sc.visit(tree)
    problems = list(sc.problems)
    for name, line in sc.imported.items():
        if name not in sc.used:
            problems.append({"tool": "ast", "rule": "unused-import", "line": line, "col": 0,
                             "msg": "`" + name + "` is imported but never used", "severity": "warning"})
    unknown = sorted(n for n in sc.used - sc.bound if not n.startswith("__"))
    for n in unknown[:12]:
        problems.append({"tool": "ast", "rule": "undefined-name", "line": 0, "col": 0,
                         "msg": "`" + n + "` is used but never defined or imported", "severity": "error"})
    shape = {"functions": sc.funcs, "classes": sc.classes, "has_main": sc.has_main,
             "lines": code.count("\n") + 1,
             "docstring": bool(ast.get_docstring(tree)),
             "has_tests": any(f.startswith("test_") for f in sc.funcs) or ">>>" in code or "assert " in code,
             "has_type_hints": bool(re.search(r"def\s+\w+\([^)]*:\s*\w", code) or "->" in code)}
    return problems, shape


# ---------------------------------------------------------------- 4. execute (separate process, timeout)
RUNNER = r'''
import doctest, io, json, os, runpy, sys, traceback
target = sys.argv[1]
out = {"import_ok": False, "import_error": None, "doctest": None, "tests": [], "stdout": "",
       "examples": None}
buf = io.StringIO()
real = sys.stdout
sys.stdout = buf
try:
    g = runpy.run_path(target, run_name="__answer__")
    out["import_ok"] = True
except BaseException:
    out["import_error"] = traceback.format_exc(limit=6)[-1800:]
    g = {}
sys.stdout = real
out["stdout"] = buf.getvalue()[-1500:]
if out["import_ok"]:
    try:
        import types
        mod = types.ModuleType("answer")
        mod.__dict__.update(g)
        r = doctest.testmod(mod, verbose=False, report=False)
        out["doctest"] = {"attempted": r.attempted, "failed": r.failed}
    except BaseException as e:
        out["doctest"] = {"error": str(e)[:200]}
    for name, fn in list(g.items()):
        if name.startswith("test_") and callable(fn):
            try:
                buf2 = io.StringIO(); sys.stdout = buf2
                fn()
                sys.stdout = real
                out["tests"].append({"name": name, "ok": True})
            except BaseException:
                sys.stdout = real
                out["tests"].append({"name": name, "ok": False,
                                     "error": traceback.format_exc(limit=4)[-700:]})
    if os.path.exists("examples.py"):
        try:
            buf3 = io.StringIO(); sys.stdout = buf3
            exec(compile(open("examples.py", encoding="utf-8").read(), "examples.py", "exec"), dict(g))
            sys.stdout = real
            out["examples"] = {"ok": True}
        except BaseException:
            sys.stdout = real
            out["examples"] = {"ok": False, "error": traceback.format_exc(limit=5)[-900:]}
print("<<<RESULT>>>" + json.dumps(out))
'''


def execute(code, timeout=25, examples=""):
    """Run the answer in a throwaway directory with a hard timeout: import it, run its doctests, run any
    `test_*` functions it defines, and finally run the `assert` examples the answer itself provided.
    Not a security sandbox - it is the model's own code, run with a time limit and no arguments."""
    res = {"ran": False, "timeout": False, "import_ok": False, "import_error": None,
           "doctest": None, "tests": [], "examples": None, "stdout": "", "secs": 0.0}
    if not code.strip():
        return res
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="rli_exec_") as d:
        target = os.path.join(d, "answer.py")
        runner = os.path.join(d, "_runner.py")
        with open(target, "w", encoding="utf-8") as f:
            f.write(code)
        with open(runner, "w", encoding="utf-8") as f:
            f.write(RUNNER)
        if examples.strip():
            with open(os.path.join(d, "examples.py"), "w", encoding="utf-8") as f:
                f.write(examples)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            p = subprocess.run([sys.executable, runner, target], capture_output=True, text=True,
                               timeout=timeout, cwd=d, env=env, stdin=subprocess.DEVNULL,
                               errors="replace")
            res["ran"] = True
            mark = "<<<RESULT>>>"
            if mark in (p.stdout or ""):
                res.update(json.loads(p.stdout.rsplit(mark, 1)[1].strip() or "{}"))
            elif p.returncode != 0:
                res["import_error"] = (p.stderr or "")[-1200:]
        except subprocess.TimeoutExpired:
            res["timeout"] = True
            res["import_error"] = "the program did not finish within " + str(timeout) + "s"
        except Exception as e:
            res["import_error"] = repr(e)[:400]
    res["secs"] = round(time.time() - t0, 2)
    return res


# ---------------------------------------------------------------- the whole check
def check(answer_text, run=True, timeout=25):
    """Everything above, on one answer. Returns findings, a shape summary, an execution result and a score
    in 0-100 that is used only to break ties between candidate answers."""
    code = extract_code(answer_text)
    truncated = looks_truncated(answer_text)
    rep = {"has_code": bool(code), "chars": len(code), "findings": [], "shape": {}, "exec": None,
           "truncated": truncated}
    if truncated:
        rep["findings"].append({"tool": "extract", "rule": "truncated", "line": 0, "col": 0,
                                "severity": "error",
                                "msg": "the answer stopped in the middle of its code block - it ran out of "
                                       "its token budget, so the solution is incomplete"})
    if not code:
        rep["findings"].append({"tool": "extract", "rule": "no-code", "line": 0, "col": 0,
                                "msg": "the answer contains no usable Python code block",
                                "severity": "error"})
        rep["counts"] = {"error": len(rep["findings"]), "warning": 0, "info": 0}
        rep["score"] = 0
        rep["summary"] = "no code found in the answer"
        return rep, code
    examples = extract_examples(answer_text, code)
    rep["examples"] = examples
    syn = syntax_errors(code)
    rep["findings"] += syn
    if not syn:
        problems, shape = ast_lint(code)
        rep["findings"] += problems + ruff_lint(code)
        shape["has_examples"] = bool(examples)
        shape["has_tests"] = shape["has_tests"] or bool(examples)
        rep["shape"] = shape
        if run:
            rep["exec"] = execute(code, timeout=timeout, examples=examples)
    rep["counts"] = {sev: sum(1 for f in rep["findings"] if f["severity"] == sev)
                     for sev in ("error", "warning", "info")}
    rep["score"] = score(rep)
    rep["summary"] = summarize(rep)
    return rep, code


def score(rep):
    """100 = compiles, runs, no findings. Errors cost the most, then a failed run, then warnings."""
    if not rep.get("has_code"):
        return 0
    v = 100
    p, b = PENALTIES, BONUSES
    c = rep.get("counts") or {}
    v -= p["error"] * c.get("error", 0)
    v -= p["warning"] * c.get("warning", 0)
    v -= p["info"] * c.get("info", 0)
    ex = rep.get("exec") or {}
    if ex.get("timeout"):
        v -= p["timeout"]
    elif ex.get("import_error"):
        v -= p["import_error"]
    dt = ex.get("doctest") or {}
    if dt.get("failed"):
        v -= p["doctest_failure"] * min(4, dt["failed"])
    failed_tests = sum(1 for t in ex.get("tests", []) if not t.get("ok"))
    v -= p["test_failure"] * min(4, failed_tests)
    exm = ex.get("examples")
    if exm and not exm.get("ok"):
        v -= p["examples_fail"]   # the answer's own worked examples do not hold: that is a real defect
    if (rep.get("shape") or {}).get("has_tests"):
        v += b["has_tests"]
    if (rep.get("shape") or {}).get("has_type_hints"):
        v += b["has_type_hints"]
    return max(0, min(100, v))


def summarize(rep):
    """One line for the console and the dashboard."""
    if not rep.get("has_code"):
        return "no code"
    c = rep.get("counts") or {}
    ex = rep.get("exec") or {}
    bits = []
    if rep.get("truncated"):
        bits.append("cut off mid-code")
    bits += [str(c.get("error", 0)) + " errors", str(c.get("warning", 0)) + " warnings"]
    if ex.get("timeout"):
        bits.append("timed out")
    elif ex.get("import_error"):
        bits.append("crashed on run")
    elif ex.get("ran"):
        bits.append("runs")
    dt = ex.get("doctest") or {}
    if dt.get("attempted"):
        bits.append("doctests " + str(dt["attempted"] - dt.get("failed", 0)) + "/" + str(dt["attempted"]))
    t = ex.get("tests") or []
    if t:
        bits.append("self-tests " + str(sum(1 for x in t if x.get("ok"))) + "/" + str(len(t)))
    exm = ex.get("examples")
    if exm:
        bits.append("own examples " + ("pass" if exm.get("ok") else "FAIL"))
    return ", ".join(bits)


def report_text(rep, limit=28):
    """The findings as plain text - this is what the model is asked to condense before the rewrite."""
    if not rep.get("has_code"):
        return "The answer contained no Python code block at all."
    lines = []
    for f in sorted(rep["findings"], key=lambda x: {"error": 0, "warning": 1, "info": 2}[x["severity"]])[:limit]:
        where = "line " + str(f["line"]) if f.get("line") else "file"
        lines.append("- [" + f["severity"] + "] " + where + " " + f["rule"] + ": " + f["msg"])
    extra = len(rep["findings"]) - limit
    if extra > 0:
        lines.append("- ... and " + str(extra) + " more findings of the same kinds")
    ex = rep.get("exec") or {}
    if ex.get("timeout"):
        lines.append("- [error] running the file did not finish within the time limit "
                     "(likely an infinite loop or a runaway computation)")
    elif ex.get("import_error"):
        lines.append("- [error] running the file raised:\n" + textwrap.indent(ex["import_error"][-900:], "    "))
    dt = ex.get("doctest") or {}
    if dt.get("failed"):
        lines.append("- [error] " + str(dt["failed"]) + " of " + str(dt.get("attempted", 0)) +
                     " doctests in the answer failed")
    for t in ex.get("tests", []):
        if not t.get("ok"):
            lines.append("- [error] the answer's own " + t["name"] + "() failed:\n" +
                         textwrap.indent((t.get("error") or "")[-600:], "    "))
    exm = ex.get("examples")
    if exm and not exm.get("ok"):
        lines.append("- [error] the worked examples the answer itself gave do not hold - running its own "
                     "asserts against its own code raised:\n" +
                     textwrap.indent((exm.get("error") or "")[-800:], "    "))
    return "\n".join(lines) if lines else "No problems found: the code compiles, lints clean and runs."


if __name__ == "__main__":     # quick manual check: python codecheck.py somefile.py
    src = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else sys.stdin.read()
    r, _ = check(src if src.lstrip().startswith("```") else "```python\n" + src + "\n```")
    print(json.dumps({k: v for k, v in r.items() if k != "findings"}, indent=1))
    print(report_text(r))
