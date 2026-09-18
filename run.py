#!/usr/bin/env python3
"""Phase 1 of the loop: build a corpus of good Python answers using the model itself.

    python run.py

One command. It downloads Ollama if it is missing, pulls the model, starts a batched llama-server sized to
the GPU, opens a live dashboard, and then, for every exercise:

    the model writes a set of exercises  (it is told which ones already exist, so sets do not repeat)
      -> three solvers answer it independently, with different sampling and different strengths
      -> a static checker compiles, lints and actually runs each answer
      -> three judges see all three answers side by side and score them in JSON
      -> the best answer, the judges' criticism and a model-written summary of the checker's findings
         go back to the model, which rewrites the answer
      -> the rewritten answer is checked again and saved as the training pair

Everything is written to disk the moment it is produced, so Ctrl-C is safe and re-running resumes at the
exact sub-step that was interrupted. `--generator latest` answers with the newest fine-tuned model while the
judges keep using the untouched base model.

`config.json` decides the rest: which model, how many solvers, how many judges, what each of them is told,
how much runs in parallel, every budget and every threshold. It is written on the first run, and a flag on
the command line overrides it for one run.
"""
import argparse, concurrent.futures as cf, json, os, random, re, sys, threading, time

import config as conf
import prompts
from codecheck import check, report_text
from common import (Backend, Dash, Interrupted, Log, Registry, STATE, STATS, Stop, T0, Gen,
                    append_jsonl, check_stop, common_state, fmt_t, gpu_status, log, phase, read_json,
                    read_jsonl, vlog, write_json)

# The answers are labelled A, B, C ... - one letter per solver in config.json. Everything below sizes
# itself to however many there are, including the judges' JSON schema and the aggregation.
LETTERS = "ABC"
SLOTS = {"A": 0, "B": 1, "C": 2}

# Which config key backs each flag. A flag the user actually passes wins; anything else comes from the file.
CONFIG_MAP = {
    "model": "model.base", "generator": "model.generator", "judge_model": "model.judge",
    "questions": "corpus.questions_per_round", "rounds": "corpus.rounds",
    "recall_full": "corpus.recall_full", "recall_chars": "corpus.recall_chars",
    "answer_tokens": "corpus.answer_tokens", "judge_tokens": "corpus.judge_tokens",
    "question_tokens": "corpus.question_tokens", "summary_tokens": "corpus.summary_tokens",
    "question_temp": "corpus.question_temp", "refine_temp": "corpus.refine_temp",
    "judge_weight": "corpus.judge_weight",
    "exec_timeout": "corpus.exec_timeout", "ctx": "corpus.ctx",
    "workers": "parallel.workers", "slots": "parallel.slots", "backend": "parallel.backend",
    "port": "server.port_corpus", "dash_port": "ui.ports.corpus", "report_every": "ui.report_every",
}


def set_slots(n):
    global LETTERS, SLOTS
    LETTERS = "".join(prompts.slot_letters(n))
    SLOTS = {c: i for i, c in enumerate(LETTERS)}


def best_answer(rec):
    """The winning answer of a record. Tolerant of a record written when the config had a different number
    of solvers, so an old run can still be reported on after the file has been edited."""
    answers = rec.get("answers") or []
    if not answers:
        return None
    i = SLOTS.get((rec.get("verdict") or {}).get("best"), 0)
    return answers[i] if i < len(answers) else answers[0]


# ---------------------------------------------------------------- tolerant JSON parsing
def loose_json(text):
    """Small models wrap JSON in prose, fences and trailing commas. Recover the object anyway, or None."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*\r?\n(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    starts = [i for i, c in enumerate(t) if c in "[{"]
    for i in starts[:4]:
        opener = t[i]
        closer = "]" if opener == "[" else "}"
        depth, instr, esc, end = 0, False, False, -1
        for j in range(i, len(t)):
            c = t[j]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
                continue
            if c == '"':
                instr = True
            elif c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        chunk = t[i:end + 1] if end > 0 else t[i:]
        for candidate in (chunk, re.sub(r",(\s*[}\]])", r"\1", chunk),
                          re.sub(r",(\s*[}\]])", r"\1", chunk) + (closer * 3)):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None


def num(x, lo=1, hi=10, default=5):
    try:
        v = float(x)
    except Exception:
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- run state on disk
class Run:
    """All state for one named run. Every write is atomic and immediate: there is no in-memory-only state
    that a Ctrl-C could lose."""

    def __init__(s, args):
        s.a = args
        s.dir = os.path.join(STATE, args.run)
        s.rounds_dir = os.path.join(s.dir, "rounds")
        s.corpus_dir = os.path.join(s.dir, "corpus")
        os.makedirs(s.rounds_dir, exist_ok=True)
        os.makedirs(s.corpus_dir, exist_ok=True)
        s.meta_path = os.path.join(s.dir, "meta.json")
        s.meta = read_json(s.meta_path, {"created": time.time(), "model": args.model, "rounds": []})
        s.backend = None
        s.active = {}          # qid -> what that worker is doing right now (for the dashboard)
        s.done_now = 0
        s.t_start = time.time()
        s.lock = threading.Lock()

    def save_meta(s):
        write_json(s.meta_path, s.meta)

    def round_dir(s, n):
        d = os.path.join(s.rounds_dir, "round_" + str(n).zfill(3))
        os.makedirs(d, exist_ok=True)
        return d

    def round_nums(s):
        out = []
        for name in os.listdir(s.rounds_dir):
            m = re.match(r"round_(\d+)$", name)
            if m and os.path.exists(os.path.join(s.rounds_dir, name, "questions.json")):
                out.append(int(m.group(1)))
        return sorted(out)

    def questions(s, n):
        return read_json(os.path.join(s.round_dir(n), "questions.json"), []) or []

    def rec_path(s, n, qid):
        return os.path.join(s.round_dir(n), "q_" + qid + ".json")

    def record(s, n, qid):
        return read_json(s.rec_path(n, qid), None)

    def save_record(s, n, rec):
        write_json(s.rec_path(n, rec["qid"]), rec)

    def pending(s, n):
        return [q for q in s.questions(n) if not (s.record(n, q["qid"]) or {}).get("complete")]

    def corpus_path(s, n):
        return os.path.join(s.corpus_dir, "round_" + str(n).zfill(3) + ".jsonl")

    def stats(s):
        """Everything the progress table and the dashboard header need."""
        rounds = []
        for n in s.round_nums():
            qs = s.questions(n)
            recs = [s.record(n, q["qid"]) for q in qs]
            done = [r for r in recs if r and r.get("complete")]
            clean = [r for r in done if r.get("clean")]
            jsc = [r["verdict"]["best_score"] for r in done if r.get("verdict")]
            b4 = [best_answer(r)["check"]["score"]
                  for r in done if r.get("verdict") and best_answer(r)]
            af = [r["final"]["check"]["score"] for r in done if r.get("final")]
            rounds.append({
                "n": n, "total": len(qs), "done": len(done), "clean": len(clean),
                "generator": (read_json(os.path.join(s.round_dir(n), "info.json"), {}) or {}).get("generator", "?"),
                "judge_score": round(sum(jsc) / len(jsc), 2) if jsc else None,
                "check_before": round(sum(b4) / len(b4), 1) if b4 else None,
                "check_after": round(sum(af) / len(af), 1) if af else None,
                "gain": round(sum(af) / len(af) - sum(b4) / len(b4), 1) if af and b4 else None,
            })
        return rounds


# ---------------------------------------------------------------- the model calls
class Loop:
    def __init__(s, r, backend):
        s.r, s.a, s.backend = r, r.a, backend

    def call(s, role, system, user, budget, temp, sampler, tag, kind, meta=None):
        g = Gen(tag=tag, kind=kind, max_tokens=budget, model=s.backend.entries[role]["id"], meta=meta or {})
        t0 = time.time()
        try:
            text, info = s.backend.chat(role, system, user, budget, temp, gen=g, sampler=sampler)
            g.done()
        except Interrupted:
            g.done("interrupted")
            raise
        except Exception as e:
            g.done(repr(e)[:200])
            raise
        info["secs"] = round(time.time() - t0, 1)
        info["tps"] = round(info["out"] / max(0.1, info["secs"]), 1)
        return text, info

    def previous_text(s, questions, budget):
        """The previous exercises written out in full, newest first, until the character budget runs out."""
        out, used = [], 0
        for q in reversed(questions or []):
            block = ("### " + str(q.get("title", "")) + "  (" + str(q.get("difficulty", "")) + ", " +
                     str(q.get("topic", "")) + ")\n" + str(q.get("question", "")).strip())
            if out and used + len(block) > budget:
                break
            out.append(block)
            used += len(block)
        return "\n\n".join(reversed(out)) if out else "(none yet)"

    # ------------------------------------------------ question set
    def avoid_text(s, prev):
        """The previous questionnaire as the prompt sees it. It goes back into the context in full, not as a
        list of names: the model can only avoid re-asking something if it can see what was actually asked.
        The most recent exercises are sent whole and anything older degrades to its title, so the prompt
        stays bounded - `recall_full` and `recall_chars` in the config decide where that line is."""
        a = s.a
        if not prev:
            return prompts.AVOID_NONE
        return prompts.fill(
            prompts.AVOID_SOME,
            exercises=s.previous_text(prev[-a.recall_full:], a.recall_chars),
            titles="\n".join("- " + str(q.get("title", "")) for q in prev[:-a.recall_full]) or "(none)")

    def question_user(s, need, prev):
        a = s.a
        easy = max(1, round(need * a.easy_share))
        hard = max(1, round(need * a.hard_share))
        short = round(need * a.short_share)
        return prompts.fill(prompts.QUESTION_USER, n=need, easy=easy, hard=hard,
                            medium=max(0, need - easy - hard), short=short, long=need - short,
                            avoid=s.avoid_text(prev))

    def make_questions(s, n_round, count):
        a = s.a
        prev = []
        for m in s.r.round_nums():
            prev += s.r.questions(m)
        system = prompts.fill(prompts.QUESTION_SYSTEM, budget=a.answer_tokens)
        out, tries, barren = [], 0, 0
        while len(out) < count and tries < 6 and not Stop.is_set():
            tries += 1
            phase("questions", "round " + str(n_round) + ": asking for " +
                  str(count - len(out)) + " more exercises (attempt " + str(tries) + ")")
            need, before = count - len(out), len(out)
            u = s.question_user(need, prev + out)
            text, info = s.call("gen", system, u, a.question_tokens, a.question_temp,
                                {"top_p": 0.95, "top_k": 60}, "question set " + str(n_round), "questions")
            items = loose_json(text)
            if isinstance(items, dict):
                items = items.get("exercises") or items.get("questions") or [items]
            if not isinstance(items, list):
                log("warn", "the question set did not parse as JSON; retrying", "yellow")
                continue
            have = {q["title"] for q in out} | {str(q.get("title", "")) for q in prev}
            for it in items:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title") or "").strip().lower().replace(" ", "-")[:60]
                body = str(it.get("question") or "").strip()
                if not title or len(body) < 40 or title in have:
                    continue
                have.add(title)
                checks = it.get("checks") or []
                if isinstance(checks, str):
                    checks = [checks]
                out.append({"qid": "r" + str(n_round).zfill(3) + "q" + str(len(out) + 1).zfill(3),
                            "title": title, "difficulty": str(it.get("difficulty", "medium")),
                            "size": str(it.get("size", "short")), "topic": str(it.get("topic", "general")),
                            "question": body, "checks": [str(c) for c in checks][:6]})
                if len(out) >= count:
                    break
            log("questions", "round " + str(n_round) + ": " + str(len(out)) + "/" + str(count) +
                " exercises after attempt " + str(tries))
            if len(out) == before:
                barren += 1
                if barren >= 2:
                    log("warn", "the model only returns exercises that already exist; stopping this set at " +
                        str(len(out)), "yellow")
                    break
            else:
                barren = 0
        return out

    # ------------------------------------------------ one exercise, end to end
    def solve(s, n_round, q):
        a = s.a
        rec = s.r.record(n_round, q["qid"]) or {
            "qid": q["qid"], "round": n_round, "question": q, "t_start": time.time(),
            "generator": s.backend.entries["gen"]["id"], "judge_model": s.backend.entries["judge"]["id"],
            "answers": [], "judges": [], "verdict": None, "lint_summary": None, "final": None,
            "complete": False, "clean": False}
        save = lambda: s.r.save_record(n_round, rec)
        where = lambda st: s.r.active.__setitem__(q["qid"], st)
        checks = prompts.fmt_checks(q.get("checks"))
        budget = a.answer_tokens
        solvers, judges = a.solvers, a.judges

        # --- 1. one independent answer per configured solver
        if len(rec["answers"]) < len(solvers):
            where("writing " + str(len(solvers)) + " answers")
            todo = [i for i in range(len(solvers)) if i >= len(rec["answers"])]
            results = {}
            with cf.ThreadPoolExecutor(max_workers=len(todo)) as ex:
                futs = {}
                for i in todo:
                    sv = solvers[i]
                    p = str(sv["name"])
                    futs[ex.submit(s.call, "gen", prompts.solver_system(sv.get("prompt", ""), budget),
                                   prompts.fill(prompts.SOLVER_USER, question=q["question"], checks=checks),
                                   budget, sv.get("temperature", 0.5),
                                   {"top_k": sv.get("top_k", 40), "top_p": sv.get("top_p", 0.95),
                                    "min_p": sv.get("min_p", 0.0)},
                                   q["qid"] + " " + LETTERS[i] + ":" + p, "answer",
                                   {"qid": q["qid"], "slot": LETTERS[i], "persona": p})] = i
                for f in cf.as_completed(futs):
                    results[futs[f]] = f.result()
            for i in sorted(results):
                text, info = results[i]
                rep, code = check(text, run=not a.no_exec, timeout=a.exec_timeout)
                rec["answers"].append({"slot": LETTERS[i], "persona": str(solvers[i]["name"]), "text": text,
                                       "code": code, "check": rep, "info": info})
                vlog("answer", q["qid"] + " " + LETTERS[i] + " (" + str(solvers[i]["name"]) + "): " +
                     rep["summary"])
            rec["answers"].sort(key=lambda x: SLOTS.get(x["slot"], 99))
            save()
        check_stop()

        # --- 2. every judge, each seeing every answer in a different order
        names = [str(j["name"]) for j in judges]
        if len(rec["judges"]) < len(judges):
            where(str(len(judges)) + " judges scoring")
            done_p = {j["persona"] for j in rec["judges"]}
            todo = [i for i, p in enumerate(names) if p not in done_p]
            got = []
            with cf.ThreadPoolExecutor(max_workers=len(todo)) as ex:
                futs = {ex.submit(s.judge, q, rec, judges[i], checks): i for i in todo}
                for f in cf.as_completed(futs):
                    got.append(f.result())
            rec["judges"] += got
            rec["judges"].sort(key=lambda j: names.index(j["persona"]) if j["persona"] in names else 99)
            save()
        check_stop()

        # --- 3. aggregate the three verdicts
        if not rec["verdict"]:
            rec["verdict"] = s.aggregate(rec)
            save()
            vlog("verdict", q["qid"] + " best=" + rec["verdict"]["best"] + " score=" +
                 str(rec["verdict"]["best_score"]))
        best = best_answer(rec)

        # --- 4. the model condenses the checker's findings
        if rec["lint_summary"] is None:
            where("summarising the checker findings")
            raw = report_text(best["check"])
            if best["check"]["score"] >= 98 and not raw.startswith("-"):
                rec["lint_summary"] = "NOTHING TO FIX"
            else:
                text, info = s.call("gen", prompts.LINT_SUMMARY_SYSTEM,
                                    prompts.fill(prompts.LINT_SUMMARY_USER, code=best["code"],
                                                 report=raw[:8000]),
                                    a.summary_tokens, 0.2, {"top_k": 20, "top_p": 0.9},
                                    q["qid"] + " checker summary", "summary", {"qid": q["qid"]})
                rec["lint_summary"] = text.strip() or "NOTHING TO FIX"
                rec["lint_info"] = info
            save()
        check_stop()

        # --- 5. the rewrite
        if not rec["final"]:
            where("rewriting the best answer")
            text, info = s.call("gen", prompts.fill(prompts.REFINE_SYSTEM, budget=budget),
                                prompts.fill(prompts.REFINE_USER, question=q["question"], checks=checks,
                                             answer=best["text"], count=len(solvers),
                                             critique=s.critique_text(rec),
                                             lint=rec["lint_summary"]),
                                budget, a.refine_temp, {"top_k": 20, "top_p": 0.9, "min_p": 0.05},
                                q["qid"] + " rewrite", "refine", {"qid": q["qid"]})
            rep, code = check(text, run=not a.no_exec, timeout=a.exec_timeout)
            rec["final"] = {"text": text, "code": code, "check": rep, "info": info}
            save()
        check_stop()

        # --- 6. save the answer and the training pair
        fin = rec["final"]
        # Every rewrite is saved. Producing an answer for each exercise is the job; deciding which
        # exercises deserve to have one is not. An answer that came out badly is still what this model
        # does with that question, and dropping it would quietly hide that. `clean` is a label for the
        # reports only - it never keeps a row out - and the checker score travels with the row so that
        # a later step can weigh it.
        rec["clean"] = bool(fin["code"]) and fin["check"]["counts"].get("error", 0) == 0
        rec["gain"] = fin["check"]["score"] - best["check"]["score"]
        rec["complete"] = True
        rec["t_end"] = time.time()
        save()
        append_jsonl(s.r.corpus_path(n_round), {
            "qid": q["qid"], "round": n_round, "title": q["title"], "topic": q["topic"],
            "difficulty": q["difficulty"], "size": q["size"], "question": q["question"],
            "answer": fin["text"], "check_score": fin["check"]["score"], "clean": rec["clean"],
            "judge_score": rec["verdict"]["best_score"], "generator": rec["generator"]})
        s.r.active.pop(q["qid"], None)
        return rec

    def judge(s, q, rec, judge_cfg, checks):
        """One judge. The answers are shown in a different order to each judge, so position cannot decide
        the winner; the mapping is stored so the verdict can be translated back."""
        persona = str(judge_cfg["name"])
        n = len(rec["answers"])
        order = list(range(n))
        random.Random(hash((q["qid"], persona)) & 0xffffffff).shuffle(order)
        user = prompts.judge_user(q["question"], checks, [rec["answers"][order[i]]["text"] for i in range(n)])
        parsed, raw, info = None, "", {}
        for attempt in range(2):
            extra = "" if attempt == 0 else ("\n\nYour previous reply was not valid JSON. Reply with the "
                                             "single JSON object only, starting with { and ending with }.")
            raw, info = s.call("judge", prompts.judge_system(judge_cfg.get("prompt", ""), n), user + extra,
                               s.a.judge_tokens, judge_cfg.get("temperature", 0.4),
                               {"top_k": judge_cfg.get("top_k", 40), "top_p": judge_cfg.get("top_p", 0.92)},
                               q["qid"] + " judge:" + persona, "judge",
                               {"qid": q["qid"], "persona": persona})
            parsed = loose_json(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("answers"), dict):
                break
            parsed = None
        out = {"persona": persona, "order": order, "raw": raw, "info": info, "ok": parsed is not None,
               "scores": {}, "best": None, "why_best": "", "fix_list": [], "notes": {}}
        if parsed:
            for shown_slot, body in parsed["answers"].items():
                key = str(shown_slot).strip().upper()[:1]
                if SLOTS.get(key, 99) >= n or not isinstance(body, dict):
                    continue
                real = LETTERS[order[SLOTS[key]]]
                out["scores"][real] = {ax: num(body.get(ax)) for ax in
                                       ("correctness", "robustness", "efficiency", "style", "overall")}
                out["notes"][real] = {k: [str(x)[:400] for x in (body.get(k) or [])][:5]
                                      for k in ("good", "bad", "doubts")}
            b = str(parsed.get("best", "")).strip().upper()[:1]
            if SLOTS.get(b, 99) < n:
                out["best"] = LETTERS[order[SLOTS[b]]]
            out["why_best"] = str(parsed.get("why_best", ""))[:600]
            out["fix_list"] = [str(x)[:400] for x in (parsed.get("fix_list") or [])][:5]
        else:
            log("warn", q["qid"] + ": judge " + persona + " produced no usable JSON twice; "
                "it is ignored for this exercise", "yellow")
        return out

    def aggregate(s, rec):
        """Pick the winner from two independent sources of evidence: what the three judges say, and what
        running the code actually proves. The judges read for intent and design, which a checker cannot do;
        the checker knows whether it compiles, lints and passes its own asserts, which a 0.8B judge often
        gets wrong from reading alone. So they are combined on one 0-10 scale, `--judge-weight` of the
        judges' verdict and the rest from the checker, rather than letting either decide by itself.

        Two hard rules sit on top of the blend: code that does not compile or does not run cannot win over
        code that does, whatever the judges thought of it; and if every judge fails to return usable JSON,
        the checker decides alone."""
        W = dict(s.a.axis_weights)
        letters = "".join(x["slot"] for x in rec["answers"]) or LETTERS
        per = {}
        for slot in letters:
            vals, overalls = [], []
            for j in rec["judges"]:
                sc = j["scores"].get(slot)
                if sc:
                    vals.append(sum(W[k] * sc.get(k, 5) for k in W))
                    overalls.append(sc["overall"])
            if vals:
                per[slot] = {"axes": round(sum(vals) / len(vals), 2),
                             "overall": round(sum(overalls) / len(overalls), 2),
                             "n": len(vals)}
            else:
                per[slot] = {"axes": 0.0, "overall": 0.0, "n": 0}
        votes = {slot: sum(1 for j in rec["judges"] if j["best"] == slot) for slot in letters}
        checker = {a["slot"]: a["check"]["score"] for a in rec["answers"]}
        runs = {a["slot"]: s.runs_ok(a["check"]) for a in rec["answers"]}
        jw = max(0.0, min(1.0, s.a.judge_weight))
        have_judges = any(p["n"] for p in per.values())

        def judged(sl):
            """The judges' view of an answer on a 0-10 scale: the overall mark, its axis mean, and a bonus
            for each judge that named it the best of the set."""
            return 0.6 * per[sl]["overall"] + 0.4 * per[sl]["axes"] + 0.5 * votes[sl]

        def blended(sl):
            if not have_judges:
                return checker.get(sl, 0) / 10.0
            return jw * judged(sl) + (1 - jw) * (checker.get(sl, 0) / 10.0)

        # working code first, then the blended score
        rank = lambda sl: (1 if runs.get(sl) else 0, round(blended(sl), 4))
        best = max(letters, key=rank)
        blend = {sl: round(blended(sl), 2) for sl in letters}
        return {"per_answer": per, "votes": votes, "checker": checker, "runs": runs, "blend": blend,
                "judge_weight": jw, "best": best,
                "best_score": per[best]["overall"] or round(checker.get(best, 0) / 10, 2),
                "best_blend": blend[best],
                "fix_list": s.merge_fixes(rec, best),
                "judges_ok": sum(1 for j in rec["judges"] if j["ok"])}

    @staticmethod
    def runs_ok(check):
        """Did this answer survive the programmatic checks: real code, no errors, and it executed."""
        if not check.get("has_code") or check.get("counts", {}).get("error"):
            return False
        ex = check.get("exec") or {}
        if not ex:
            return True                      # execution was switched off with --no-exec
        return bool(ex.get("ran")) and not ex.get("timeout") and not ex.get("import_error")

    def merge_fixes(s, rec, best):
        """The fix lists of the judges that picked this answer come first, then the others', de-duplicated."""
        out, seen = [], set()
        for j in sorted(rec["judges"], key=lambda x: 0 if x["best"] == best else 1):
            for f in j["fix_list"] + (j["notes"].get(best, {}).get("bad") or []):
                key = re.sub(r"[^a-z0-9 ]", "", f.lower())[:60]
                if key and key not in seen:
                    seen.add(key)
                    out.append(f)
        return out[:8]

    def critique_text(s, rec):
        """What the judges said about the winning answer, as the rewrite prompt sees it."""
        best = rec["verdict"]["best"]
        lines = []
        for j in rec["judges"]:
            sc = j["scores"].get(best)
            head = "Reviewer (" + j["persona"] + ")"
            if sc:
                head += " scored it correctness " + str(sc["correctness"]) + "/10, robustness " + \
                        str(sc["robustness"]) + "/10, efficiency " + str(sc["efficiency"]) + "/10, style " + \
                        str(sc["style"]) + "/10, overall " + str(sc["overall"]) + "/10."
            lines.append(head)
            notes = j["notes"].get(best, {})
            for k, label in (("good", "keeps"), ("bad", "must fix"), ("doubts", "is unsure about")):
                for item in notes.get(k, []):
                    lines.append("  - " + label + ": " + item)
            lines.append("")
        fixes = rec["verdict"]["fix_list"]
        if fixes:
            lines.append("The combined fix list, most important first:")
            lines += [str(i + 1) + ". " + f for i, f in enumerate(fixes)]
        return "\n".join(lines).strip() or "(the reviewers produced no usable feedback)"


# ---------------------------------------------------------------- progress reporting
def report(r, final=False):
    a = r.a
    rows = r.stats()
    if not rows:
        return
    w = Log.paint
    head = ("  round  gen        done         clean   judge  checker  after  gain")
    lines = ["", w(head, "bold")]
    for row in rows:
        gain = row["gain"]
        gtxt = ("+" if gain and gain > 0 else "") + (str(gain) if gain is not None else "-")
        lines.append("  " + str(row["n"]).rjust(5) + "  " + str(row["generator"])[:9].ljust(9) + "  " +
                     (str(row["done"]) + "/" + str(row["total"])).rjust(8) + "  " +
                     str(row["clean"]).rjust(9) + "  " +
                     (str(row["judge_score"]) if row["judge_score"] is not None else "-").rjust(6) + "  " +
                     (str(row["check_before"]) if row["check_before"] is not None else "-").rjust(7) + "  " +
                     (str(row["check_after"]) if row["check_after"] is not None else "-").rjust(6) + "  " +
                     w(gtxt.rjust(5), "green" if gain and gain > 0 else "dim"))
    g = gpu_status()
    el = time.time() - T0
    tps = STATS["out"] / max(1.0, el)
    line = ("  " + fmt_t(el) + " elapsed | " + str(STATS["calls"]) + " model calls | " +
            format(STATS["out"] / 1000, ".1f") + "k tokens out (" + format(tps, ".0f") + " tok/s) | " +
            (r.backend.status() if r.backend else ""))
    if g:
        line += " | GPU " + (str(g["util"]) + "% " if g["util"] >= 0 else "") + \
                str(g["used"]) + "/" + str(g["total"]) + " MB"
    lines += [w(line, "dim"), ""]
    Log.block(lines)


# ---------------------------------------------------------------- dashboard API
def make_api(r):
    def api(what, q):
        if what == "state":
            rows = r.stats()
            active = dict(r.active)
            return common_state(r.backend, {
                "kind": "corpus", "run": r.a.run, "rounds": rows, "active": active,
                "models": {"generator": r.backend.entries["gen"]["id"] if r.backend else None,
                           "judge": r.backend.entries["judge"]["id"] if r.backend else None,
                           "base": (Registry.load().get("base") or {}).get("name")},
                "registry": Registry.load(),
                "config": {k: v for k, v in vars(r.a).items() if not k.startswith("_")}})
        if what == "questions":
            n = int(q.get("round") or (r.round_nums() or [1])[-1])
            out = []
            for item in r.questions(n):
                rec = r.record(n, item["qid"]) or {}
                out.append({"qid": item["qid"], "title": item["title"], "topic": item["topic"],
                            "difficulty": item["difficulty"], "size": item["size"],
                            "complete": bool(rec.get("complete")), "clean": bool(rec.get("clean")),
                            "best": (rec.get("verdict") or {}).get("best"),
                            "judge_score": (rec.get("verdict") or {}).get("best_score"),
                            "before": ((best_answer(rec) or {}).get("check", {}).get("score")
                                       if rec.get("verdict") else None),
                            "after": (rec.get("final") or {}).get("check", {}).get("score"),
                            "active": r.active.get(item["qid"])})
            return {"round": n, "rounds": r.round_nums(), "questions": out}
        if what == "question":
            n, qid = int(q.get("round") or 1), str(q.get("qid") or "")
            rec = r.record(n, qid)
            return rec
        return None
    return api


# ---------------------------------------------------------------- what to do this run
def decide_round(r, args):
    """Resume an unfinished round, or start a new one - asking the user when it is ambiguous."""
    nums = r.round_nums()
    if nums:
        last = nums[-1]
        left = r.pending(last)
        if left and not args.new:
            log("plan", "round " + str(last) + " has " + str(len(left)) + " of " +
                str(len(r.questions(last))) + " exercises left; resuming it", "bold")
            return last, False
        if left and args.new:
            log("plan", "round " + str(last) + " still has " + str(len(left)) +
                " exercises left - finish it first (re-run without --new), or pass --force-new", "yellow")
            if not args.force_new:
                return last, False
    if args.new or args.force_new:
        return (nums[-1] + 1 if nums else 1), True
    if not nums:
        return 1, True
    if args.no_new:
        return None, False
    if not sys.stdin.isatty():
        log("plan", "every round is finished and there is no terminal to ask; pass --new for another set")
        return None, False
    print("")
    print("  Round " + str(nums[-1]) + " is finished (" + str(len(r.questions(nums[-1]))) +
          " exercises, " + str(len(read_jsonl(r.corpus_path(nums[-1])))) + " answers in the corpus).")
    try:
        ans = input("  Generate a new set of " + str(r.a.questions) + " exercises? [Y/n] ").strip().lower()
    except EOFError:
        ans = "n"
    if ans in ("", "y", "yes"):
        return nums[-1] + 1, True
    return None, False


# ---------------------------------------------------------------- main
def main():
    O = conf.opt()      # a flag with no default: when it is not passed, config.json decides
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    conf.add_args(ap)
    ap.add_argument("--model", default=O, help="base model, pulled automatically on the first run")
    ap.add_argument("--run", default="default", help="run name; state/<run>/ holds everything")
    ap.add_argument("--generator", default=O,
                    help="who writes the questions and the answers: base | latest | v2 | path.gguf")
    ap.add_argument("--judge-model", default=O, help="who judges (kept on the base model by design)")
    ap.add_argument("--questions", type=int, default=O, help="exercises per round")
    ap.add_argument("--recall-full", type=int, default=O,
                    help="how many of the most recent exercises are shown in full when writing the next set")
    ap.add_argument("--recall-chars", type=int, default=O,
                    help="character budget for those exercises; anything older degrades to its title")
    ap.add_argument("--new", action="store_true", help="start a new round without asking")
    ap.add_argument("--force-new", action="store_true", help="start a new round even if one is unfinished")
    ap.add_argument("--no-new", action="store_true", help="only finish what is pending, never ask")
    ap.add_argument("--rounds", type=int, default=O, help="how many rounds to run in one go")
    ap.add_argument("--workers", type=int, default=O, help="exercises in flight; 0 = fit to the GPU slots")
    ap.add_argument("--slots", type=int, default=O, help="llama-server parallel slots; 0 = fit to free VRAM")
    ap.add_argument("--backend", default=O, choices=["auto", "llama", "ollama"])
    ap.add_argument("--ctx", type=int, default=O, help="context per slot; 0 = computed from the budgets")
    ap.add_argument("--answer-tokens", type=int, default=O,
                    help="token ceiling for one answer; the spec's limit for an exercise")
    ap.add_argument("--judge-tokens", type=int, default=O)
    ap.add_argument("--question-tokens", type=int, default=O)
    ap.add_argument("--summary-tokens", type=int, default=O)
    ap.add_argument("--question-temp", type=float, default=O)
    ap.add_argument("--refine-temp", type=float, default=O)
    ap.add_argument("--judge-weight", type=float, default=O,
                    help="how much the judges decide the winner, the rest coming from the programmatic "
                         "checker (0 = the checker alone, 1 = the judges alone)")
    ap.add_argument("--exec-timeout", type=int, default=O, help="seconds an answer may run for")
    ap.add_argument("--no-exec", action="store_true", help="lint only, never run the generated code")
    ap.add_argument("--port", type=int, default=O)
    ap.add_argument("--report-every", type=int, default=O)
    ap.add_argument("--dash-port", type=int, default=O)
    ap.add_argument("--no-dash", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--dashboard", action="store_true", help="serve the dashboard for saved state and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    Log.setup(not a.no_color, a.verbose)
    if conf.handle_meta(a):
        return
    cfg = conf.load(a.config or None)
    conf.apply(a, cfg, CONFIG_MAP)
    a.config = cfg.path

    # The solvers and the judges are whatever config.json lists: their number, their sampling and their
    # prompt all come from there, and everything below sizes itself to it.
    a.solvers, a.judges = cfg.solvers, cfg.judges
    set_slots(len(a.solvers))
    a.axis_weights = cfg.get("corpus.axis_weights")
    a.easy_share = cfg.get("corpus.difficulty_mix.easy", 0.3)
    a.hard_share = cfg.get("corpus.difficulty_mix.hard", 0.25)
    a.short_share = cfg.get("corpus.short_share", 0.6)
    if not cfg.get("corpus.run_code", True):
        a.no_exec = True
    if not cfg.get("ui.dashboard", True):
        a.no_dash = True
    if not cfg.get("ui.open_browser", True):
        a.no_browser = True
    Stop.install()

    r = Run(a)
    # The judge is the expensive role: it reads every answer in full (never truncated, never summarised)
    # and then writes its verdict. The solvers only ever hold one answer, so they are served from a second,
    # much smaller context - which is what keeps the parallel slots affordable.
    a.gen_ctx = a.ctx or (a.answer_tokens + 4000)
    a.judge_ctx = a.ctx or (len(a.solvers) * a.answer_tokens + a.judge_tokens + 4000)
    a.ctx = a.gen_ctx

    if a.dashboard:
        Dash(make_api(r), "corpus").start(a.dash_port, not a.no_browser)
        log("dash", "view only: serving saved state, Ctrl-C to stop", "bold")
        while not Stop.sleep(1):
            pass
        return

    banner(a, r)
    n_round, is_new = decide_round(r, a)
    if n_round is None:
        log("done", "nothing to do: every round is finished. `--new` starts another set, "
                    "`python finetune.py` trains on what is there.", "green")
        return

    # ---- backend: the generator (maybe fine-tuned) and the judges (always the base model)
    phase("startup", "preparing the model servers")
    backend = Backend(a.gen_ctx, a.slots, a.port, a.backend)
    Stop.on_exit(backend.stop)
    r.backend = backend
    files = backend.ensure_weights(a.model)
    base = Registry.set_base(a.model, files["model"], files.get("params"))
    gen_entry = Registry.resolve(a.generator)
    if gen_entry is None:
        sys.exit("unknown --generator " + a.generator + " (have: base, " +
                 ", ".join(v["id"] for v in Registry.load()["versions"]) + ")")
    judge_entry = Registry.resolve(a.judge_model) or base
    backend.serve("gen", gen_entry, a.gen_ctx)
    backend.serve("judge", judge_entry, a.judge_ctx)
    log("ready", "generator = " + gen_entry["id"] + " | judges = " + judge_entry["id"] + " | " +
        backend.status(), "bold")

    if not a.no_dash:
        Dash(make_api(r), "corpus").start(a.dash_port, not a.no_browser)

    workers = a.workers or max(2, int(backend.total_slots() / 2))
    loop = Loop(r, backend)

    stop_rep = threading.Event()

    def reporter():
        while not stop_rep.wait(a.report_every):
            report(r)
    threading.Thread(target=reporter, daemon=True).start()

    try:
        for k in range(a.rounds):
            if Stop.is_set():
                break
            if k > 0:
                n_round, is_new = (r.round_nums()[-1] + 1), True
            do_round(r, loop, n_round, is_new, workers, gen_entry, judge_entry)
            is_new = True
    except Interrupted:
        pass
    except KeyboardInterrupt:
        Stop.set()
    finally:
        stop_rep.set()
        report(r, final=True)
        backend.stop()

    if Stop.is_set():
        log("stop", "stopped cleanly - every finished step is on disk. Re-run the same command to resume.", "yellow")
    else:
        total = sum(len(read_jsonl(r.corpus_path(n))) for n in r.round_nums())
        log("done", "corpus now holds " + str(total) + " answers across " +
            str(len(r.round_nums())) + " rounds", "green")
        log("next", "train on it with:  python finetune.py        then measure:  python benchmark.py", "bold")


def do_round(r, loop, n_round, is_new, workers, gen_entry, judge_entry):
    if is_new and not r.questions(n_round):
        qs = loop.make_questions(n_round, r.a.questions)
        if not qs:
            log("warn", "the model produced no usable exercises; stopping", "yellow")
            return
        write_json(os.path.join(r.round_dir(n_round), "questions.json"), qs)
        write_json(os.path.join(r.round_dir(n_round), "info.json"),
                   {"generator": gen_entry["id"], "judge": judge_entry["id"], "created": time.time(),
                    "count": len(qs)})
        r.meta.setdefault("rounds", []).append({"n": n_round, "generator": gen_entry["id"],
                                                "count": len(qs), "created": time.time()})
        r.save_meta()
        log("questions", "round " + str(n_round) + ": " + str(len(qs)) + " exercises written", "green")

    todo = r.pending(n_round)
    if not todo:
        log("round", "round " + str(n_round) + " is already complete")
        return
    phase("solving", "round " + str(n_round) + ": " + str(len(todo)) + " exercises")
    log("round", "round " + str(n_round) + ": " + str(len(todo)) + " exercises to do, " +
        str(workers) + " in flight (" + str(len(r.a.solvers)) + " answers + " + str(len(r.a.judges)) +
        " judges run concurrently inside each)", "bold")
    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(loop.solve, n_round, q): q for q in todo}
        try:
            for f in cf.as_completed(futs):
                q = futs[f]
                try:
                    rec = f.result()
                except Interrupted:
                    continue
                except Exception as e:
                    log("warn", q["qid"] + " failed: " + repr(e)[:160], "yellow")
                    continue
                done += 1
                eta = (time.time() - t0) / done * (len(todo) - done)
                mark = "clean" if rec["clean"] else "saved, has errors"
                log("result", q["qid"] + " " + q["title"][:38].ljust(38) + " best=" + rec["verdict"]["best"] +
                    " judges=" + str(rec["verdict"]["best_score"]) + "/10  checker " +
                    str((best_answer(rec) or {}).get("check", {}).get("score")) + "->" +
                    str(rec["final"]["check"]["score"]) + "  " + mark + "   (" + str(done) + "/" +
                    str(len(todo)) + ", eta " + fmt_t(eta) + ")",
                    "green" if rec["clean"] else "yellow")
        except (KeyboardInterrupt, Interrupted):
            Stop.set()
        if Stop.is_set():
            for f in futs:
                f.cancel()


def banner(a, r):
    g = gpu_status()
    lines = ["", Log.paint("  Recursive self-improvement - phase 1: building the answer corpus", "bold"),
             "  config     " + os.path.basename(getattr(a, "config", "") or "config.json"),
             "  model      " + a.model + "   generator=" + a.generator + "   judges=" + a.judge_model,
             "  run        " + a.run + "   ->  " + r.dir,
             "  agents     " + str(len(a.solvers)) + " solvers (" +
             ", ".join(str(x["name"]) for x in a.solvers) + ")  |  " + str(len(a.judges)) + " judges (" +
             ", ".join(str(x["name"]) for x in a.judges) + ")",
             "  budgets    answer " + str(a.answer_tokens) + " tok (the exercise ceiling), judge " +
             str(a.judge_tokens) + " tok",
             "  context    solvers " + str(a.gen_ctx) + " per slot, judges " + str(a.judge_ctx) +
             " per slot (every answer in full, never truncated)",
             "  checker    compile + AST + ruff" + ("" if not a.no_exec else " (execution off)") +
             ", scoring only - no answer is filtered out",
             "  machine    " + (g["name"] + " " + str(g["total"]) + " MB " +
                                ("unified" if g.get("unified") else "VRAM") if g else "no GPU detected"),
             ""]
    Log.block(lines)


if __name__ == "__main__":
    main()
