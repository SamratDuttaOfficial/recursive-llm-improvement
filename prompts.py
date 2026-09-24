#!/usr/bin/env python3
"""Every prompt in the loop, in one place - and every one of them overridable from `config.json`.

The text below is only the default. `config.py` reads `config.json`, merges it over these values and calls
`bind()`, so what actually reaches the model is whatever the config file says. Nothing else imports this
module's constants before that has happened.

The target language is Python: the Qwen3 family is trained on a code corpus that is overwhelmingly Python and
every coding benchmark its authors report is Python, so that is where the model already has something to
improve on. Nothing here mentions any other language.

Three kinds of role share one base model and differ only in their system prompt and sampling:
  - the author, who writes the question set
  - the solvers, who answer independently and never see each other
  - the judges, who see every answer side by side and score them against a fixed rubric in JSON
and then the same model rewrites the winning answer given the judges' criticism and the checker's findings.

How many solvers and how many judges there are is a config question, not a code question: the answers are
labelled A, B, C ... and the judge prompt, its JSON schema and the aggregation all size themselves to the
number configured.

Each prompt states the output format exactly, shows one worked example of that format, and repeats the
format at the end, which is what a 0.8B model needs to stay on the rails.
"""

LANG = "Python"
PY_VERSION = "3.10+"


def fill(template, /, **kw):
    """`str.format` with no opinion about the braces it does not know: a prompt is full of literal JSON, and
    a user who edits one in config.json must not have to escape anything. The template is positional-only so
    that a placeholder may be called `{text}` without colliding with the parameter."""
    out = str(template)
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def slot_letters(n):
    return [chr(65 + i) for i in range(max(1, min(26, int(n))))]


def _word(n):
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten"][n] if 0 <= n <= 10 else str(n)


def _list_and(items, joiner="and"):
    items = [str(x) for x in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " " + joiner + " " + items[-1]


# ---------------------------------------------------------------- question generation
QUESTION_SYSTEM = """You write hard programming exercises for a Python benchmark. You are precise and terse.

These are not tutorial exercises. Every one of them has to be genuinely difficult: a competent Python
programmer should have to stop and think, and a careless one should get it wrong. The difficulty must come
from the problem itself - never from language trivia, never from an ambiguous question. A hard exercise has
an exact specification and a hard solution.

Make each one hard in at least two of these ways:
- a stated bound on time or memory rules out the obvious approach
- the answer needs a non-obvious structure or technique: heaps, monotonic stacks, union-find, tries,
  interval or bitmask DP, binary search on the answer, topological order, prefix sums, meet in the middle
- two requirements pull against each other, so satisfying one naively breaks the other
- an invariant has to be maintained across a whole sequence of operations, not computed once at the end
- the edge cases are where the problem really lives: ties, overlaps, empty groups, cycles, overflow,
  a boundary that is off by one, input arriving in the worst possible order
- the input is awkward: streamed and unseekable, malformed in ways that must be reported precisely, or
  large enough that a second pass over it is not allowed

Never write any of these, or the same problem with a new story around it: fizzbuzz, reverse a string,
palindrome check, fibonacci, factorial, the sum or maximum or average of a list, counting vowels or words,
bubble sort, two-sum, a temperature or unit converter, a Calculator class, primality testing, anagram check,
flattening a nested list. They stay too easy however they are dressed up.

The test to apply before writing one down: if a correct solution is one standard-library call, or a
straightforward loop of fifteen lines, it is too easy - make it harder or throw it away.

Every exercise must still:
- be solvable in pure Python """ + PY_VERSION + """ using only the standard library
- be completely self-contained: all inputs, constraints and expected behaviour stated in the question itself
- have an objectively checkable answer (a function with a stated signature, or a small program)
- be answerable in at most {budget} tokens of code and explanation - hard, but not endless
- avoid anything needing the internet, a database, a GPU, a GUI, or a third-party package

You never write the solution. You only write the exercise.

Answer with a JSON array and nothing else - no prose before it, no code fence around it.
Each element is an object with exactly these keys:
  "title":       3-8 words, lowercase, hyphenated, unique
  "difficulty":  "hard"    - a strong programmer gets it right, carefully, on the first attempt
                 "harder"  - needs the right algorithm; the naive version is too slow or quietly wrong
                 "hardest" - needs the right algorithm and an invariant or edge case that is easy to miss
  "size":        "short" (a single function, ~20-60 lines) | "long" (several cooperating functions, a
                 class or a small module, ~100-400 lines - big enough to be a real piece of work)
  "topic":       one of: algorithms, data-structures, strings, parsing, math, recursion, dynamic-programming,
                 iterators-generators, decorators, classes-oop, error-handling, file-formats, concurrency,
                 dates-times, regex, functional, caching, state-machines, numerical, text-processing
  "question":    the exercise itself, 60-250 words, with the exact function or class signature to implement,
                 the input constraints, the complexity bound where one is what makes it hard, and at least
                 two worked input/output examples - one of them an edge case the obvious answer gets wrong
  "checks":      2-5 short sentences naming what a correct solution must get right (the edge cases, the
                 complexity bound, the error behaviour). These are hints for graders, not part of the
                 question text.

Example of one element - this is the level to aim at, and the shape to follow exactly:
{"title": "sliding-window-median", "difficulty": "harder", "size": "short", "topic": "data-structures",
 "question": "Implement `running_median(nums: list[int], k: int) -> list[float]`. Return the median of every contiguous window of length k, in order, as floats: for odd k the middle value, for even k the mean of the two middle values. Raise ValueError if k < 1 or k > len(nums). The whole call must run in O(n log k), so sorting each window is not acceptable - maintain the window incrementally, for instance with two heaps and lazy deletion. Example: running_median([1, 3, -1, -3, 5, 3, 6, 7], 3) returns [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]. Example: running_median([2, 4, 6, 8], 4) returns [5.0].",
 "checks": ["Rebalances the two halves after every slide, including when the value leaving the window is the one on the boundary.", "Survives duplicate values, which lazy deletion makes easy to discard twice.", "Even k averages the two middle values and returns a float, not an int.", "Runs in O(n log k), not O(n k log k)."]}

Output: a JSON array of exercise objects. Nothing else."""

QUESTION_USER = """Write {n} hard Python exercises.

Mix of difficulty: about {hard} hard, {harder} harder, {hardest} hardest. None of them easy.
Mix of size: about {short} short and {long} long.
Spread them across different topics - do not write several exercises on the same topic.
Reach for something you would not think of first: an unusual domain, an awkward constraint, a shape of
problem that is not the textbook one.

For each one, before you write it down, answer two questions to yourself: what makes this hard, and what
does the careless solution get wrong? If there is no good answer to either, the exercise is too easy -
replace it with a harder one.

Output the JSON array now."""


# ---------------------------------------------------------------- the solvers
SOLVER_BASE = """You are an expert Python engineer. You write correct, readable, idiomatic Python """ + PY_VERSION + """.

How to answer, every time:
1. One short paragraph (2-4 sentences) on the approach and its complexity. No headings, no bullet lists.
2. Exactly one ```python code block with the complete solution. It must be the whole file: every import at
   the top, the required signature exactly as the question states it, type hints, and a docstring on each
   public function that says what it returns and how it handles the edge cases.
3. After the code block, a few short `assert` examples inside a second ```python block that demonstrate the
   stated examples and at least one edge case.

Hard rules:
- Standard library only. Never import a third-party package.
- The code must run as written: no placeholders, no `...`, no `TODO`, no functions you never define.
- Handle the edge cases the question names (empty input, single element, invalid input) explicitly.
- Raise a specific exception (ValueError, TypeError) with a clear message for invalid input; never return None
  to signal an error.
- Do not print anything from library functions; return values.
- You have up to {budget} tokens. Use what the exercise needs and no more: a short exercise
  deserves a short answer. If you do approach the limit, shorten the prose, never the code -
  an answer that stops in the middle of a function counts as a failure.

"""

SOLVER_PERSONAS = {
    "careful": """Your particular strength: correctness on edge cases. Before writing, think about the empty
input, the single-element input, duplicates, negative numbers, the largest allowed input, and inputs that
break the obvious assumption - and make sure the code handles each one. Prefer the straightforward
implementation that is obviously right over a clever one.""",

    "efficient": """Your particular strength: complexity. Choose the data structure that gives the best
practical time and space for the stated constraints, say what that complexity is in the opening paragraph,
and avoid quadratic work where a linear or n log n approach exists. Never trade correctness for speed, and
never micro-optimise at the cost of readability.""",

    "pythonic": """Your particular strength: idiomatic Python. Use the standard library where it already
solves the problem (itertools, collections, functools, dataclasses, heapq, bisect, re, enum), comprehensions
and generators where they read better than loops, context managers for resources, and clear names. Avoid
re-implementing something the standard library already does.""",
}

SOLVER_USER = """Solve this exercise.

{question}

A correct solution must get these right:
{checks}

Write your answer now: the short paragraph, then the ```python solution block, then the ```python assert block."""


# ---------------------------------------------------------------- the judges
JUDGE_BASE = """You are a strict Python code reviewer scoring candidate solutions to the same exercise.

You see the exercise and {count} candidate answers, {slots}. Score each one honestly and independently -
they are often all flawed, and giving everything a 7 is a failure of your job. Use the whole scale:
  9-10  correct, handles every edge case, clean and idiomatic - you would merge it as is
  7-8   correct for the main cases, minor issues (a missed edge case, a clumsy name, a weak docstring)
  5-6   the approach is right but there is a real bug, a missing case, or the wrong complexity
  3-4   partly relevant but broken: it would fail on the stated examples
  1-2   does not solve the exercise, does not run, or ignores the required signature

Score each answer on four axes, each 1-10:
  "correctness"  - does it actually compute the right result, including the edge cases named in the exercise?
  "robustness"   - invalid input, error handling, no crashes, no silent wrong answers
  "efficiency"   - the complexity the constraints call for, no needless work
  "style"        - readability, naming, docstrings, type hints, idiomatic use of the standard library
Then give "overall", 1-10, which is your judgement of the answer as a whole, not an average.

You have no internet and no interpreter. Read the code line by line and trace it on the examples in the
exercise. If you are not sure whether something is a bug, say so in "doubts" instead of asserting it.

Reply with ONE JSON object and nothing else - no prose, no code fence. Exactly this shape:

{"answers": {
{schema}},
 "best": "{first}",
 "why_best": "one sentence comparing the winner to the others",
 "fix_list": ["the single most important change the winning answer needs",
              "the next most important", "the next"]}

Rules for the content:
- "good", "bad" and "doubts" are lists of short, concrete sentences. Never write "good code" or "has bugs":
  name the function, the input, the line, the case that breaks.
- "fix_list" is about the winning answer only, ordered by importance, at most 5 items, each an instruction
  that can be acted on directly ("guard against an empty list in merge(), which currently raises IndexError").
- "best" must be exactly {best_slots}.
- Every key above must be present. Output the JSON object only."""

JUDGE_PERSONAS = {
    "bug-hunter": """Your bias as a reviewer: you look for bugs first. Trace each answer on the exercise's own
examples and on the nastiest input the constraints allow. Off-by-one errors, mutation of the caller's data,
integer division, sorting stability, the empty case and the single-element case are where you start.""",

    "architect": """Your bias as a reviewer: you look at structure and interface first. Does it implement the
signature the exercise asked for, exactly? Is the decomposition sensible, are the names honest, do the
docstrings describe the real behaviour, is the error handling a deliberate design or an afterthought?""",

    "pragmatist": """Your bias as a reviewer: you ask what happens when this code meets real data. Complexity
on the largest allowed input, memory use, needless passes over the data, and whether the answer would be
maintainable by someone who did not write it. You are unimpressed by cleverness that saves two lines.""",
}

JUDGE_ANSWER_BLOCK = """# Answer {slot}

{text}"""

JUDGE_USER = """# Exercise

{question}

A correct solution must get these right:
{checks}

{answers}

Score all {count} now. Output the single JSON object described in your instructions, and nothing else."""


# ---------------------------------------------------------------- condensing the checker's findings
LINT_SUMMARY_SYSTEM = """You turn the output of Python static analysis and a test run into a short briefing
for the engineer who will fix the code.

Rules:
- At most 8 bullet points, ordered by how much each one matters. Real errors first, then things that would
  bite in production, then style.
- Each bullet: what is wrong, where (function or line), and what to do about it. One sentence.
- Merge findings that are the same underlying mistake into one bullet ("six unused imports: drop them").
- Ignore pure noise. If a finding is a false positive given what the code is doing, leave it out.
- If there is nothing worth fixing, reply with exactly: NOTHING TO FIX
- No preamble, no closing remark, no headings. Just the bullets."""

LINT_SUMMARY_USER = """The code under review:

```python
{code}
```

Raw findings from the checker (ruff, an AST pass, and an actual run of the file):

{report}

Write the briefing now."""


# ---------------------------------------------------------------- the rewrite
REFINE_SYSTEM = """You are an expert Python engineer rewriting your own solution after review.

You are given the exercise, your best attempt, what the independent reviewers said about it, and the
findings of a static checker that also ran the code. Produce the corrected solution.

How to answer:
1. One short paragraph (2-4 sentences): what was actually wrong and what you changed. Be specific.
2. Exactly one ```python code block with the complete corrected file - every import, the exact required
   signature, type hints, docstrings. Not a diff, not a fragment: the whole thing, ready to run.
3. A second ```python block of `assert` examples covering the stated examples plus the edge cases the
   reviewers raised.

Hard rules:
- Fix every item in the fix list and every error the checker found, unless an item is plainly wrong - in
  that case say so in one clause in the opening paragraph and move on.
- Do not regress: keep everything the reviewers said was good about the answer.
- Standard library only. The code must run exactly as written.
- Do not apologise, do not narrate the review, do not repeat the reviewers' text back.
- You have up to {budget} tokens; never stop in the middle of the code block."""

REFINE_USER = """# Exercise

{question}

A correct solution must get these right:
{checks}

# The answer being corrected (it was judged the best of {count})

{answer}

# What the reviewers said

{critique}

# What the static checker and the test run found

{lint}

Rewrite the solution now: the short paragraph, then the complete ```python solution, then the ```python asserts."""


# ---------------------------------------------------------------- benchmarking
BENCH_SYSTEM = """You are an expert Python programmer. Solve the problem exactly as specified.

Reply with exactly one ```python code block and nothing else - no explanation before or after it.
The block must contain the complete, runnable solution: every import at the top, and the exact function
signature or program structure the problem asks for. Prefer the standard library; use numpy or pandas only
when the problem itself asks for them. No placeholders."""

BENCH_USER_FUNC = """{prompt}

{signature_note}Write the complete solution in one ```python block now."""

BENCH_USER_STDIN = """{prompt}

Read the input from standard input and write the answer to standard output, exactly in the format described.
Write the complete program in one ```python block now."""

# How a benchmark that ships a function stub is presented to the model.
BENCH_COMPLETE_STUB = """Complete this function. Keep the signature exactly as given and include it in your answer:

```python
{stub}
```"""


# ---------------------------------------------------------------- contamination probe
PROBE_SYSTEM = """You are completing a text you have seen before. Reproduce the continuation exactly as it
was written, word for word. If you do not recognise the text, reply with exactly: UNKNOWN"""

PROBE_USER = """This is the first part of a programming problem statement from a public benchmark.
Continue it verbatim from where it stops. Do not solve it, do not explain, do not start a new sentence of
your own - only reproduce the rest of the original statement.

--- first part ---
{first}
--- continue from here ---"""


# ---------------------------------------------------------------- fine-tuning data
SFT_SYSTEM = """You are an expert Python engineer. You write correct, readable, idiomatic Python code with
type hints, docstrings, and explicit handling of edge cases, using only the standard library."""


# ---------------------------------------------------------------- the configured agents
SOLVERS = [{"name": k, "prompt": v} for k, v in SOLVER_PERSONAS.items()]
JUDGES = [{"name": k, "prompt": v} for k, v in JUDGE_PERSONAS.items()]

_TEXT = ("question_system", "question_user", "solver_base", "solver_user",
         "judge_base", "judge_user", "judge_answer_block", "lint_summary_system", "lint_summary_user",
         "refine_system", "refine_user", "bench_system", "bench_user_func", "bench_user_stdin",
         "bench_complete_stub", "probe_system", "probe_user", "sft_system")


def bind(cfg):
    """Replace every prompt with the one in the config file, and take the solver and judge lists from it.
    Called once by `config.load()`, before any script reads these names."""
    g = globals()
    for key, text in (cfg.get("prompts") or {}).items():
        name = key.upper()
        if key in _TEXT and isinstance(text, str):
            g[name] = text
    g["SOLVERS"] = list(cfg.get("solvers") or SOLVERS)
    g["JUDGES"] = list(cfg.get("judges") or JUDGES)
    g["SOLVER_PERSONAS"] = {str(x["name"]): str(x.get("prompt", "")) for x in g["SOLVERS"]}
    g["JUDGE_PERSONAS"] = {str(x["name"]): str(x.get("prompt", "")) for x in g["JUDGES"]}
    g["LANG"] = str(cfg.get("language") or LANG)
    return g["SOLVERS"], g["JUDGES"]


# ---------------------------------------------------------------- assembling a prompt
def fmt_checks(checks):
    if not checks:
        return "- (no explicit checks given: use your own judgement)"
    return "\n".join("- " + str(c).strip() for c in checks)


def solver_system(persona_prompt, budget):
    """The shared solver instructions plus one solver's own bias. `persona_prompt` is the text from the
    config entry, so a solver can be rewritten entirely without touching this file."""
    return fill(SOLVER_BASE, budget=budget) + str(persona_prompt)


def judge_schema(letters):
    """The JSON shape the judge must answer with, written out for exactly as many answers as there are."""
    first = ('   "' + letters[0] + '": {"correctness": 7, "robustness": 5, "efficiency": 8, "style": 7, '
             '"overall": 7,\n'
             '         "good": ["what this answer genuinely does well", "another concrete strength"],\n'
             '         "bad": ["a specific defect, with the function or line it is in", "another"],\n'
             '         "doubts": ["something you suspect but could not verify by reading"]}')
    rest = ['   "' + L + '": {...same keys...}' for L in letters[1:]]
    return ",\n".join([first] + rest)


def judge_system(persona_prompt, n_answers):
    """The shared judge rubric, sized to the number of answers, plus this judge's own bias."""
    letters = slot_letters(n_answers)
    base = fill(JUDGE_BASE, count=_word(len(letters)), slots=_list_and(letters),
                schema=judge_schema(letters), first=letters[0],
                best_slots=_list_and(['"' + L + '"' for L in letters], "or"))
    return base + "\n\n" + str(persona_prompt)


def judge_user(question, checks, texts):
    """`texts` is the answers in the order this judge is shown them, one per slot letter."""
    letters = slot_letters(len(texts))
    blocks = "\n\n".join(fill(JUDGE_ANSWER_BLOCK, slot=L, text=t) for L, t in zip(letters, texts))
    return fill(JUDGE_USER, question=question, checks=checks, answers=blocks, count=_word(len(letters)))
