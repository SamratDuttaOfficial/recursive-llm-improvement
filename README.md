# Recursive self-improvement of a small LLM

Can `qwen3.5:0.8b` make itself better at Python by generating its own exercises, criticising its own
answers, and training on the result? This runs that loop end to end, on your own GPU, automatically.

```bash
run.bat          # Windows
./run.sh         # macOS / Linux
```

That is the only command you need, on a machine with nothing installed. It prepares Python, downloads
Ollama if it is not there, pulls the model, starts its own batched `llama-server` sized to the GPU it
finds, opens a live dashboard at <http://127.0.0.1:8777>, and begins.

**It installs nothing outside this folder.** If the machine already has Python 3.9+, that one is used as
is. If it does not, a private CPython is unpacked into `.python/` for this project alone - no admin or
sudo, no PATH changes, no effect on any Python you use for other work. Everything else
(PyTorch or MLX, ruff, pyarrow) goes into `.venv/` the first time it is needed. Delete `.python/` and
`.venv/` and the machine is exactly as it was.

If you would rather use your own interpreter, `python run.py` still works and behaves identically.

`run.py` does not stop on its own: a finished round is followed by a new set of exercises and it keeps
going until you stop it. **Ctrl-C is a smooth stop** - nothing new is started, the exercises already in
flight run to the end and are saved, and then it exits. A second Ctrl-C drops them instead, a third quits
on the spot. Everything already produced is on disk whichever you press, and re-running the same command
continues from exactly there.

## Three scripts, one job each

| run it | does | state | UI |
|---|---|---|---|
| `run.bat` / `./run.sh` | writes the exercises and builds the corpus of good answers | `state/<run>/rounds/` | :8777 |
| `finetune.bat` / `./finetune.sh` | trains on those answers, produces `v1`, `v2`, ... | `state/<run>/ft/<v>/` | :8778 |
| `benchmark.bat` / `./benchmark.sh` | measures every model that has no results yet | `state/<run>/bench/` | :8779 |

Each runs on its own, resumes on its own, and can be used without the others. Every launcher calls
`setup.bat` / `setup.sh` first, which prepares Python and the virtual environment and then does nothing on
every later run; you can also run it by itself to get the machine ready in advance. Arguments pass straight
through - `run.bat --generator latest --new`.

`pipeline.bat` / `./pipeline.sh` is optional and only calls the three in order. The rest are not scripts:
`common.py`, `codecheck.py`, `config.py` and `prompts.py` are shared libraries, and `train_worker.py` is
launched *by* `finetune.py` inside the venv.

## One config file

`config.json` is written next to the scripts on the first run and decides everything the three phases do.
Change it, run again, and whatever it names - a different model, another benchmark - downloads itself.

```jsonc
"model":      { "base": "qwen3.5:0.8b", "generator": "base", "judge": "base" },
"parallel":   { "workers": 0, "slots": 0, "backend": "auto" },   // 0 = measure the GPU and decide
"corpus":     { "questions_per_round": 24, "answer_tokens": 20000, "judge_weight": 0.7, ... },
"solvers":    [ { "name": "careful", "temperature": 0.25, "top_k": 20, "prompt": [...] }, ... ],
"judges":     [ { "name": "bug-hunter", "temperature": 0.1, "top_k": 15, "prompt": [...] }, ... ],
"checker":    { "ruff_rules": "...", "penalties": { "error": 34, ... } },
"finetune":   { "epochs": 3, "rank": 32, "lr": 0.0001, ... },
"benchmarks": { "use": ["lbpp", "evoeval"], "catalog": { ... where each one is downloaded from ... } },
"prompts":    { "question_system": [...], "judge_base": [...], ... every prompt, in full },
```

- **How many solvers and judges** is the length of those two lists. Add a fourth judge and the judges' JSON
  schema, the answer labels (A, B, C, D...), the aggregation and the dashboard all follow; nothing in the
  code assumes three. The `prompt` of each entry is that agent's own bias, appended to the shared rubric.
- **Every prompt is in the file**, written as a list of lines so it stays readable. `{...}` markers are
  filled in by the code and have to stay; the rest is yours. `{count}`, `{slots}`, `{schema}` and
  `{best_slots}` in the judge prompt expand to match however many agents there are.
- **Adding a benchmark** is a `catalog` entry: where to download it (`hf-parquet`, `hf-jsonl` or a URL),
  which of its columns hold the prompt, the tests and the reference solution, which of the four shapes it
  has (`lbpp`, `evalplus`, `mbpp`, `lcb`) and which harness runs an answer. Name it in `use` and it fetches
  itself. The five it already knows are there to copy from.
- Comments (`//`, `#`) and trailing commas are allowed. A misspelled key is reported rather than ignored.
- **A command-line flag always wins** over the file, for one run: `run.bat --questions 8 --judge-weight 1`.
  `--config other.json` uses a different file, `--print-config` shows what is actually in effect, and
  `--init-config` writes a fresh copy of the defaults.

## The loop

```
   run.py                      finetune.py                  benchmark.py
   ──────                      ───────────                  ────────────
   write exercises             build the training set       download LBPP + EvoEval
   3 solvers answer            LoRA on the corpus           probe the base model for memorisation
   checker: compile/lint/run                                run every problem, execute the real tests
   3 judges score in JSON      merge + convert to GGUF      compare each version against the base
   pick the winner             register as v1, v2, ...
   rewrite it with the
     criticism + findings
   save the training pair
```

Then start again with `python run.py --generator latest --new`: the newest fine-tuned model writes the
answers, while **the judges always stay on the untouched base model**, so the yardstick never moves.

Or run the whole cycle in one go:

```bash
python pipeline.py --cycles 2
```

## Phase 1 - building the corpus (`run.py`)

**The exercises, and how hard they are.** The model is asked for a JSON array of Python exercises, each
with the exact signature to implement, worked examples, and a list of what a correct answer must get right.
Nothing may need more than **20,000 tokens** of answer - a short one takes a few hundred, a long one a few
thousand. A round is filled **a batch at a time** (`--question-batch 6`) rather than in one call: asked for
two dozen JSON objects at once a 0.8B model writes a handful and closes the array, and a set that does
comply overruns the token ceiling and is cut off mid-object - either way the round ends far short.

**There is no easy band.** Left alone a small model writes fizzbuzz, and a corpus of fizzbuzz teaches
nothing, so the exercise prompt does the work of raising the floor. The difficulty scale runs `hard` ->
`harder` -> `hardest` and nothing below it exists. The writer is told what hardness is allowed to come from
- a stated time or memory bound that rules out the obvious approach, a non-obvious structure (heaps,
monotonic stacks, union-find, tries, interval or bitmask DP, binary search on the answer), requirements
that pull against each other, an invariant to maintain across a whole sequence of operations, edge cases
that are the real problem, or input that is streamed, malformed or too large for a second pass - and it
must use at least two of them per exercise. It is also given the list of exercises it may never write
(fizzbuzz, reverse a string, palindrome, fibonacci, two-sum, a Calculator class, anagrams, flatten a nested
list, and the rest), the test to apply before writing one down (*if a correct solution is one
standard-library call or a straightforward fifteen-line loop, it is too easy*), and one worked example -
the sliding-window median in O(n log k) - to anchor the level. `corpus.difficulty_mix` in `config.json`
only says how much of each round is `harder` and `hardest`; the rest is `hard`.

Expect the scores to be lower than they were on easy exercises. That is the point: a 0.8B model that
fails an interval-DP problem and then rewrites it under criticism produces a more useful training pair
than one that nails another string reversal.

Variety comes from **sampling, not from recall**. Nothing the model has already written goes back into its
context; it is simply run hot - `question_temp 1.25`, `top_k 120`, `top_p 0.98`, `min_p 0.02`, all in
`config.json` - and every batch is an independent draw. The only thing carried across is the set of titles
already used, which is enforced here rather than argued about in the prompt. Turn the temperature up for
stranger exercises, down if the JSON starts coming back malformed.

**It runs until you stop it.** When a round is finished the next set is written straight away, and that
goes on for as long as you leave it - `corpus.rounds` is 0 by default, which means no end. `--rounds 3`
caps one run at three sets; `--no-new` finishes only what is already pending and then exits. The first
Ctrl-C takes the rest of the queue away and waits for the exercises in flight: each one finishes its
judges, its rewrite and its corpus row, so nothing is left half-answered and the log says how many it is
waiting for. A second Ctrl-C drops them where they stand - a part-streamed answer is discarded rather than
checked and saved as though the model had finished it - and re-running picks up at the exact sub-step that
was interrupted.

**Three answers.** Three solvers run concurrently on the same model with different sampling and different
strengths - `careful` (temperature 0.25, edge cases first), `efficient` (0.7, complexity first) and
`pythonic` (1.0, the standard library first). They never see each other's work. Three is the default, not a
rule: the `solvers` list in `config.json` decides how many there are and what each one is told.

**The checker** (`codecheck.py`) is the part that does not depend on any model's opinion. For each answer it
extracts the code, compiles it, runs an AST pass (undefined names, unused imports, mutable default
arguments, bare `except`, `is` against a literal, missing docstrings) and `ruff`, then executes the file in
a separate process with a timeout: its doctests, any `test_*` functions, and - the sharpest test of all -
the `assert` examples the answer itself wrote. An answer whose own worked examples do not hold is caught
here even when it lints perfectly clean.

**Three judges.** Three more instances of the base model, each with its own sampling and its own bias
(`bug-hunter`, `architect`, `pragmatist`), see all three answers side by side - **whole, never truncated and
never summarised**, which is why the judge server gets a context three times the size of a solver's. Each
scores every answer on correctness, robustness, efficiency and style, lists concrete strengths, defects and
doubts, names a winner and writes a fix list, all as one JSON object. The three answers are shown to each
judge **in a different order**, so position cannot decide the outcome, and the mapping is stored with the
record. A judge that returns unusable JSON is retried once, then ignored. As with the solvers, the `judges`
list in `config.json` sets how many there are; the rubric, the JSON schema it demands and the answer labels
all resize themselves to match.

**Choosing the winner** uses both kinds of evidence rather than either alone. The judges read for intent and
design, which no checker can do; the checker knows what actually compiles, lints and passes its own asserts,
which a 0.8B judge often gets wrong from reading. The two are blended on one 0-10 scale - `--judge-weight
0.7` by default, `0` for the checker alone, `1` for the judges alone - with two rules on top: **code that
does not run can never beat code that does**, whatever the judges thought of it, and if all three judges
fail to return usable JSON the checker decides by itself.

**The rewrite.** The winning answer, the merged criticism and a model-written summary of the checker's raw
findings go back to the model, which produces the corrected solution. That is checked again and saved as
the training pair, with the before/after checker scores so you can see whether the rewrite actually helped.

**A rewrite that came out worse is refused.** A second attempt by the same 0.8B model is not always an
improvement - it drops the import it was using, re-indents a working function into a syntax error, or hangs
on an input the original handled. So the rewrite is compared against the answer it was asked to improve,
and the exercise keeps the judges' pick instead if any of this is true:

- it came back with no usable code
- it times out, stops importing, or fails its own worked examples where the original did not
- the checker finds more errors, or more warnings, than in the answer it was given

The refused rewrite is still stored on the record - the detail view shows it with the reason - it just does
not go into the corpus. `reverted` in the round table and on the dashboard counts how often this happens;
a high number means the rewrite step is hurting more than helping.

**Every rewrite is saved, and every saved answer is trained on.** There is no quality gate anywhere in the
pipeline: one exercise in, one training pair out, and the whole corpus goes into the fine-tune. A rewrite
that still has checker errors is written all the same and flagged `clean: false`, because what the model
does badly is a fact about the model, and dropping it would hide that from the very report meant to show
it. The checker score is on every row, so the spread is visible without anything being thrown away.

## Phase 2 - fine-tuning (`finetune.py`)

LoRA (rank 32, alpha 64, 3 epochs, cosine schedule) over the corpus in ChatML format, then merged
into the base weights and converted to GGUF so the next round can be served by the same `llama-server`.

- **NVIDIA / CPU**: PyTorch + PEFT, bf16, gradient checkpointing, resumed from the last `checkpoint-N`.
- **Apple silicon**: `mlx-lm` LoRA on the unified-memory GPU, resumed from the last adapter file.

The right stack is installed automatically for whichever machine you are on. Out of memory is not a failure:
the sequence length halves, then the base drops to 4-bit, and training restarts from the last checkpoint.
The base model is also converted to GGUF at the same precision, so the benchmark compares like with like.

Seven phases - `dataset, deps, base weights, train, merge, gguf, register` - each recorded as it completes.

### Making the *next* version - the two strategies

Once `v1` exists there are two different experiments, and the difference matters:

```bash
python finetune.py                          # v2 = a clean run from the BASE model on the whole corpus
python finetune.py --from-model latest      # v2 = v1's own weights, trained further on the new round only
```

**From base (the default).** Each version is one clean training run over everything generated so far, so
`v3` is not `v1`'s mistakes compounded three times, every version stays directly comparable to the base,
and a bad round can be dropped with `--rounds 1,3`. It costs a full training run each time, which for a
0.8B LoRA is minutes, not hours.

**Stacking (`--from-model latest`).** Keeps training the newest version's weights. Cheaper per cycle and the
learning accumulates - but so does the drift, and it is the setting where a self-training loop can quietly
collapse onto its own style. Because a stacked version has already seen its parent's data, it trains **only
on the rounds its parent never saw**; `finetune.py` works that out from the registry and says so. Pass
`--rounds` yourself to override it.

Either way the registry records the lineage - `parent`, `trained_on` (the rounds this run used) and
`rounds` (everything in its history) - so a chain of versions is readable afterwards.

`pipeline.py --cycles 3` retrains from base each cycle; add `--stack` to chain them instead.

**You cannot accidentally train the same model twice.** Before training, the dataset is fingerprinted; if an
existing version was built from the same parent, on the same data, with the same recipe, the run stops and
tells you to generate another round first (`--force` overrides). Each version keeps its checkpoints, merged
weights and GGUF - roughly 3 GB - and the run prints where that went so you can delete `ckpt/` once you are
happy with it.

## Phase 3 - benchmarking (`benchmark.py`)

Which benchmarks run is `benchmarks.use` in `config.json`; every one it knows how to fetch is described in
`benchmarks.catalog` beside it. Two Python benchmarks are on by default, both run by actually executing the
unit tests, pass@1, greedy:

| | |
|---|---|
| **LBPP** | *Less Basic Python Problems* (Cohere) - 161 usable problems of 162. Written for the paper *On Leakage of Code Generation Evaluation Datasets* as a benchmark that was **fresh and unleaked at creation**, and published with every field zlib+base64 encoded so crawlers never ingest the problems or their solutions as plain text. Harder than HumanEval and MBPP. |
| **EvoEval** | 199 usable problems of 200. Every HumanEval task rewritten so that the **memorised HumanEval solution is the wrong answer**: `subtle` changes the requirement in ways that are easy to miss, `creative` restates it as an unrelated story. Same difficulty as HumanEval - which is where a 0.8B model can still score - but nothing can be answered from memory. Its `splits` also offer `difficult`, `combine` and `tool_use`. |

The catalog also holds `humanevalplus` and `mbppplus` - both public for years, so contaminated for any
recent model, and out of `use` for exactly that reason - and `lcb` (LiveCodeBench), whose test cases ship
inline so the download runs to several GB, with competitive-programming problems far above a 0.8B model.
Naming any of them in `use` is enough; it downloads itself on the next run. A benchmark that is not there
yet is a new `catalog` entry - a repository, a file, and which of its columns hold the prompt, the tests
and the reference solution.

**On contamination.** LiveCodeBench's rolling date cut-off is the usual answer to leakage, but every public
release ends in April 2025 while this model's training data runs into 2026, so a date filter proves nothing
here; and HumanEval and MBPP have been public for years, which is why neither is used by default. Both
default benchmarks are unanswerable from memory by construction - LBPP because it was never published as
plain text, EvoEval because the memorised answer is the wrong one. Rather than take either on trust, every
problem in **both** is put through a **memorization probe**
against the base model: it is shown the first 45% of the statement and asked to reproduce the rest word for
word, and a statement it can reproduce (8-gram overlap above 0.45, or a verbatim run of 28+ words) is
flagged. Every score is then reported twice - over all problems, and over the subset the model demonstrably
has **not** memorised. That second column is the one to read.

**The harness checks itself.** On first use every problem's own reference solution is run through this
harness; a problem whose official solution does not pass here is dropped rather than counted as a model
failure. LBPP: 161 of 162. EvoEval: 199 of 200.

**The base model is measured once and never again.** Later runs only measure versions that have no results
yet (`--force` re-measures). The base keeps whichever identity it was first measured under, so converting it
to f16 during the first fine-tune - which is what makes the comparison like-for-like - cannot give it a
second identity and quietly earn it a second benchmark run.

## Why Python

The Qwen3 family is trained on a code corpus that is overwhelmingly Python, and every coding benchmark its
authors report - HumanEval, MBPP, LiveCodeBench - is Python. It is the language the model already has real
competence in, which is what makes an improvement measurable rather than noise. Nothing in the prompts
mentions any other language.

## Using the whole GPU, on either machine

One `llama-server` process **per role** with **N parallel slots** (batched decoding), and the slot count
fitted to the memory that is actually free: it loads, measures, and grows until the free memory minus a
margin is used. Solvers and judges get separate servers even on identical weights, because a judge needs
roughly three times the context; giving the solvers that same context would reserve a KV cache they never
touch and cost most of the slots. On top of that, several exercises are in flight at once and each one runs its three solvers, then its
three judges, concurrently - so the batch stays full.

- **NVIDIA** - CUDA backend, VRAM read from `nvidia-smi`.
- **AMD** - ROCm or Vulkan, VRAM from `rocm-smi`.
- **Apple silicon** - Metal, with the budget taken as 70% of unified RAM and free memory from `vm_stat`;
  training uses MLX on the same unified memory.

If `llama-server` cannot be used at all, everything falls back to Ollama's own server automatically.
`--slots N` pins the slot count, `--workers N` the number of exercises in flight.

## The dashboard

`dashboard.html`, served on 127.0.0.1 by whichever phase is running and opened for you (`--no-browser` to
stop that, `--no-dash` to skip it, `--dashboard` to serve saved state without running anything). Opened
straight from disk it finds whichever phase is running by probing ports 8777-8781.

- **Overview** - the pipeline with the live stage, per-round progress, judge scores, checker score before
  and after the rewrite, and what the rewrite is worth on average.
- **Live** - every model call streaming right now, with its token budget, speed and text; click for the full
  output.
- **Corpus** - every exercise, its three answers with per-axis judge scores and the checker's findings, the
  fix list, the condensed findings and the rewritten answer.
- **Training** - loss curve, step, epoch, checkpoint, the recipe, the trainer's own output.
- **Benchmark** - pass@1 per model and benchmark with the change against the base, the contamination screen,
  and every problem's generated code and failure.
- **Events** - the console feed.

## Where things are

```
config.json                                   the model, the agents, the prompts, the benchmarks
state/<run>/rounds/round_001/questions.json   the exercise set
state/<run>/rounds/round_001/q_r001q007.json  one exercise: every answer, every judge, verdict, rewrite
state/<run>/corpus/round_001.jsonl            one training pair per exercise, nothing filtered out
state/<run>/corpus/<anything>/*.jsonl         another machine's corpus, merged in - see below
state/<run>/ft/v1/                            dataset, checkpoints, adapter, merged model, progress
state/<run>/ft/v2/                            the next version (from base, or stacked on v1)
state/<run>/bench/<model>/<bench>/            one file per problem, plus the summaries
state/<run>/bench/comparison.json             the table
models/registry.json                          base + every version, and where their weights are
.python/                                      a private Python, only when the machine had none
.venv/                                        everything heavier than the standard library
data/                                         downloaded benchmarks and the contamination screens
logs/                                         llama-server, ollama and trainer output
```

## Building the corpus on more than one machine

Generation is the slow part, so it is worth splitting across whatever hardware you have and training on the
combined result. Run `run.py` on each machine as normal - nothing has to be coordinated - then bring the
corpora together before `finetune.py`.

Every corpus row is stamped with the machine that wrote it (`host`) and the run it belongs to, so a merged
corpus still says where each answer came from. There are two ways to combine them:

```bash
cp -r /path/from/laptop/state/default/corpus  state/default/corpus/from-laptop
```

```bash
./finetune.sh --corpus /mnt/share/laptop-corpus --corpus /mnt/share/workstation-corpus
```

The first copies the files in; the search under `corpus/` is recursive, so `from-laptop/round_001.jsonl`
does not collide with the local `round_001.jsonl` - which matters, because every machine starts at round 1
and names its files identically. The second reads them where they are and copies nothing. Both can be used
at once, and naming the same corpus twice does not count it twice.

Answers are then de-duplicated by exercise, so the same exercise solved on two machines contributes one
training pair, and a later round's answer replaces an earlier one. `finetune.py` prints what it read:

```
corpus   4 corpus files, 9 answers, 7 after de-duplicating by exercise
             3  default/corpus/round_001.jsonl
             3  default/corpus/from-laptop/round_001.jsonl
             1  default/corpus/from-laptop/round_002.jsonl
             2  laptop-corpus/round_001.jsonl
dataset  7 training examples from rounds 1, 2 across 3 machines (avg 321 chars)
```

Only the corpus needs merging. Fine-tuning and benchmarking then run once, on the machine with the GPU.

A corpus written by an earlier version has none of these fields - no `host`, no `run`, no `kept`, no
`clean` - and merges in exactly the same way. It trains as it always did and reports its machine as
unknown; only the question and the answer are actually required of a row. Mixed old and new corpora in one
folder are fine.

## Flags worth knowing

Every flag below has a home in `config.json`; passing it overrides the file for that one run. All three
scripts also take `--config PATH`, `--print-config` and `--init-config`.

**`run.py`** `--generator base|latest|v2` who answers (judges stay on base) - `--questions 24` per round -
`--rounds 3` rounds in one go, 0 (the default) means until Ctrl-C - `--new` / `--no-new` - `--workers N` -
`--slots N` -
`--answer-tokens 20000` the ceiling for one answer - `--judge-weight 0.7` judges against checker -
`--question-batch 6` exercises per generation call - `--question-temp 1.25` / `--question-top-k 120` /
`--question-top-p` / `--question-min-p` how adventurous the exercise writer is -
`--no-exec` lint without running - `--run name` a separate experiment - `-v`

**`finetune.py`** `--rounds 1,2` which corpus rounds - `--corpus PATH` another machine's corpus, repeatable
- `--from-model base|latest|v1` retrain from base or
stack on a version - `--version v4` name it - `--epochs`, `--lr`, `--rank`, `--seq-len`, `--batch`,
`--accum` - `--load-4bit` - `--force` train despite the duplicate guard - `--restart` - `--no-gguf`

**`benchmark.py`** `--models base,v1` - `--benchmarks lbpp,evoeval` any names from `benchmarks.catalog` -
`--splits evoeval=subtle,creative` - `--limit N` - `--no-probe` skip the contamination screen -
`--no-self-check` - `--force` re-measure - `--report` print and exit

**`pipeline.py`** `--cycles N` - `--stack` chain versions instead of retraining from base -
`--from-stage corpus|finetune|benchmark` - `--skip finetune`

## Requirements

A GPU worth using. That is all: `setup.bat` / `setup.sh` brings its own Python when the machine has none,
and everything else installs itself on first use. The orchestration is standard library only; `.venv` holds
ruff, pyarrow, numpy/pandas (some benchmark problems need them) and the training stack.
