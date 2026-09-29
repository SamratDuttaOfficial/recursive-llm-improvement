#!/usr/bin/env python3
"""The actual training step. Runs inside the project venv (never in the stdlib-only orchestrator), so it may
import torch / peft / mlx. `finetune.py` starts it, reads its PROGRESS lines and restarts it on failure.

Two backends, chosen by the machine:
  cuda / cpu     - transformers + PEFT LoRA, bf16, gradient checkpointing, resumed from the last checkpoint
  apple silicon  - MLX LoRA on the unified-memory GPU in a training loop of its own, sized to the memory that
                   is free, resumed from the last adapter it saved and merged into the base model's own files

Both print one JSON PROGRESS line per logging step so the dashboard can draw the loss curve. Everything here
is idempotent: killed at any point, the next start continues.
"""
import argparse, json, math, os, re, shutil, subprocess, sys, time

sys.stdout.reconfigure(line_buffering=True)


def emit(kind, **data):
    print("PROGRESS " + json.dumps({"kind": kind, "t": time.time(), **data}), flush=True)


# ---------------------------------------------------------------- torch / PEFT
def train_torch(a):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                              Trainer, TrainerCallback, TrainingArguments)

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (
        torch.float16 if use_cuda else torch.float32)
    emit("setup", device="cuda" if use_cuda else "cpu", dtype=str(dtype).split(".")[-1],
         gpu=torch.cuda.get_device_name(0) if use_cuda else "cpu")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    emit("data", examples=len(rows))

    def encode(batch):
        out = tok(batch["text"], truncation=True, max_length=a.seq_len)
        return out

    ds = Dataset.from_list([{"text": r["text"]} for r in rows]).map(
        encode, batched=True, remove_columns=["text"])

    quant = None
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=dtype, quantization_config=quant, trust_remote_code=True,
        device_map={"": 0} if use_cuda else None, attn_implementation="eager")
    model.config.use_cache = False
    if a.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in targets if t in present] or ["q_proj", "v_proj"]
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             bias="none", task_type="CAUSAL_LM", target_modules=targets))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emit("model", trainable=trainable, total=sum(p.numel() for p in model.parameters()), targets=targets)

    ckpt_dir = os.path.join(a.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume = None
    if os.path.isdir(ckpt_dir):
        cks = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
        if cks:
            resume = os.path.join(ckpt_dir, max(cks, key=lambda d: int(d.split("-")[1])))
            emit("resume", checkpoint=os.path.basename(resume))

    class Progress(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if not logs:
                return
            emit("step", step=state.global_step, max_steps=state.max_steps,
                 loss=logs.get("loss"), lr=logs.get("learning_rate"),
                 epoch=round(state.epoch or 0, 3), grad_norm=logs.get("grad_norm"))

        def on_save(self, args_, state, control, **kw):
            emit("checkpoint", step=state.global_step)

    targs = TrainingArguments(
        output_dir=ckpt_dir, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=1, save_steps=a.save_steps, save_total_limit=2,
        bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", weight_decay=0.01, max_grad_norm=1.0, report_to=[],
        dataloader_num_workers=0, seed=a.seed, save_safetensors=True, disable_tqdm=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, callbacks=[Progress()],
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train(resume_from_checkpoint=resume)
    adapter = os.path.join(a.out, "adapter")
    trainer.model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    emit("trained", adapter=adapter)

    if a.merge:
        emit("merging", to=os.path.join(a.out, "merged"))
        del model, trainer
        import gc
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(a.base, dtype="auto", trust_remote_code=True,
                                                    device_map="cpu")
        merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        out = os.path.join(a.out, "merged")
        merged.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        emit("merged", path=out)
    emit("done")


# ---------------------------------------------------------------- Apple silicon / MLX
# The MLX path runs its own training loop instead of mlx-lm's trainer, so that everything deciding how much
# memory a step needs is sized here - and measured before training starts:
#   - Qwen3.5's linear-attention (Gated DeltaNet) layers. mlx-lm's fused kernel for them has no backward pass,
#     so training goes through `gated_delta_ops`, a Python loop with one step per token whose backprop keeps two
#     16x128x128 float32 states per token per layer. `chunked_gated_delta` computes the same recurrence 64
#     tokens at a time; where it does not agree with mlx-lm, mlx-lm's loop runs under checkpointing instead.
#   - The output layer. A 248k-token vocabulary makes one 2048-token example's logits 1 GB in bf16, and the
#     loss's backward pass holds several copies; the loss here takes the vocabulary SLICE tokens at a time.
#   - Every decoder layer runs under gradient checkpointing.
#   - MLX sizes itself by the GPU working set, about 3/4 of RAM, which on a busy Mac is far more than is free.
#     It is held to what is free, and the sequence length is the longest whose measured step fits.

mx = None               # mlx.core, bound by train_mlx: MLX exists only on Apple silicon
CHUNK = 64              # tokens per piece of the linear-attention recurrence
SLICE = 256             # tokens per piece of the vocabulary-wide loss
CHUNK_CALLS = [0]       # how many times training went through the replaced recurrence


def _unit_lower_inverse(L):
    """(I + L)^-1 for a strictly lower-triangular L (..., n, n): forward substitution on 16x16 blocks, joined
    blockwise. Matmuls, slices and concatenations only, so it runs and differentiates on the GPU, where MLX's
    own linalg routines do not."""
    n = L.shape[-1]
    if n > 16:
        h = n // 2
        top = _unit_lower_inverse(L[..., :h, :h])
        bottom = _unit_lower_inverse(L[..., h:, h:])
        corner = -(bottom @ (L[..., h:, :h] @ top))
        zeros = mx.zeros(tuple(top.shape[:-1]) + (n - h,), dtype=L.dtype)
        return mx.concatenate([mx.concatenate([top, zeros], axis=-1),
                               mx.concatenate([corner, bottom], axis=-1)], axis=-2)
    eye = mx.eye(n, dtype=L.dtype)
    rows = [mx.broadcast_to(eye[0], tuple(L.shape[:-1]))]
    for i in range(1, n):
        rows.append(eye[i] - (L[..., i:i + 1, :i] @ mx.stack(rows, axis=-2))[..., 0, :])
    return mx.stack(rows, axis=-2)


def chunked_gated_delta(q, k, v, g, beta, state=None):
    """mlx-lm's `gated_delta_ops` for a per-head decay, 64 tokens at a time. Same arguments and results:
    q, k (B, T, Hk, Dk), v (B, T, Hv, Dv), g and beta (B, T, Hv), state (B, Hv, Dv, Dk); returns y
    (B, T, Hv, Dv) in q's dtype and the final float32 state.

    Per token:  S = g S;  u = beta (v - S k);  S = S + u k^T;  y = S q.  Inside a chunk that starts from S0,
    with gamma_i the decay since the chunk began, the updates solve a unit lower-triangular system
    (I + A) U = beta V - beta gamma K S0^T, where A_ij = beta_i (gamma_i / gamma_j) k_i.k_j for j < i. So
    every u, every y and the chunk's final state come from a few matrix products, and only the walk from one
    chunk to the next is sequential."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype, f32, C = q.dtype, mx.float32, CHUNK
    if Hv // Hk > 1:
        q, k = mx.repeat(q, Hv // Hk, -2), mx.repeat(k, Hv // Hk, -2)
    pad = (-T) % C
    N = (T + pad) // C

    def chunks(x):      # (B, T, H, ...) -> (B, H, N, C, ...), zero-padded: a padded step changes nothing
        x = mx.moveaxis(x.astype(f32), 1, 2)
        if pad:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 3))
        return x.reshape((B, Hv, N, C) + tuple(x.shape[3:]))

    q, k, v, beta = chunks(q), chunks(k), chunks(v), chunks(beta)
    G = mx.cumsum(chunks(mx.log(mx.maximum(g.astype(f32), 1e-30))), axis=-1)       # log gamma
    incl = mx.tril(mx.ones((C, C), dtype=f32)) > 0              # j <= i
    strict = mx.tril(mx.ones((C, C), dtype=f32), -1) > 0        # j < i
    decay = mx.where(incl, mx.exp(mx.where(incl, G[..., :, None] - G[..., None, :], 0.0)), 0.0)
    kT = k.swapaxes(-1, -2)
    inv = _unit_lower_inverse(mx.where(strict, beta[..., None] * (k @ kT) * decay, 0.0))
    gamma = mx.exp(G)
    w_v = inv @ (v * beta[..., None])
    w_k = inv @ (k * (beta * gamma)[..., None])
    attn = mx.where(incl, (q @ kT) * decay, 0.0)
    q_in = q * gamma[..., None]
    k_out = k * mx.exp(G[..., -1:] - G)[..., None]
    g_end = mx.exp(G[..., -1])[..., None, None]

    S = mx.zeros((B, Hv, Dv, Dk), dtype=f32) if state is None else state.astype(f32)
    ys = []
    for n in range(N):
        S_T = S.swapaxes(-1, -2)
        u = w_v[:, :, n] - w_k[:, :, n] @ S_T
        ys.append(q_in[:, :, n] @ S_T + attn[:, :, n] @ u)
        S = S * g_end[:, :, n] + u.swapaxes(-1, -2) @ k_out[:, :, n]
    y = mx.stack(ys, axis=2).reshape(B, Hv, N * C, Dv)[:, :, :T]
    return mx.moveaxis(y, 1, 2).astype(out_dtype), S


def _checkpointed(ops):
    """mlx-lm's own per-token loop, 64 tokens at a time under gradient checkpointing, so backprop keeps one state
    per 64 tokens instead of one per token. The fallback when chunked_gated_delta does not agree with mlx-lm."""
    def run(q, k, v, g, beta, state=None, mask=None):
        if state is None:
            state = mx.zeros((q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]), dtype=mx.float32)
        ys = []
        for s in range(0, q.shape[1], CHUNK):
            m = None if mask is None else mask[:, s:s + CHUNK]
            piece = mx.checkpoint(lambda *xs, m=m: ops(*xs, m))
            y, state = piece(*(x[:, s:s + CHUNK] for x in (q, k, v, g, beta)), state)
            ys.append(y)
        return mx.concatenate(ys, axis=1), state
    return run


def _disagreement(ops, candidate):
    """Worst relative difference between `candidate` and mlx-lm's loop, over both results and the gradients of
    all six inputs, on a problem with padding, repeated key heads, a starting state and decays close to zero."""
    B, T, Hk, Hv, Dk, Dv = 2, 150, 2, 4, 32, 16
    mx.random.seed(7)
    unit = lambda x: x / mx.sqrt((x * x).sum(-1, keepdims=True))      # the model L2-normalises q and k
    args = (unit(mx.random.normal((B, T, Hk, Dk))) * Dk ** -0.5, unit(mx.random.normal((B, T, Hk, Dk))),
            mx.random.normal((B, T, Hv, Dv)), mx.exp(-mx.exp(mx.random.normal((B, T, Hv)))),
            mx.sigmoid(mx.random.normal((B, T, Hv))), 0.1 * mx.random.normal((B, Hv, Dv, Dk)))
    wy, ws = mx.random.normal((B, T, Hv, Dv)), mx.random.normal((B, Hv, Dv, Dk))

    def loss(fn):
        def f(*x):
            y, s = fn(*x)
            return (y * wy).sum() + (s * ws).sum()
        return f

    def diff(a, b):
        return (mx.abs(a - b).max() / (mx.abs(a).max() + 1e-6)).item()

    worst = [diff(a, b) for a, b in zip(ops(*args), candidate(*args))]
    grads = [mx.grad(loss(fn), argnums=list(range(6)))(*args) for fn in (ops, candidate)]
    return max(worst + [diff(a, b) for a, b in zip(*grads)])


def install_recurrence():
    """Replace mlx-lm's per-token training loop for linear attention with a memory-light one: the chunked
    recurrence when it agrees with mlx-lm on this machine, mlx-lm's own loop under checkpointing otherwise.
    Returns what is in place, for the log."""
    try:
        from mlx_lm.models import gated_delta
        ops = gated_delta.gated_delta_ops
    except Exception as e:
        return "mlx-lm's own loop (there is no gated_delta_ops to replace: " + type(e).__name__ + ")"
    fallback, chunked = _checkpointed(ops), False
    try:
        err = _disagreement(ops, chunked_gated_delta)
        chunked = err < 1e-3
        how = ("chunked, agrees with mlx-lm to " + format(err, ".0e") if chunked else
               "chunked disagreed with mlx-lm (" + format(err, ".1e") + ")")
    except Exception as e:
        how = "chunked could not be checked (" + repr(e)[:160] + ")"

    def patched(q, k, v, g, beta, state=None, mask=None):
        CHUNK_CALLS[0] += 1
        if chunked and mask is None and g.ndim == 3:
            return chunked_gated_delta(q, k, v, g, beta, state)
        return fallback(q, k, v, g, beta, state, mask)

    gated_delta.gated_delta_ops = patched       # gated_delta_update looks it up by name on every call
    return how if chunked else "mlx-lm's loop under checkpointing, 64 tokens at a time - " + how


def _mem(name):
    """An MLX memory call by name: top-level in current MLX, under mx.metal in older releases."""
    return getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)


def _free_bytes():
    """Memory macOS could hand this process now: its free, inactive, purgeable and speculative pages."""
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1))
        pages = re.findall(r"Pages (?:free|inactive|purgeable|speculative):\s+(\d+)", vm)
        return page * sum(int(p) for p in pages) if pages else None
    except Exception:
        return None


def _memory_budget():
    """The bytes this run may use: what is free right now less 2 GB for everything else, and no more than the GPU
    working set. MLX otherwise goes by the working set alone - 27 GB on a 36 GB Mac, which on a busy machine is
    twice what is free - and keeps freed buffers cached up to it. The limit, and a cache a quarter its size,
    are set on MLX. Returns (budget or None when nothing could be read, what was read)."""
    info_fn = _mem("device_info")
    try:
        info = info_fn() if info_fn else {}
    except Exception:
        info = {}
    ram = int(info.get("memory_size") or 0)
    working = int(info.get("max_recommended_working_set_size") or 0) or int(ram * 0.75)
    free = _free_bytes()
    if free is None and not working:
        return None, {"ram": ram, "working": working, "free": free}
    budget = working if free is None else max(2 << 30, min(working or free, free - (2 << 30)))
    for name, value in (("set_memory_limit", budget), ("set_cache_limit", budget // 4)):
        fn = _mem(name)
        if fn:
            try:
                fn(value)
            except Exception:
                pass
    return budget, {"ram": ram, "working": working, "free": free}


def _memory_now():
    """GB in use, in MLX's buffer cache, and at the peak since the last call - which this call resets."""
    def get(name):
        fn = _mem(name)
        return round(fn() / 2 ** 30, 2) if fn else None
    now = (get("get_active_memory"), get("get_cache_memory"), get("get_peak_memory"))
    reset = _mem("reset_peak_memory")
    if reset:
        reset()
    return now


def _fit_length(measure, want, budget):
    """The longest sequence, at most `want` tokens, whose training step fits in `budget` bytes. Real steps are
    measured at 256 and 512 tokens, the line through them is extended to 85% of the budget, and the length that
    gives is measured too and stepped down until it fits. `measure(n)` returns a step's peak bytes, or inf when
    it ran out of memory. Returns (length, or 0 when not even the shortest fits; {length: peak bytes})."""
    peaks = {}
    lo, hi = min(want, 256), min(want, 512)
    for n in (lo, hi):
        if n not in peaks:
            peaks[n] = measure(n)
    if peaks[lo] > budget:
        return 0, peaks
    slope = (peaks[hi] - peaks[lo]) / (hi - lo) if hi > lo else 0.0
    guess = want if slope <= 0 else hi + int((0.85 * budget - peaks[hi]) / slope)
    n = want if guess >= want else max(lo, guess // 64 * 64)
    while True:
        if n not in peaks:
            peaks[n] = measure(n)
        if peaks[n] <= 0.9 * budget or n <= lo:
            return n, peaks
        n = max(lo, (n * 3 // 4) // 64 * 64)


def _ce_sum(logits, targets):
    logits = logits.astype(mx.float32)
    return (mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]).sum()


def _make_loss(model, sample):
    """The training loss - mean cross-entropy over an example's tokens - from the model's hidden states, with the
    vocabulary taken SLICE tokens at a time under checkpointing, so the full logits never exist at once. Falls
    back to the model's own logits when its output layer cannot be split off and shown to give the same logits
    on `sample`. Returns (loss(model, tokens), what it does, for the log)."""
    def whole(model, tokens):
        return _ce_sum(model(tokens[:, :-1]), tokens[:, 1:]) / (tokens.shape[1] - 1)

    try:
        inner = model["language_model"] if "language_model" in model else model     # Qwen3.5 wraps its text model
        body = inner["model"]
        head = inner["lm_head"] if "lm_head" in inner else body["embed_tokens"].as_linear
        model.eval()
        ref = model(sample).astype(mx.float32)
        diff = (mx.abs(ref - head(body(sample)).astype(mx.float32)).max() / (mx.abs(ref).max() + 1e-6)).item()
    except Exception as e:
        return whole, "the whole vocabulary at once (" + type(e).__name__ + " splitting off the output layer)"
    finally:
        model.train()
    if not diff < 1e-2:
        return whole, ("the whole vocabulary at once (the split-off output layer differed by " +
                       format(diff, ".0e") + ")")

    def sliced(model, tokens):
        targets = tokens[:, 1:]
        h = body(tokens[:, :-1])
        total = mx.array(0.0, dtype=mx.float32)
        for s in range(0, targets.shape[1], SLICE):
            part = mx.checkpoint(lambda x, t=targets[:, s:s + SLICE]: _ce_sum(head(x), t))
            total = total + part(h[:, s:s + SLICE])
        return total / targets.shape[1]
    return sliced, "the vocabulary " + str(SLICE) + " tokens at a time"


def _checkpoint_layers(model):
    """Gradient checkpointing for every decoder layer (they share one class): a layer's activations are recomputed
    during backprop instead of kept. mlx-lm's `grad_checkpoint`, restated so it does not hang on its trainer."""
    cls = type(model.layers[0])
    if getattr(cls, "_checkpointed", False):
        return
    call = cls.__call__

    def checkpointed(self, *args, **kwargs):
        def inner(params, *args, **kwargs):
            self.update(params)
            return call(self, *args, **kwargs)
        return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)

    cls.__call__ = checkpointed
    cls._checkpointed = True


def _encode(tokenizer, text):
    """Token ids for one training text, ending in the end-of-sequence token as mlx-lm's datasets do."""
    ids = list(tokenizer.encode(text))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not ids or ids[-1] != eos):
        ids.append(eos)
    return ids


def _lr_schedule(lr, updates, done):
    """The PyTorch recipe's schedule for MLX: linear warm-up over the first 5% of optimizer updates, then a
    cosine to zero. `done` is how many updates an interrupted run already made, so a resumed run continues
    the curve instead of warming up again."""
    warm = max(1, math.ceil(0.05 * updates))
    span = max(1, updates - warm)

    def at(step):                   # step: the optimizer's update counter, a uint64 array
        t = step.astype(mx.float32) + done
        cosine = 0.5 * lr * (1 + mx.cos(math.pi * mx.minimum(mx.maximum(t - warm, 0) / span, 1.0)))
        return mx.where(t < warm, lr * t / warm, cosine)
    return at


def _num(x):
    """A float out of whatever mlx-lm reports (a Python number or a 0-d array); None when it is absent."""
    try:
        return float(x.item() if hasattr(x, "item") else x)
    except (TypeError, ValueError):
        return None


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def train_mlx(a):
    """LoRA on the unified-memory GPU with MLX, in this process and in a training loop of its own (see above).
    Resumes from the last adapter it saved at the example it had reached, then merges the adapter into the base
    model's own files."""
    global mx
    import gc
    import mlx.core
    mx = mlx.core

    texts = [json.loads(l)["text"] for l in open(a.data, encoding="utf-8") if l.strip()]
    cut = max(1, int(len(texts) * 0.05))
    valid, train = texts[:cut], texts[cut:]
    emit("data", examples=len(texts), train=len(train), valid=len(valid))

    adapter = os.path.join(a.out, "adapter")
    os.makedirs(adapter, exist_ok=True)
    lora_params = {"rank": a.rank, "scale": a.alpha / a.rank, "dropout": a.dropout}
    # What a saved adapter must match to be continued. Sequence length and batch size may change between
    # attempts - finetune.py lowers them after running out of memory - so progress is counted in examples.
    recipe = {"lora": lora_params, "lr": a.lr, "epochs": a.epochs, "accum": a.accum, "train": len(train),
              "seed": a.seed}
    goal = math.ceil(len(train) * a.epochs)
    prog = _read_json(os.path.join(adapter, "progress.json"))
    if prog.get("recipe") != recipe:
        prog = {}
    final = os.path.join(adapter, "adapters.safetensors")
    if prog.get("finished") and os.path.exists(final):
        emit("resume", checkpoint="adapters.safetensors (training already finished)")
    else:
        _train_mlx(a, train, valid, adapter, lora_params, recipe, goal, prog)
        gc.collect()
        if _mem("clear_cache"):
            _mem("clear_cache")()

    if a.merge:
        out = os.path.join(a.out, "merged")
        scale = (_read_json(os.path.join(adapter, "adapter_config.json")).get("lora_parameters") or {}).get(
            "scale", lora_params["scale"])
        emit("merging", to=out)
        n = merge_lora_into_hf(a.base, final, float(scale), out)
        emit("merged", path=out, tensors=n)
    emit("done")


def _train_mlx(a, train, valid, adapter, lora_params, recipe, goal, prog):
    import numpy as np
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    import mlx_lm
    from mlx_lm.tuner.utils import linear_to_lora_layers

    budget, seen_mem = _memory_budget()             # before anything is allocated
    how = install_recurrence()
    if a.load_4bit:
        emit("warn", msg="4-bit base weights are not implemented for MLX; training in the base precision")

    model, tokenizer = mlx_lm.load(a.base)
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), lora_params)
    _checkpoint_layers(model)
    keep = os.path.join(adapter, "resume.safetensors")
    seen = int(prog.get("seen") or 0) if os.path.exists(keep) else 0
    if seen:
        model.load_weights(keep, strict=False)
    model.train()
    trainable = tree_flatten(model.trainable_parameters())
    emit("model", trainable=sum(v.size for _, v in trainable),
         total=sum(v.size for _, v in tree_flatten(model.parameters())),
         targets=sorted({k.split(".")[-2] for k, _ in trainable}))

    enc = [_encode(tokenizer, t) for t in train]
    venc = [ids for ids in (_encode(tokenizer, t) for t in valid[:25]) if len(ids) > 1]
    want = max(64, a.seq_len)
    stream = []                                     # real tokens to measure steps with
    for ids in enc:
        stream += ids
        if len(stream) > want:
            break
    loss_fn, loss_how = _make_loss(model, mx.array([stream[:16]], dtype=mx.int32))
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def measure(n):
        tokens = mx.array([(stream * (n // max(1, len(stream)) + 2))[:n + 1]], dtype=mx.int32)
        clear = _mem("clear_cache")
        if clear:
            clear()
        _memory_now()
        try:
            mx.eval(loss_and_grad(model, tokens))
        except RuntimeError as e:
            if not re.search(r"memory|malloc", str(e), re.I):
                raise
            return float("inf")
        finally:
            if clear:
                clear()
        return (_memory_now()[2] or 0) * 2 ** 30

    if budget:
        max_len, peaks = _fit_length(measure, want, budget)
    else:
        max_len, peaks = want, {}
    gb = lambda b: None if b is None or b == float("inf") else round(b / 2 ** 30, 1)
    emit("memory", ram_gb=gb(seen_mem["ram"]), working_set_gb=gb(seen_mem["working"]),
         free_gb=gb(seen_mem["free"]), budget_gb=gb(budget), measured={str(n): gb(p) for n, p in sorted(peaks.items())},
         max_len=max_len)
    if not max_len:
        raise SystemExit("out of memory: a training step needs " + str(gb(peaks[min(peaks)])) + " GB even at " +
                         str(min(peaks)) + " tokens and " + str(gb(budget)) + " GB is free for it - close other "
                         "apps and re-run")
    if max_len < want:
        emit("warn", msg="only " + str(max_len) + " of the configured " + str(want) + " tokens fit in the free "
                         "memory, so each example is cut there")
    cut = sum(len(ids) - 1 > max_len for ids in enc)
    if cut:
        emit("warn", msg=str(cut) + " of " + str(len(enc)) + " training examples are longer than " + str(max_len) +
                         " tokens; the rest of each is not trained on")
    per_update = max(1, a.batch) * max(1, a.accum)
    opt = optim.AdamW(learning_rate=_lr_schedule(a.lr, math.ceil(goal / per_update), seen // per_update),
                      weight_decay=0.01)
    emit("setup", device="mlx", steps=goal, gpu="Apple silicon (unified memory)", recurrence=how,
         loss=loss_how, max_len=max_len)
    if seen:
        emit("resume", checkpoint="resume.safetensors, " + str(seen) + " of " + str(goal) + " examples done")

    def save(path):
        tmp = path[:-len(".safetensors")] + ".partial.safetensors"    # MLX insists on the extension
        mx.save_safetensors(tmp, dict(tree_flatten(model.trainable_parameters())))
        os.replace(tmp, path)

    def validate(step):
        model.eval()
        total = count = 0
        for ids in venc:
            ids = ids[:max_len + 1]
            total += loss_fn(model, mx.array([ids], dtype=mx.int32)).item() * (len(ids) - 1)
            count += len(ids) - 1
        model.train()
        emit("val", step=step, loss=round(total / max(1, count), 4))

    every = max(25, a.save_steps, goal // 50)
    if not seen:
        validate(0)
    acc, acc_n, perm = None, 0, None
    for i in range(seen, goal):
        epoch, pos = divmod(i, len(enc))
        if perm is None or pos == 0:
            perm = np.random.default_rng(a.seed + epoch).permutation(len(enc))
        ids = enc[perm[pos]][:max_len + 1]
        loss, grads = loss_and_grad(model, mx.array([ids], dtype=mx.int32))
        if i == seen and not CHUNK_CALLS[0] and any(hasattr(layer, "linear_attn") for layer in model.layers):
            emit("warn", msg="the linear-attention recurrence was replaced but training never called it: this "
                             "mlx-lm reaches it another way, so its memory use is mlx-lm's")
        acc = grads if acc is None else tree_map(lambda x, y: x + y, acc, grads)
        acc_n += 1
        norm = None
        if acc_n == per_update or i + 1 == goal:
            grads, norm = optim.clip_grad_norm(tree_map(lambda x: x / acc_n, acc), 1.0)
            opt.update(model, grads)
            acc, acc_n = None, 0
        mx.eval([loss, model.trainable_parameters(), opt.state] + ([acc] if acc is not None else []))
        active, cached, peak = _memory_now()
        emit("step", step=i + 1, max_steps=goal, loss=round(loss.item(), 4), lr=_num(opt.learning_rate),
             tokens=len(ids) - 1, peak_gb=peak, active_gb=active, cache_gb=cached,
             grad_norm=None if norm is None else round(_num(norm), 4), epoch=round((i + 1) / len(enc), 3))
        if (i + 1) % max(1, a.save_steps) == 0 and i + 1 < goal:
            save(keep)
            _write_json(os.path.join(adapter, "progress.json"),
                        {"recipe": recipe, "seen": i + 1, "goal": goal, "max_len": max_len})
            emit("checkpoint", step=i + 1)
        if (i + 1) % every == 0 or i + 1 == goal:
            validate(i + 1)

    save(os.path.join(adapter, "adapters.safetensors"))
    _write_json(os.path.join(adapter, "adapter_config.json"),
                {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora_params,
                 "model": a.base})
    _write_json(os.path.join(adapter, "progress.json"),
                {"recipe": recipe, "seen": goal, "goal": goal, "max_len": max_len, "finished": True})
    if os.path.exists(keep):
        os.remove(keep)
    emit("trained", adapter=adapter, replaced_recurrence_calls=CHUNK_CALLS[0])


# ---------------------------------------------------------------- the adapter, merged into the base's own files
# `mlx_lm fuse` saves the model the way MLX holds it, and for Qwen3.5 that is not the Hugging Face layout: the
# loader transposes the conv1d weights, adds 1 to the RMSNorm weights and renames every key, so the GGUF
# converter cannot use what fuse writes. Merging into the original files keeps the layout exactly: a tensor
# the LoRA does not touch is copied byte for byte, and each adapted weight becomes W + scale (A B)^T.
_ST = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}      # numpy has no bfloat16, so its bits are handled by hand


def _st_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        head = json.loads(f.read(n))
    head.pop("__metadata__", None)
    return head, 8 + n


def _st_read(np, path, info, base):
    a, b = info["data_offsets"]
    dt = np.dtype(_ST[info["dtype"]])
    x = np.fromfile(path, dtype=dt, count=(b - a) // dt.itemsize, offset=base + a)
    if info["dtype"] == "BF16":
        x = (x.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float32).reshape(info["shape"])


def _st_bytes(np, x, dtype):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":                                 # round to nearest even, as a cast would
        u = x.view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()
    return x.astype(_ST[dtype]).tobytes()


def _hf_key(module, names):
    """The base-model weight a LoRA module adapts. MLX calls Qwen3.5's text model `language_model.model.*`
    where the checkpoint says `model.language_model.*`; most other models keep the checkpoint's names."""
    tries = [module]
    if module.startswith("language_model.model."):
        tries.append("model.language_model." + module[len("language_model.model."):])
    if module.startswith("language_model."):
        tries.append(module[len("language_model."):])
    for t in tries:
        if t + ".weight" in names:
            return t + ".weight"
    if "layers." not in module:
        return None
    tail = module[module.index("layers."):] + ".weight"
    hits = [n for n in names if (n == tail or n.endswith("." + tail)) and not n.startswith("mtp.")]
    return hits[0] if len(hits) == 1 else None


def merge_lora_into_hf(base_dir, adapter_file, scale, out_dir):
    """base_dir with the LoRA in adapter_file folded into its weights, written to out_dir with every other file
    of base_dir. Built in a side folder and moved into place, so a half-written merge never looks finished.
    Returns how many weights changed."""
    import numpy as np
    shards, where = {}, {}
    for f in sorted(os.listdir(base_dir)):
        if f.endswith(".safetensors"):
            head, shards[f] = _st_header(os.path.join(base_dir, f))
            where.update((name, (f, info)) for name, info in head.items())
    lhead, lbase = _st_header(adapter_file)
    pairs = {}
    for name, info in lhead.items():
        module, _, part = name.rpartition(".")
        if part not in ("lora_a", "lora_b"):
            raise SystemExit("unexpected tensor " + name + " in " + adapter_file)
        pairs.setdefault(module, {})[part] = info
    edits = {}
    for module, ab in sorted(pairs.items()):
        key = _hf_key(module, where)
        if key is None or len(ab) != 2:
            raise SystemExit("cannot find the base weight for the adapter's " + module)
        f, info = where[key]
        lo_a, lo_b = (_st_read(np, adapter_file, ab[p], lbase) for p in ("lora_a", "lora_b"))
        if list(info["shape"]) != [lo_b.shape[1], lo_a.shape[0]] or info["dtype"] not in _ST:
            raise SystemExit("the adapter's " + module + " does not fit " + key + " " + str(info["shape"]) +
                             " " + info["dtype"])
        edits.setdefault(f, []).append((info, scale * (lo_a @ lo_b).T))
    tmp = out_dir + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in os.listdir(base_dir):
        if os.path.isfile(os.path.join(base_dir, f)):
            shutil.copyfile(os.path.join(base_dir, f), os.path.join(tmp, f))
    for f, todo in edits.items():
        with open(os.path.join(tmp, f), "r+b") as fh:
            for info, delta in todo:
                start, end = info["data_offsets"]
                blob = _st_bytes(np, _st_read(np, os.path.join(base_dir, f), info, shards[f]) + delta,
                                 info["dtype"])
                if len(blob) != end - start:
                    raise SystemExit("merged tensor has the wrong size in " + f)
                fh.seek(shards[f] + start)
                fh.write(blob)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return sum(len(t) for t in edits.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    ap.add_argument("--base", required=True, help="HF-format base model directory")
    ap.add_argument("--data", required=True, help="jsonl with a 'text' field per example")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    backend = a.backend
    if backend == "auto":
        backend = "torch"
        try:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                import mlx.core  # noqa: F401
                backend = "mlx"
        except Exception:
            backend = "torch"
    try:
        (train_mlx if backend == "mlx" else train_torch)(a)
    except KeyboardInterrupt:
        emit("interrupted")
        sys.exit(130)
    except BaseException as e:
        emit("error", error=repr(e)[:500], kind_detail=type(e).__name__)
        raise


if __name__ == "__main__":
    main()
