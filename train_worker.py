#!/usr/bin/env python3
"""The actual training step. Runs inside the project venv (never in the stdlib-only orchestrator), so it may
import torch / peft / mlx. `finetune.py` starts it, reads its PROGRESS lines and restarts it on failure.

Two backends, chosen by the machine:
  cuda / cpu     - transformers + PEFT LoRA, bf16, gradient checkpointing, resumed from the last checkpoint
  apple silicon  - mlx-lm LoRA on the unified-memory GPU, in this process, resumed from the last adapter it
                   saved and merged into the base model's own Hugging Face files

Both print one JSON PROGRESS line per logging step so the dashboard can draw the loss curve. Everything here
is idempotent: killed at any point, the next start continues.
"""
import argparse, json, math, os, re, shutil, sys, time

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
# Most of Qwen3.5's layers are linear attention (Gated DeltaNet), and mlx-lm's fused Metal kernel for them has
# no backward pass. The model therefore asks for `use_kernel=not self.training`, and every training step goes
# through `gated_delta_ops`: a Python loop with one step per token. Backprop through that loop keeps two
# 16x128x128 float32 states per token per layer - over 4 GiB for one layer of a 2048-token example, gradient
# checkpointing or not - and builds tens of thousands of tiny ops. That ran a MacBook out of memory at step 6,
# at 20-70 s a step. `chunked_gated_delta` computes the same recurrence 64 tokens at a time with matrix
# products (the chunkwise form flash-linear-attention and transformers use), so backprop keeps one state per
# chunk. It replaces mlx-lm's loop only after agreeing with it, values and gradients, on the machine itself.

mx = None               # mlx.core, bound by train_mlx: MLX exists only on Apple silicon
CHUNK = 64
CHUNK_CALLS = [0]       # how many times training went through the chunked recurrence


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


def _check_chunked(ops):
    """Worst relative difference between chunked_gated_delta and mlx-lm's loop, over both results and the
    gradients of all six inputs, on a problem with padding, repeated key heads, a starting state and decays
    close to zero."""
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

    worst = [diff(a, b) for a, b in zip(ops(*args), chunked_gated_delta(*args))]
    grads = [mx.grad(loss(fn), argnums=list(range(6)))(*args) for fn in (ops, chunked_gated_delta)]
    return max(worst + [diff(a, b) for a, b in zip(*grads)])


def install_chunked_recurrence():
    """Swap mlx-lm's per-token training loop for chunked_gated_delta once the two agree here.
    Returns (installed, what to report)."""
    try:
        from mlx_lm.models import gated_delta
        ops = gated_delta.gated_delta_ops
    except Exception as e:
        return False, "this mlx-lm has no gated_delta_ops to replace (" + type(e).__name__ + ")"
    try:
        err = _check_chunked(ops)
    except Exception as e:
        return False, "the self-check could not run: " + repr(e)[:200]
    if not err < 1e-3:
        return False, "the self-check disagreed with mlx-lm (relative difference " + format(err, ".1e") + ")"

    def patched(q, k, v, g, beta, state=None, mask=None):
        if mask is not None or g.ndim != 3:     # padded batches and per-channel decay: mlx-lm's own loop
            return ops(q, k, v, g, beta, state, mask)
        CHUNK_CALLS[0] += 1
        return chunked_gated_delta(q, k, v, g, beta, state)

    gated_delta.gated_delta_ops = patched       # gated_delta_update looks it up by name on every call
    return True, "chunked (agrees with mlx-lm to " + format(err, ".0e") + ")"


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
    """mlx-lm's LoRA trainer on the unified-memory GPU, run in this process so the recurrence patch applies
    and progress arrives as numbers instead of console text. Resumes from the last adapter it saved, then
    merges it into the base model's own files."""
    global mx
    import gc
    import mlx.core
    mx = mlx.core

    data_dir = os.path.join(a.out, "mlx_data")
    os.makedirs(data_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    cut = max(1, int(len(rows) * 0.05))
    with open(os.path.join(data_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows[cut:]:
            f.write(json.dumps({"text": r["text"]}) + "\n")
    with open(os.path.join(data_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
        for r in rows[:cut]:
            f.write(json.dumps({"text": r["text"]}) + "\n")
    n_train = len(rows) - cut
    emit("data", examples=len(rows), train=n_train, valid=cut)

    adapter = os.path.join(a.out, "adapter")
    os.makedirs(adapter, exist_ok=True)
    lora_params = {"rank": a.rank, "scale": a.alpha / a.rank, "dropout": a.dropout}
    # What a saved adapter must match to be continued. Sequence length and batch size may change between
    # attempts - finetune.py lowers them after running out of memory - so progress is counted in examples.
    recipe = {"lora": lora_params, "lr": a.lr, "epochs": a.epochs, "accum": a.accum, "train": n_train,
              "seed": a.seed}
    goal = math.ceil(n_train * a.epochs)
    prog = _read_json(os.path.join(adapter, "progress.json"))
    if prog.get("recipe") != recipe:
        prog = {}
    final = os.path.join(adapter, "adapters.safetensors")
    if prog.get("finished") and os.path.exists(final):
        emit("resume", checkpoint="adapters.safetensors (training already finished)")
    else:
        _train_mlx(a, data_dir, adapter, lora_params, recipe, goal, n_train, prog)
        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()

    if a.merge:
        out = os.path.join(a.out, "merged")
        scale = (_read_json(os.path.join(adapter, "adapter_config.json")).get("lora_parameters") or {}).get(
            "scale", lora_params["scale"])
        emit("merging", to=out)
        n = merge_lora_into_hf(a.base, final, float(scale), out)
        emit("merged", path=out, tensors=n)
    emit("done")


def _train_mlx(a, data_dir, adapter, lora_params, recipe, goal, n_train, prog):
    import inspect, types
    import numpy as np
    import mlx_lm
    from mlx_lm import lora
    from mlx_lm.tuner import trainer
    from mlx_lm.tuner.datasets import load_dataset

    keep = os.path.join(adapter, "resume.safetensors")
    for f in os.listdir(adapter):           # a numbered save the last run died before recording
        if re.match(r"\d+_adapters\.safetensors$", f):
            os.remove(os.path.join(adapter, f))
    seen = int(prog.get("seen") or 0) if os.path.exists(keep) else 0
    per_update = a.batch * max(1, a.accum)
    steps_done = math.ceil(seen / a.batch)
    remaining = max(1, math.ceil((goal - seen) / a.batch))
    total = steps_done + remaining

    installed, how = install_chunked_recurrence()
    if not installed:
        emit("warn", msg="linear-attention layers train through mlx-lm's per-token loop, which is slow and "
                         "needs several GB per layer - " + how)
    if a.load_4bit:
        emit("warn", msg="4-bit base weights are not implemented for MLX; training in the base precision")
    emit("setup", device="mlx", steps=total, gpu="Apple silicon (unified memory)", recurrence=how)

    over = {"model": a.base, "train": True, "test": False, "data": data_dir, "fine_tune_type": "lora",
            "adapter_path": adapter, "batch_size": a.batch, "iters": remaining, "learning_rate": a.lr,
            "num_layers": -1, "lora_parameters": lora_params, "grad_accumulation_steps": a.accum,
            "optimizer": "adamw", "optimizer_config": {"adamw": {"weight_decay": 0.01}},
            "max_seq_length": a.seq_len, "grad_checkpoint": True, "steps_per_report": 1,
            "steps_per_eval": max(25, a.save_steps, remaining // 50), "save_every": a.save_steps,
            "seed": a.seed + seen, "resume_adapter_file": keep if seen else None}
    defaults = dict(getattr(lora, "CONFIG_DEFAULTS", {}))
    unread = [k for k in ("lora_parameters", "grad_accumulation_steps", "optimizer") if k not in defaults]
    if unread:
        emit("warn", msg="this mlx-lm ignores " + ", ".join(unread) + "; update it (.venv/bin/pip install -U "
                         "mlx-lm) to train the configured recipe")
    schedule = _lr_schedule(a.lr, math.ceil(goal / per_update), seen // per_update)
    if hasattr(lora, "build_schedule"):     # train_model builds the schedule through this name
        lora.build_schedule = lambda _config: schedule
        over["lr_schedule"] = {"name": "cosine", "warmup": 0, "arguments": [a.lr]}
    else:
        emit("warn", msg="this mlx-lm takes no learning-rate schedule; training at a constant " + str(a.lr))
    args = types.SimpleNamespace(**{**defaults, **over})

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    np.random.seed(args.seed)
    model, tokenizer = mlx_lm.load(a.base)
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    gdn = any(hasattr(layer, "linear_attn") for layer in getattr(model, "layers", []))

    class Report(getattr(trainer, "TrainingCallback", object)):
        tokens = 0.0

        def on_train_loss_report(self, info):
            it = int(info.get("iteration") or 0)
            if it == 1 and installed and gdn and not CHUNK_CALLS[0]:
                emit("warn", msg="the chunked recurrence is installed but training never called it: this "
                                 "mlx-lm reaches the recurrence another way, so memory stays high")
            done = _num(info.get("trained_tokens")) or 0.0
            emit("step", step=steps_done + it, max_steps=total, loss=_num(info.get("train_loss")),
                 lr=_num(info.get("learning_rate")), tokens=int(done - self.tokens),
                 peak_gb=_num(info.get("peak_memory")), epoch=round((seen + it * a.batch) / n_train, 3))
            self.tokens = done
            self.collect()

        def on_val_loss_report(self, info):
            emit("val", step=steps_done + int(info.get("iteration") or 0), loss=_num(info.get("val_loss")),
                 secs=_num(info.get("val_time")))

        def collect(self):
            """mlx-lm saves adapters.safetensors and a numbered copy every `save_every` steps. Keep the newest as
            resume.safetensors, with how many examples it has seen, and delete the numbered copies: at rank
            32 each is ~90 MB and a run writes hundreds."""
            saved = sorted((int(m.group(1)), m.group(0)) for m in
                           (re.match(r"(\d+)_adapters\.safetensors$", f) for f in os.listdir(adapter)) if m)
            if not saved:
                return
            shutil.copyfile(os.path.join(adapter, saved[-1][1]), keep + ".tmp")
            os.replace(keep + ".tmp", keep)
            _write_json(os.path.join(adapter, "progress.json"),
                        {"recipe": recipe, "seen": seen + saved[-1][0] * a.batch, "goal": goal})
            for _, name in saved:
                os.remove(os.path.join(adapter, name))
            emit("checkpoint", step=steps_done + saved[-1][0])

    if seen:
        emit("resume", checkpoint="resume.safetensors, " + str(seen) + " of " + str(goal) + " examples done")
    extra = [tokenizer] if "tokenizer" in inspect.signature(lora.train_model).parameters else []
    lora.train_model(args, model, *extra, train_set, valid_set, Report())
    for f in os.listdir(adapter):
        if re.match(r"\d+_adapters\.safetensors$", f):
            os.remove(os.path.join(adapter, f))
    _write_json(os.path.join(adapter, "progress.json"),
                {"recipe": recipe, "seen": goal, "goal": goal, "finished": True})
    emit("trained", adapter=adapter, chunked_calls=CHUNK_CALLS[0])


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
