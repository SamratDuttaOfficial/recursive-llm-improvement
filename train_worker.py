#!/usr/bin/env python3
"""The actual training step. Runs inside the project venv (never in the stdlib-only orchestrator), so it may
import torch / peft / mlx. `finetune.py` starts it, reads its PROGRESS lines and restarts it on failure.

Two backends, chosen by the machine:
  cuda / cpu     - transformers + PEFT LoRA, bf16, gradient checkpointing, resumed from the last checkpoint
  apple silicon  - mlx-lm LoRA on the unified-memory GPU, resumed from the last adapter file

Both write checkpoints into <out>/ckpt and print one JSON PROGRESS line per logging step so the dashboard can
draw the loss curve. Everything here is idempotent: killed at any point, the next start continues.
"""
import argparse, json, math, os, sys, time

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
def train_mlx(a):
    """mlx-lm's LoRA trainer on the unified-memory GPU. Resumes from the adapter it last wrote."""
    import subprocess
    from mlx_lm import convert as mlx_convert   # noqa: F401  (import proves the package works)

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
    emit("data", examples=len(rows), train=len(rows) - cut, valid=cut)

    adapter = os.path.join(a.out, "adapter")
    os.makedirs(adapter, exist_ok=True)
    steps = max(30, math.ceil(len(rows) * a.epochs / max(1, a.batch)))
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--model", a.base, "--train",
           "--data", data_dir, "--adapter-path", adapter, "--batch-size", str(a.batch),
           "--iters", str(steps), "--learning-rate", str(a.lr), "--num-layers", "-1",
           "--steps-per-report", "1", "--steps-per-eval", str(max(25, a.save_steps)),
           "--save-every", str(a.save_steps), "--max-seq-length", str(a.seq_len),
           "--grad-checkpoint"]
    if os.path.exists(os.path.join(adapter, "adapters.safetensors")):
        cmd += ["--resume-adapter-file", os.path.join(adapter, "adapters.safetensors")]
        emit("resume", checkpoint="adapters.safetensors")
    emit("setup", device="mlx", steps=steps, gpu="Apple silicon (unified memory)")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, errors="replace")
    import re
    pat = re.compile(r"Iter (\d+):.*?Train loss ([\d.]+)", re.I)
    for line in p.stdout:
        line = line.rstrip()
        m = pat.search(line)
        if m:
            emit("step", step=int(m.group(1)), max_steps=steps, loss=float(m.group(2)))
        elif line.strip():
            emit("log", msg=line[:300])
    if p.wait() != 0:
        raise SystemExit("mlx_lm lora exited with " + str(p.returncode))
    emit("trained", adapter=adapter)

    if a.merge:
        out = os.path.join(a.out, "merged")
        emit("merging", to=out)
        r = subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", a.base,
                            "--adapter-path", adapter, "--save-path", out,
                            "--de-quantize"], text=True)
        if r.returncode != 0:
            raise SystemExit("mlx_lm fuse failed")
        emit("merged", path=out)
    emit("done")


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
