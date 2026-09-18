#!/usr/bin/env python3
"""Shared machinery for the recursive self-improvement loop: logging, the event feed the dashboard reads,
downloads, cross-platform GPU probing, resumable state, graceful shutdown, the llama-server / Ollama backends,
the model registry, the Python code checker and the dashboard HTTP server.

Standard library only. Heavy dependencies (torch, mlx, pyarrow, ...) live in a project-local venv that
`Venv` creates and fills on demand, so nothing ever has to be installed by hand.
"""
import atexit, itertools, json, os, platform, re, shutil, signal, subprocess, sys, tarfile, threading, time
import urllib.error, urllib.parse, urllib.request, webbrowser, zipfile
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
STATE = os.path.join(ROOT, "state")
LOGS = os.path.join(ROOT, "logs")
DASH_HTML = os.path.join(ROOT, "dashboard.html")
for _d in (DATA, MODELS, STATE, LOGS):
    os.makedirs(_d, exist_ok=True)

UA = {"User-Agent": "RecursiveLLMImprovement/1.0 (research; python-urllib)"}
WIN = platform.system() == "Windows"
MAC = platform.system() == "Darwin"
ARM_MAC = MAC and platform.machine() in ("arm64", "aarch64")
LOCK = threading.Lock()
T0 = time.time()
STATS = Counter()           # tokens in/out, calls (under LOCK)
LIVE = {}                   # gen id -> Gen: every model call in flight, streamed by the dashboard
RECENT = deque(maxlen=120)  # finished Gens, so a call that just ended is still visible
EVENTS, EVSEQ = deque(maxlen=4000), itertools.count(1)
PHASE = {"name": "idle", "detail": ""}   # what the process is doing right now (shown in the UI)


# ---------------------------------------------------------------- logging + event feed
class Log:
    color, verbose = False, False
    C = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "magenta": "35", "dim": "2", "bold": "1"}

    @classmethod
    def setup(cls, color=True, verbose=False):
        cls.color = color and sys.stdout.isatty()
        cls.verbose = verbose
        if cls.color and WIN:
            os.system("")  # turn on ANSI escapes in the Windows console
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    @classmethod
    def paint(cls, text, *styles):
        if not cls.color or not styles:
            return str(text)
        return "".join(f"\033[{cls.C[s]}m" for s in styles) + str(text) + "\033[0m"

    @classmethod
    def line(cls, tag, msg, *styles, **data):
        event(tag, msg, **data)
        ts = time.strftime("%H:%M:%S")
        with LOCK:
            print(ts + " " + cls.paint("[" + tag.ljust(9) + "]", "dim") + " " + cls.paint(msg, *styles), flush=True)

    @classmethod
    def block(cls, lines):
        with LOCK:
            print("\n".join(lines), flush=True)


def event(kind, msg, **data):
    """Append to the dashboard's event feed (deque.append / itertools.count are atomic)."""
    EVENTS.append({"seq": next(EVSEQ), "t": round(time.time(), 1), "kind": kind, "msg": msg, **data})


def log(tag, msg, *styles, **data):
    Log.line(tag, msg, *styles, **data)


def vlog(tag, msg, *styles, **data):
    if Log.verbose:
        Log.line(tag, msg, *styles, **data)


def phase(name, detail=""):
    PHASE.update(name=name, detail=detail)
    event("phase", (name + " " + detail).strip(), phase=name)


def fmt_t(secs):
    secs = int(max(0, secs))
    if secs < 60:
        return str(secs) + "s"
    if secs < 3600:
        return str(secs // 60) + "m" + str(secs % 60).zfill(2) + "s"
    return str(secs // 3600) + "h" + str(secs % 3600 // 60).zfill(2) + "m"


def fmt_mb(got, total):
    pct = str(round(100 * got / total)).rjust(3) + "%" if total else "   ?"
    return pct + "  " + format(got / 1e6, "6.0f") + "/" + format(total / 1e6, ".0f") + " MB"


def stamp(t=None):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t or time.time()))


# ---------------------------------------------------------------- graceful shutdown
class Stop:
    """One process-wide stop flag. Ctrl-C sets it; every loop checks it and returns, every step has already
    written its state, so re-running resumes exactly where it stopped. A second Ctrl-C exits immediately."""
    flag = threading.Event()
    _cleanup = []
    _hits = 0

    @classmethod
    def set(cls):
        cls.flag.set()

    @classmethod
    def is_set(cls):
        return cls.flag.is_set()

    @classmethod
    def sleep(cls, secs):
        """Interruptible sleep. True when we were asked to stop."""
        return cls.flag.wait(secs)

    @classmethod
    def on_exit(cls, fn):
        cls._cleanup.append(fn)

    @classmethod
    def run_cleanup(cls):
        for fn in reversed(cls._cleanup):
            try:
                fn()
            except Exception:
                pass
        cls._cleanup.clear()

    @classmethod
    def install(cls):
        def handler(*_):
            cls._hits += 1
            if cls._hits == 1:
                cls.set()
                log("stop", "Ctrl-C: finishing the step in flight and saving, then exiting; re-run to resume "
                            "(Ctrl-C again quits immediately)", "yellow")
            else:
                log("stop", "second Ctrl-C: quitting now", "yellow")
                cls.run_cleanup()
                os._exit(130)
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handler)
        atexit.register(cls.run_cleanup)


class Interrupted(Exception):
    """Raised inside a worker when a stop was requested: the caller saves nothing new and returns."""


def check_stop():
    if Stop.is_set():
        raise Interrupted()


# ---------------------------------------------------------------- http + downloads
def http(url, data=None, timeout=600, headers=None):
    body = json.dumps(data).encode() if data is not None else None
    h = dict(UA)
    h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(url, dest, quiet=False):
    """An existing destination is kept as is; a partial download is discarded and restarted."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if not quiet:
        log("download", url)
    tmp = dest + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r, open(tmp, "wb") as f:
        total, got, t = int(r.headers.get("Content-Length") or 0), 0, time.time()
        while True:
            if Stop.is_set():
                f.close()
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise Interrupted()
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if not quiet and time.time() - t > 5:
                t = time.time()
                log("download", os.path.basename(dest).ljust(34) + " " + fmt_mb(got, total))
    os.replace(tmp, dest)
    if not quiet:
        log("download", os.path.basename(dest).ljust(34) + " done (" + format(got / 1e6, ".1f") + " MB)")
    return dest


def hf_url(repo, path, kind="datasets"):
    base = "https://huggingface.co/datasets/" if kind == "datasets" else "https://huggingface.co/"
    return base + repo + "/resolve/main/" + path


# ---------------------------------------------------------------- GPU probing (NVIDIA / AMD / Apple)
_GPU_CACHE = {"t": 0.0, "v": None}


def _nvidia():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        name, u, m, t = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"kind": "cuda", "name": name, "util": int(float(u)), "used": int(float(m)), "total": int(float(t))}
    except Exception:
        return None


def _amd():
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--showuse", "--json"],
                             capture_output=True, text=True, timeout=5).stdout
        j = json.loads(out)
        c = j[sorted(j)[0]]
        used = int(c.get("VRAM Total Used Memory (B)", 0)) // 2 ** 20
        total = int(c.get("VRAM Total Memory (B)", 0)) // 2 ** 20
        return {"kind": "rocm", "name": "AMD GPU", "util": int(float(c.get("GPU use (%)", 0))),
                "used": used, "total": total}
    except Exception:
        return None


def _apple():
    """Apple silicon shares one pool of memory with the GPU. `total` is the working set Metal may use
    (~70% of RAM, which is what the default wired limit works out to) and `used` comes from vm_stat."""
    if not ARM_MAC:
        return None
    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
        total_mb = mem // 2 ** 20
        budget = int(total_mb * 0.70)
        free_mb = budget
        try:
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", vm).group(1))
            pages = lambda k: int(re.search(k + r":\s+(\d+)", vm).group(1))
            free_pages = pages("Pages free") + pages("Pages inactive") + pages("Pages purgeable")
            free_mb = min(budget, free_pages * page // 2 ** 20)
        except Exception:
            pass
        name = "Apple silicon GPU"
        try:
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or name
        except Exception:
            pass
        return {"kind": "metal", "name": name, "util": -1, "used": budget - free_mb, "total": budget,
                "unified": True, "ram": total_mb}
    except Exception:
        return None


def gpu_status(max_age=1.0):
    """{kind, name, util, used, total} in MB, or None. Cached for a second: many threads poll it."""
    now = time.time()
    if now - _GPU_CACHE["t"] < max_age:
        return _GPU_CACHE["v"]
    v = _apple() or _nvidia() or _amd()
    _GPU_CACHE.update(t=now, v=v)
    return v


def gpu_free_mb():
    g = gpu_status()
    return max(0, g["total"] - g["used"]) if g else 0


# ---------------------------------------------------------------- resumable state
def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    """Atomic: neither a crash nor a Ctrl-C can leave a half-written state file behind."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp" + str(os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


_APPEND = threading.Lock()


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _APPEND:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


class Store:
    """The state of one run: a directory of small JSON files, each written atomically the moment it changes.
    Everything the phases need in order to resume lives here and nowhere else."""

    def __init__(s, kind, run="default"):
        s.dir = os.path.join(STATE, run, kind)
        os.makedirs(s.dir, exist_ok=True)
        s.run, s.kind = run, kind
        s._cache, s._lock = {}, threading.Lock()

    def path(s, name):
        return os.path.join(s.dir, name + ".json")

    def get(s, name, default=None):
        with s._lock:
            if name in s._cache:
                return s._cache[name]
            v = read_json(s.path(name), default)
            s._cache[name] = v
            return v

    def put(s, name, obj):
        with s._lock:
            s._cache[name] = obj
            write_json(s.path(name), obj)
        return obj

    def has(s, name):
        with s._lock:
            return name in s._cache or os.path.exists(s.path(name))

    def names(s):
        return sorted(f[:-5] for f in os.listdir(s.dir) if f.endswith(".json"))


# ---------------------------------------------------------------- one model call in flight
class Gen:
    """A single streaming generation. The dashboard reads `parts` live, so the user can watch every agent and
    judge write its answer. A cheap degeneration guard stops a call that has started repeating itself."""
    ids = itertools.count(1)
    REPEAT_SPAN, REPEAT_COPIES, CHECK_EVERY = 60, 3, 200

    def __init__(s, tag="", kind="call", max_tokens=0, model="", meta=None, guard=True):
        s.id = next(Gen.ids)
        s.tag, s.kind, s.max, s.model, s.meta = tag, kind, max_tokens, model, meta or {}
        s.t0, s.t1, s.t_end = time.time(), None, None
        s.guard, s.looped = guard, False
        s.parts, s.n, s.last = [], 0, 0
        s.err = None
        with LOCK:
            LIVE[s.id] = s

    def reset(s):
        s.parts, s.n, s.last, s.t1, s.looped = [], 0, 0, None, False

    def text(s):
        return "".join(s.parts)

    def add(s, piece):
        """Append one streamed chunk. True means: stop generating, this call is going in circles."""
        if not piece:
            return False
        s.parts.append(piece)
        s.n += 1
        if s.t1 is None:
            s.t1 = time.time()
        if Stop.is_set():
            return True
        if not s.guard or s.n - s.last < s.CHECK_EVERY:
            return False
        s.last = s.n
        words = re.findall(r"[A-Za-z0-9_]+", s.text().lower())
        if len(words) < s.REPEAT_SPAN * s.REPEAT_COPIES:
            return False
        tail = " ".join(words[-s.REPEAT_SPAN:])
        if " ".join(words).count(tail) >= s.REPEAT_COPIES:
            s.looped = True
            return True
        return False

    def done(s, err=None):
        s.t_end, s.err = time.time(), err
        with LOCK:
            LIVE.pop(s.id, None)
            RECENT.append(s)

    def speed(s):
        dt = (s.t_end or time.time()) - (s.t1 or s.t0)
        return s.n / dt if dt > 0.05 else 0.0

    def brief(s):
        return {"id": s.id, "tag": s.tag, "kind": s.kind, "model": s.model, "n": s.n, "max": s.max,
                "t0": round(s.t0, 1), "end": round(s.t_end, 1) if s.t_end else None,
                "tps": round(s.speed(), 1), "looped": s.looped, "err": s.err, **s.meta}


# ---------------------------------------------------------------- Ollama: model downloads + fallback backend
class Ollama:
    """Used for two things: pulling the GGUF weights (it is the simplest reliable model source on every
    platform) and, when llama-server cannot be used, serving them itself."""

    def __init__(s, port, ctx):
        s.port, s.ctx = port, ctx
        s.url = "http://127.0.0.1:" + str(port)
        s.proc, s.exe = None, None

    def binary(s):
        if s.exe:
            return s.exe
        b = shutil.which("ollama")
        if b:
            s.exe = b
            return b
        d = os.path.join(ROOT, "ollama_bin")
        os.makedirs(d, exist_ok=True)
        rel = "https://github.com/ollama/ollama/releases/latest/download/"
        if WIN:
            exe = os.path.join(d, "ollama.exe")
            if not os.path.exists(exe):
                zipfile.ZipFile(download(rel + "ollama-windows-amd64.zip", os.path.join(d, "ollama.zip"))).extractall(d)
        else:
            name = "ollama-darwin.tgz" if MAC else "ollama-linux-amd64.tgz"
            exe = os.path.join(d, "bin", "ollama")
            if not os.path.exists(exe):
                tarfile.open(download(rel + name, os.path.join(d, name))).extractall(d)
                if not os.path.exists(exe):
                    exe = os.path.join(d, "ollama")
                os.chmod(exe, 0o755)
        s.exe = exe
        return exe

    def alive(s):
        try:
            http(s.url + "/api/version", timeout=3)
            return True
        except Exception:
            return False

    def start(s):
        s.binary()
        if s.alive():
            return True
        env = dict(os.environ)
        env.update({"OLLAMA_HOST": "127.0.0.1:" + str(s.port), "OLLAMA_NUM_PARALLEL": "1",
                    "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_FLASH_ATTENTION": "1",
                    "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_MAX_QUEUE": "4096"})
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
        lf = open(os.path.join(LOGS, "ollama_server.log"), "ab")
        s.proc = subprocess.Popen([s.exe, "serve"], env=env, stdout=lf, stderr=lf, **kw)
        for _ in range(90):
            if s.alive():
                return True
            if Stop.is_set():
                return False
            time.sleep(1)
        log("warn", "ollama server did not start; see logs/ollama_server.log", "yellow")
        return False

    def stop(s):
        kill_tree(s.proc)
        s.proc = None

    def tags(s):
        try:
            return json.loads(http(s.url + "/api/tags")).get("models", [])
        except Exception:
            return []

    def pull(s, model):
        names = [m.get("name", "") for m in s.tags()]
        if model in names or (model + ":latest") in names:
            log("model", "weights for " + model + " already present")
            return
        log("model", "pulling " + model + " (first run only)")
        req = urllib.request.Request(s.url + "/api/pull",
                                     data=json.dumps({"model": model, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            t = 0.0
            for line in r:
                if Stop.is_set():
                    raise Interrupted()
                j = json.loads(line)
                if "error" in j:
                    sys.exit("pull failed: " + str(j["error"]))
                if j.get("total") and time.time() - t > 5:
                    t = time.time()
                    log("model", model.ljust(28) + " " + fmt_mb(j.get("completed", 0), j["total"]))
        log("model", "pulled " + model)

    def chat(s, model, system, user, max_tokens, temp, gen=None, sampler=None):
        body = {"model": model, "stream": True, "keep_alive": -1, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"num_predict": max_tokens, "temperature": temp, "num_ctx": s.ctx,
                            **{k: v for k, v in (sampler or {}).items() if k in ("top_k", "top_p", "min_p")}}}
        g = gen or Gen(max_tokens=max_tokens, model=model)
        for attempt in range(20):
            check_stop()
            g.reset()
            try:
                req = urllib.request.Request(s.url + "/api/chat", data=json.dumps(body).encode(),
                                             headers={**UA, "Content-Type": "application/json"})
                nin = nout = None
                with urllib.request.urlopen(req, timeout=3600) as r:
                    for line in r:
                        j = json.loads(line)
                        if "error" in j:
                            raise RuntimeError(str(j["error"])[:200])
                        if g.add(j.get("message", {}).get("content", "")):
                            break
                        if j.get("done"):
                            nin, nout = j.get("prompt_eval_count"), j.get("eval_count")
                            break
                nin = nin or (len(system) + len(user)) // 4
                nout = nout or g.n
                with LOCK:
                    STATS["in"] += nin
                    STATS["out"] += nout
                    STATS["calls"] += 1
                text = re.sub(r"<think>.*?</think>", "", g.text(), flags=re.S).strip()
                return text, {"in": nin, "out": nout, "looped": g.looped}
            except Interrupted:
                raise
            except Exception as e:
                if Stop.is_set():
                    raise Interrupted()
                log("warn", "ollama call failed: " + repr(e)[:120] + "  (retry " + str(attempt + 1) + ")", "yellow")
                if Stop.sleep(min(20, 3 * (attempt + 1))):
                    raise Interrupted()
                if not s.alive():
                    s.stop()
                    s.start()
        raise RuntimeError("ollama chat failed repeatedly")


def kill_tree(proc):
    """Kill a server and every process it started (the GPU runner is a child)."""
    if not proc or proc.poll() is not None:
        return
    try:
        if WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(30):
                if proc.poll() is not None:
                    return
                time.sleep(0.1)
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------- llama-server: one process, N parallel slots
def ollama_model_files(model):
    """The GGUF blob and default sampling parameters of a model Ollama has pulled."""
    root = os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")
    name, _, tag = model.partition(":")
    parts = name.split("/")
    if len(parts) == 1:
        path = ["registry.ollama.ai", "library", parts[0]]
    elif len(parts) == 2:
        path = ["registry.ollama.ai"] + parts
    else:
        path = parts
    mf = os.path.join(root, "manifests", *path, tag or "latest")
    if not os.path.exists(mf):
        return None
    blob = lambda d: os.path.join(root, "blobs", d.replace(":", "-"))
    out = {"model": None, "params": {}}
    try:
        layers = json.load(open(mf))["layers"]
    except Exception:
        return None
    for l in layers:
        if l["mediaType"].endswith("image.model"):
            out["model"] = blob(l["digest"])
        elif l["mediaType"].endswith("image.params"):
            try:
                out["params"] = json.load(open(blob(l["digest"])))
            except Exception:
                pass
    return out if out["model"] and os.path.exists(out["model"]) else None


def gpu_backend_libs(libdir):
    """GPU backend libraries shipped with Ollama, best first: newest CUDA, then ROCm, then Metal, then Vulkan.
    On Apple silicon Metal is compiled into the binary, so an empty list there simply means 'no extra library'."""
    if not os.path.isdir(libdir):
        return []

    def key(sub):
        m = re.search(r"(\d+)", sub)
        rank = 0 if "cuda" in sub else 1 if ("rocm" in sub or "hip" in sub) else 2 if "metal" in sub else 3
        return (rank, -int(m.group(1)) if m else 0)

    out = []
    for sub in sorted((d for d in os.listdir(libdir) if os.path.isdir(os.path.join(libdir, d))), key=key):
        for f in os.listdir(os.path.join(libdir, sub)):
            if re.match(r"(lib)?ggml-(cuda|hip|rocm|vulkan|metal)\.(dll|so|dylib)$", f):
                out.append(os.path.join(libdir, sub, f))
    return out


class LlamaServer:
    """Ollama's own llama-server binary, run directly with N parallel slots so the whole GPU stays busy
    (Ollama itself serves this architecture one request at a time). One instance serves one model; the
    Backend below runs one instance per model when the generator and the judges differ.

    Works the same on CUDA, ROCm, Vulkan and Apple-silicon Metal: the slot count is fitted to the memory
    actually free, which on a Mac is a slice of unified RAM."""
    MAX_SLOTS = 32

    def __init__(s, port, ctx, want_slots, gguf, params, ollama_exe, name="model"):
        s.port, s.ctx, s.want, s.name = port, ctx, want_slots, name
        s.gguf, s.params = gguf, params or {}
        s.url = "http://127.0.0.1:" + str(port)
        s.proc, s.slots, s.inflight, s.lock, s.crashes = None, want_slots or 4, 0, threading.Lock(), []
        s.ngl = 999
        s.libdir = os.path.join(os.path.dirname(os.path.realpath(ollama_exe)), "lib", "ollama") if ollama_exe else ""
        exe = "llama-server.exe" if WIN else "llama-server"
        s.exe = os.path.join(s.libdir, exe) if s.libdir else ""
        if s.exe and not os.path.exists(s.exe):
            alt = shutil.which(exe)
            s.exe = alt or s.exe
        s.backend, s.logf = None, os.path.join(LOGS, "llama_server.log")

    def available(s):
        return bool(s.exe) and os.path.exists(s.exe) and s.gguf and os.path.exists(s.gguf)

    def _alive(s):
        try:
            return urllib.request.urlopen(s.url + "/health", timeout=3).status == 200
        except Exception:
            return False

    def _launch(s, backend, ngl=999):
        libs = [s.libdir] + ([os.path.dirname(backend)] if backend else [])
        env = dict(os.environ)
        var = "PATH" if WIN else ("DYLD_LIBRARY_PATH" if MAC else "LD_LIBRARY_PATH")
        env[var] = os.pathsep.join([p for p in libs if p] + [env.get(var, "")])
        if backend:
            env["GGML_BACKEND_PATH"] = backend
        s.ngl = ngl
        args = [s.exe, "--model", s.gguf, "--host", "127.0.0.1", "--port", str(s.port), "--no-webui", "--offline",
                "-c", str(s.slots * s.ctx), "-np", str(s.slots), "-ngl", str(ngl), "--no-jinja",
                "--chat-template", "chatml", "--flash-attn", "on", "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0", "-b", "1024", "-ub", "1024", "--no-log-prefix", "--no-log-timestamps"]
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WIN else {"start_new_session": True}
        lf = open(s.logf, "ab")
        s.proc = subprocess.Popen(args, env=env, stdout=lf, stderr=lf, **kw)
        for _ in range(240):
            if s._alive():
                return True
            if s.proc.poll() is not None or Stop.is_set():
                break
            time.sleep(1)
        s.stop()
        return False

    def stop(s):
        kill_tree(s.proc)
        s.proc = None

    def start(s):
        return s._launch(s.backend, s.ngl)

    def setup(s):
        """Pick a GPU backend that actually puts the weights on the GPU, then grow the slot count until the
        free GPU memory is used up. Returns False when no backend works (the caller falls back to Ollama)."""
        weights = os.path.getsize(s.gguf) // 2 ** 20
        g0 = gpu_status(max_age=0)
        candidates = gpu_backend_libs(s.libdir) or [None]   # None = whatever is compiled in (Metal, CPU)
        for lib in candidates:
            if Stop.is_set():
                return False
            s.slots = s.want or s._guess_slots(weights)
            ok = s._launch(lib)
            while not ok and s.slots > 1:
                s.slots = max(1, s.slots // 2)
                ok = s._launch(lib)
            if not ok:
                continue
            g1 = gpu_status(max_age=0)
            on_gpu = not (g0 and g1) or g1["used"] - g0["used"] >= weights // 3 or g1.get("unified")
            if not on_gpu:
                log("warn", (os.path.basename(os.path.dirname(lib)) if lib else "built-in") +
                    ": weights did not land on the GPU, trying the next backend", "yellow")
                s.stop()
                continue
            s.backend = lib
            used = (g1["used"] - g0["used"]) if (g0 and g1 and not g1.get("unified")) else None
            where = os.path.basename(os.path.dirname(lib)) if lib else (
                "metal (built in)" if ARM_MAC else "built-in")
            log("llama", s.name + ": backend " + where + ", " + str(s.slots) + " slots x ctx " + str(s.ctx) +
                (", " + str(used) + " MB VRAM (" + str(weights) + " MB weights)" if used else
                 " (" + str(weights) + " MB weights)"))
            s._grow(lib, weights, g1)
            return True
        # Nothing would load on the GPU. Usually that means the memory is taken by something else (another
        # run, a game, a browser). Run on the CPU rather than giving up: slow, but the loop still makes
        # progress, and re-running once the GPU is free picks the GPU up again.
        free = gpu_free_mb()
        log("warn", "no GPU backend could load the model" +
            (" - only " + str(free) + " MB of GPU memory is free, the weights need " + str(weights) +
             " MB (something else is using the GPU)" if free < weights else "") +
            "; falling back to the CPU", "yellow")
        s.slots = s.want or max(2, min(8, (os.cpu_count() or 4) // 2))
        if s._launch(None, ngl=0):
            log("llama", s.name + ": running on the CPU, " + str(s.slots) + " slots x ctx " + str(s.ctx) +
                " (free the GPU and re-run to go back to it)", "yellow")
            return True
        return False

    def kv_mb_per_slot(s):
        """Rough KV-cache cost of one slot at this context, with q8_0 keys and values. Only an opening
        guess - `_grow` measures the real figure once the model is loaded - but it stops a 64k-context judge
        server from optimistically asking for eight slots and failing to allocate eight times over."""
        return max(24.0, 0.03 * s.ctx)

    def _guess_slots(s, weights):
        free = gpu_free_mb() or (weights * 4)
        room = (free - weights - 700) / s.kv_mb_per_slot()
        return max(1, min(8, int(room)))

    def _grow(s, lib, weights, g1):
        """Add slots while there is memory for them: every extra slot only costs its share of the KV cache."""
        if s.want or not g1:
            return
        free = max(0, g1["total"] - g1["used"])
        margin = 1200 if g1.get("unified") else 700
        per_slot = s.kv_mb_per_slot()
        room = int((free - margin) / per_slot)
        target = max(1, min(s.MAX_SLOTS, s.slots + max(0, room)))
        if target <= s.slots:
            return
        old = s.slots
        s.stop()
        s.slots = target
        if s._launch(lib):
            g2 = gpu_status(max_age=0)
            log("llama", s.name + ": grew to " + str(s.slots) + " slots (~" + str(round(per_slot)) +
                " MB each)" + (" - GPU " + str(g2["used"]) + "/" + str(g2["total"]) + " MB" if g2 else ""))
        else:
            s.slots = old
            s._launch(lib)
            log("llama", s.name + ": could not grow, staying at " + str(s.slots) + " slots")

    def status(s):
        with s.lock:
            return str(s.inflight) + "/" + str(s.slots) + " slots busy"

    def _ensure(s):
        with s.lock:
            if s._alive() or Stop.is_set():
                return
            s.crashes = [t for t in s.crashes if time.time() - t < 600] + [time.time()]
            if len(s.crashes) >= 3 and s.slots > 1:
                s.slots = max(1, s.slots // 2)
                s.crashes = []
                log("warn", s.name + ": llama-server died 3 times in 10 min, halving to " +
                    str(s.slots) + " slots", "yellow")
            log("warn", s.name + ": llama-server is down, restarting", "yellow")
            s.stop()
            s.start()

    def count(s, text):
        try:
            return len(json.loads(http(s.url + "/tokenize", {"content": text}, timeout=60)).get("tokens", []))
        except Exception:
            return len(text) // 4

    def chat(s, model, system, user, max_tokens, temp, gen=None, sampler=None):
        """Streams into `gen`; closing the connection makes llama-server free the slot immediately."""
        prompt = ("<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user +
                  "<|im_end|>\n<|im_start|>assistant\n" + "<think>\n\n</think>\n\n")
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": temp, "stream": True,
                "cache_prompt": True, "stop": ["<|im_end|>", "<|endoftext|>"]}
        for k, v in s.params.items():
            if k in ("top_k", "top_p", "min_p", "presence_penalty", "repeat_penalty", "frequency_penalty"):
                body[k] = v
        for k, v in (sampler or {}).items():
            body[k] = v
        g = gen or Gen(max_tokens=max_tokens, model=s.name)
        with s.lock:
            s.inflight += 1
        try:
            for attempt in range(20):
                check_stop()
                g.reset()
                try:
                    req = urllib.request.Request(s.url + "/completion", data=json.dumps(body).encode(),
                                                 headers={**UA, "Content-Type": "application/json"})
                    nout = None
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        for line in r:
                            if not line.startswith(b"data:"):
                                continue
                            j = json.loads(line[5:])
                            if "error" in j:
                                raise RuntimeError(str(j["error"])[:200])
                            if g.add(j.get("content", "")):
                                break
                            if j.get("stop"):
                                nout = j.get("tokens_predicted")
                                break
                    nout = nout or g.n
                    nin = s.count(prompt)
                    with LOCK:
                        STATS["in"] += nin
                        STATS["out"] += nout
                        STATS["calls"] += 1
                    return g.text().strip(), {"in": nin, "out": nout, "looped": g.looped}
                except Interrupted:
                    raise
                except Exception as e:
                    if Stop.is_set():
                        raise Interrupted()
                    log("warn", s.name + ": call failed: " + repr(e)[:120] +
                        "  (retry " + str(attempt + 1) + ")", "yellow")
                    if Stop.sleep(min(20, 3 * (attempt + 1))):
                        raise Interrupted()
                    s._ensure()
            raise RuntimeError("llama-server chat failed repeatedly")
        finally:
            with s.lock:
                s.inflight -= 1


# ---------------------------------------------------------------- the model registry
class Registry:
    """Which models exist: the base model and every fine-tuned version produced so far.
    models/registry.json is the single source of truth, so any script can pick one up."""
    PATH = os.path.join(MODELS, "registry.json")

    @classmethod
    def load(cls):
        return read_json(cls.PATH, {"base": None, "versions": []})

    @classmethod
    def save(cls, reg):
        write_json(cls.PATH, reg)

    @classmethod
    def set_base(cls, name, gguf, params):
        reg = cls.load()
        reg["base"] = {"id": "base", "name": name, "gguf": gguf, "params": params or {}}
        cls.save(reg)
        return reg["base"]

    @classmethod
    def add_version(cls, info):
        reg = cls.load()
        reg["versions"] = [v for v in reg["versions"] if v["id"] != info["id"]] + [info]
        reg["versions"].sort(key=lambda v: v.get("n", 0))
        cls.save(reg)
        return info

    @classmethod
    def latest(cls):
        reg = cls.load()
        return reg["versions"][-1] if reg["versions"] else reg.get("base")

    @classmethod
    def next_n(cls):
        reg = cls.load()
        return 1 + max([v.get("n", 0) for v in reg["versions"]] or [0])

    @classmethod
    def resolve(cls, spec):
        """'base' | 'base_f16' | 'latest' | 'v3' | a path to a .gguf -> the entry to serve."""
        reg = cls.load()
        if not spec or spec == "base":
            return reg.get("base")
        if spec == "base_f16":
            return reg.get("base_f16") or reg.get("base")
        if spec == "latest":
            return cls.latest()
        for v in reg["versions"]:
            if v["id"] == spec:
                return v
        if spec.endswith(".gguf") and os.path.exists(spec):
            return {"id": os.path.basename(spec), "name": os.path.basename(spec), "gguf": spec, "params": {}}
        return None


# ---------------------------------------------------------------- the backend the phases talk to
class Backend:
    """Serves one or two models at once (a fine-tuned generator plus the untouched base model the judges use)
    and hands out `.chat(...)`. Prefers batched llama-server; falls back to Ollama when that is impossible."""

    def __init__(s, ctx, slots=0, port=11500, prefer="auto"):
        s.ctx, s.slots, s.port, s.prefer = ctx, slots, port, prefer
        s.ollama = Ollama(port + 90, ctx)
        s.servers, s.entries, s.kind = {}, {}, "none"
        s.lock = threading.Lock()

    def ensure_weights(s, model_name):
        """Make sure the base GGUF exists on disk, pulling it with Ollama the first time."""
        files = ollama_model_files(model_name)
        if files is None:
            s.ollama.start()
            s.ollama.pull(model_name)
            files = ollama_model_files(model_name)
            s.ollama.stop()
        if files is None:
            raise SystemExit("could not locate the weights for " + model_name + " after pulling them")
        return files

    def serve(s, role, entry, ctx=0):
        """Start (or reuse) a server for `entry` under the name `role` ('gen' or 'judge').

        Each role gets the context it actually needs. A judge reads all three answers whole, so its context
        is roughly three times a solver's; giving the solvers that same huge context would reserve a KV
        cache they never use and cost most of the parallel slots. Two servers are therefore run even for the
        same weights when the roles need different contexts - the second copy of a 0.8B model costs about a
        gigabyte and buys back several slots."""
        ctx = ctx or s.ctx
        with s.lock:
            for r, srv in s.servers.items():
                if s.entries.get(r, {}).get("gguf") == entry["gguf"] and getattr(srv, "ctx", 0) >= ctx:
                    s.servers[role], s.entries[role] = srv, entry     # same weights, big enough: share it
                    return srv
            port = s.port + 3 * len(set(id(x) for x in s.servers.values()))
            exe = s.ollama.binary()
            srv = LlamaServer(port, ctx, s.slots, entry["gguf"], entry.get("params"), exe,
                              name=role + ":" + entry["id"])
            Stop.on_exit(srv.stop)
            if s.prefer != "ollama" and srv.available() and srv.setup():
                s.servers[role], s.entries[role] = srv, entry
                s.kind = "llama"
                return srv
            # Ollama can only serve a model from its own registry, so a fine-tuned GGUF has no fallback:
            # say so plainly rather than failing later with an unhelpful "model not found".
            if entry.get("id") not in ("base", "base_f16"):
                raise SystemExit(
                    "llama-server could not be started for " + entry["id"] + ", and Ollama cannot serve a "
                    "locally produced GGUF. Free up GPU memory (or pass --backend llama --slots 1) and "
                    "re-run; the weights are at " + str(entry.get("gguf")))
            log("warn", "llama-server unusable for " + entry["id"] + "; falling back to Ollama", "yellow")
            if not s.ollama.alive():
                s.ollama.start()
            Stop.on_exit(s.ollama.stop)
            s.ollama.pull(entry["name"])
            s.servers[role], s.entries[role] = s.ollama, entry
            s.kind = "ollama"
            return s.ollama

    def chat(s, role, system, user, max_tokens, temp, gen=None, sampler=None):
        srv = s.servers[role]
        name = s.entries[role]["name"]
        return srv.chat(name, system, user, max_tokens, temp, gen=gen, sampler=sampler)

    def total_slots(s):
        n = 0
        for srv in set(s.servers.values()):
            n += getattr(srv, "slots", 1)
        return max(1, n)

    def status(s):
        bits = []
        for role, srv in s.servers.items():
            bits.append(role + " " + (srv.status() if hasattr(srv, "status") else "ollama"))
        return " | ".join(dict.fromkeys(bits))

    def stop(s):
        for srv in set(s.servers.values()):
            try:
                srv.stop()
            except Exception:
                pass
        s.ollama.stop()


# ---------------------------------------------------------------- the project venv (heavy dependencies)
class Venv:
    """A project-local virtual environment for the things that cannot be pure stdlib (torch, mlx, pyarrow).
    Created on first use and remembered, so the user never installs anything by hand."""
    DIR = os.path.join(ROOT, ".venv")

    @classmethod
    def python(cls):
        return os.path.join(cls.DIR, "Scripts", "python.exe") if WIN else os.path.join(cls.DIR, "bin", "python")

    @classmethod
    def ensure(cls):
        if os.path.exists(cls.python()):
            return cls.python()
        log("venv", "creating the project virtual environment (" + cls.DIR + ")")
        subprocess.run([sys.executable, "-m", "venv", cls.DIR], check=True)
        subprocess.run([cls.python(), "-m", "pip", "install", "-q", "--upgrade", "pip", "wheel"], check=False)
        return cls.python()

    @classmethod
    def have(cls, *modules):
        if not os.path.exists(cls.python()):
            return False
        code = "import " + ", ".join(modules)
        return subprocess.run([cls.python(), "-c", code], capture_output=True).returncode == 0

    @classmethod
    def pip(cls, packages, index=None, note=""):
        cls.ensure()
        if not packages:
            return
        log("venv", "installing " + (note or " ".join(packages))[:90])
        cmd = [cls.python(), "-m", "pip", "install", "--upgrade"] + list(packages)
        if index:
            cmd += ["--index-url", index]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            raise SystemExit("could not install " + " ".join(packages) + " into " + cls.DIR)

    @classmethod
    def need(cls, modules, packages, index=None, note=""):
        """Install only when the modules are missing, so re-runs cost nothing."""
        if cls.have(*modules):
            return
        cls.pip(packages, index=index, note=note)

    @classmethod
    def run(cls, args, **kw):
        return subprocess.run([cls.python()] + list(args), **kw)

    @classmethod
    def runner(cls):
        """The interpreter that generated code is executed with: the project venv when it exists (so a
        benchmark problem that legitimately needs numpy can run), otherwise this interpreter."""
        p = cls.python()
        return p if os.path.exists(p) else sys.executable


# ---------------------------------------------------------------- the dashboard server
class DashServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = not WIN


class Dash:
    """Serves dashboard.html plus the read-only JSON it polls, on 127.0.0.1 only. Each phase passes an `api`
    callable that answers its own routes; the shared ones (live calls, events, gpu, phase) are handled here."""

    def __init__(s, api=None, title="run"):
        s.api_extra, s.title, s.url, s.srv = api, title, None, None

    def start(s, port=8777, browser=True):
        dash = s

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
                if host not in ("127.0.0.1", "localhost"):
                    return self.reply(403, b"forbidden", "text/plain")
                try:
                    if u.path in ("/", "/index.html", "/dashboard.html"):
                        with open(DASH_HTML, "rb") as f:
                            return self.reply(200, f.read(), "text/html; charset=utf-8")
                    if u.path.startswith("/api/"):
                        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
                        data = dash.route(u.path[5:], q)
                        if data is not None:
                            return self.reply(200, json.dumps(data, ensure_ascii=False, default=str).encode(),
                                              "application/json")
                    self.reply(404, b"not found", "text/plain")
                except Exception as e:
                    self.reply(500, json.dumps({"error": repr(e)}).encode(), "application/json")

            def reply(self, code, body, ctype):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Cache-Control", "no-store")
                    origin = self.headers.get("Origin") or ""
                    if origin == "null" or re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", origin):
                        self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

        for p in range(port, port + 25):
            try:
                s.srv = DashServer(("127.0.0.1", p), Handler)
                break
            except OSError:
                continue
        else:
            log("warn", "dashboard: no free port in " + str(port) + "-" + str(port + 24), "yellow")
            return None
        s.url = "http://127.0.0.1:" + str(p) + "/"
        threading.Thread(target=s.srv.serve_forever, daemon=True).start()
        Stop.on_exit(lambda: s.srv.shutdown())
        log("dash", "live dashboard: " + s.url, "bold")
        if browser:
            threading.Thread(target=webbrowser.open, args=(s.url,), daemon=True).start()
        return s

    def route(s, what, q):
        if what == "live":
            with LOCK:
                cur = list(LIVE.values())
                recent = [g for g in RECENT if g.t_end and time.time() - g.t_end < 20]
            want = int(q.get("id") or 0)
            out = {"calls": [g.brief() for g in cur + recent]}
            if want:
                for g in cur + recent:
                    if g.id == want:
                        out["text"] = g.text()
                        out["call"] = g.brief()
            else:
                out["preview"] = {g.id: g.text()[-1400:] for g in cur}
            return out
        if what == "events":
            ev = list(EVENTS)
            since = int(q.get("since") or 0)
            return {"boot": T0, "events": [e for e in ev if e["seq"] > since][-600:],
                    "last": ev[-1]["seq"] if ev else 0}
        if what == "gpu":
            return {"gpu": gpu_status(), "phase": dict(PHASE), "stats": dict(STATS), "t": time.time()}
        return s.api_extra(what, q) if s.api_extra else None


def common_state(backend=None, extra=None):
    """The block of status every phase reports to the dashboard header."""
    g = gpu_status()
    st = {"t": time.time(), "boot": T0, "elapsed": time.time() - T0, "phase": dict(PHASE),
          "gpu": g, "stats": dict(STATS), "platform": platform.system(), "machine": platform.machine(),
          "stopping": Stop.is_set(),
          "backend": {"kind": backend.kind, "status": backend.status(), "slots": backend.total_slots(),
                      "models": {r: e.get("id") for r, e in backend.entries.items()}} if backend else None}
    if extra:
        st.update(extra)
    return st
