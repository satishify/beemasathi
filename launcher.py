#!/usr/bin/env python3
"""BeemaSathi launcher for macOS, Linux and Windows.

Checks the environment, installs what is missing, keeps the hosted model
settings in .env, and starts or stops the app. The hosted endpoint is tried
first; local Ollama is the fallback.

Usage:
  python launcher.py            interactive menu
  python launcher.py start      start in the background
  python launcher.py stop       stop the app
  python launcher.py status     show backend and app state
  python launcher.py config     edit the hosted settings in .env
  python launcher.py check      report the environment, install nothing
  python launcher.py index ...  rebuild the vector index

On macOS/Linux use ./beemasathi.sh; on Windows double-click
Start BeemaSathi.bat. Both run this file.
"""

from __future__ import annotations

import getpass
import json
import os
import shutil
import signal
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

if sys.version_info < (3, 11):
    print(
        "Python 3.11 or newer is required. This interpreter is %d.%d."
        % sys.version_info[:2]
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
IS_WINDOWS = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
URL = "http://127.0.0.1:8501"
HEALTH = URL + "/_stcore/health"
OLLAMA = "http://127.0.0.1:11434"
LOCAL_MODEL = "qwen2.5:7b"
CHROMA = ROOT / "data" / "chroma_db" / "chroma.sqlite3"

_USER = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
RUN_DIR = Path(tempfile.gettempdir()) / ("beemasathi-" + _USER.replace(" ", "_"))
APP_PID = RUN_DIR / "streamlit.pid"
OLLAMA_PID = RUN_DIR / "ollama.pid"
APP_LOG = RUN_DIR / "streamlit.log"
OLLAMA_LOG = RUN_DIR / "ollama.log"

MANAGED = (
    "BEEMASATHI_LLM_BASE",
    "BEEMASATHI_LLM_KEY",
    "BEEMASATHI_LLM_MODEL",
    "BEEMASATHI_LLM_CA",
    "BEEMASATHI_EMBED_BASE",
    "BEEMASATHI_EMBED_KEY",
    "BEEMASATHI_EMBED_KEY_FILE",
    "BEEMASATHI_EMBED_MODEL",
    "BEEMASATHI_EMBED_CA",
)


def say(message: str = "") -> None:
    print(message, flush=True)


def note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- settings -------------------------------------------------------------


def read_env_file() -> dict:
    values = {}
    path = ROOT / ".env"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name.startswith("export "):
            name = name[len("export "):].strip()
        values[name] = value.strip().strip('"').strip("'")
    return values


def env_value(name: str, default: str = "") -> str:
    if name in os.environ:
        return os.environ[name]
    return read_env_file().get(name, default)


def hosted_config() -> tuple:
    base = (env_value("BEEMASATHI_LLM_BASE") or "").rstrip("/")
    key = env_value("BEEMASATHI_LLM_KEY")
    ca = env_value("BEEMASATHI_LLM_CA")
    return base, key, ca


def embedding_config() -> tuple:
    base = (env_value("BEEMASATHI_EMBED_BASE") or "").rstrip("/")
    key = env_value("BEEMASATHI_EMBED_KEY")
    key_file = env_value("BEEMASATHI_EMBED_KEY_FILE")
    ca = env_value("BEEMASATHI_EMBED_CA")
    return base, key, key_file, ca


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ssl_context(ca: str) -> ssl.SSLContext:
    if ca:
        return ssl.create_default_context(cafile=str(resolve_path(ca)))
    return ssl.create_default_context()


def hosted_state() -> str:
    base, key, ca = hosted_config()
    if not (base and key):
        return "unconfigured"
    if ca and not resolve_path(ca).is_file():
        return "ca"
    try:
        request = urllib.request.Request(
            base + "/models",
            headers={"Authorization": "Bearer " + key},
        )
        with urllib.request.urlopen(
            request, timeout=8, context=ssl_context(ca)
        ) as response:
            return "ok" if response.status < 500 else "error"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "auth"
        return "ok" if exc.code < 500 else "error"
    except Exception:
        return "unreachable"


def hosted_summary() -> str:
    base, key, _ = hosted_config()
    if base and key:
        return "%s (%s)" % (base, env_value("BEEMASATHI_LLM_MODEL", "Qwen/Qwen3-8B"))
    return "not configured"


def embedding_summary() -> str:
    base, key, key_file, _ = embedding_config()
    has_key = bool(key) or (bool(key_file) and resolve_path(key_file).is_file())
    model = env_value("BEEMASATHI_EMBED_MODEL", "bge-small-en-v1.5")
    if base and has_key:
        return "hosted %s (%s)" % (base, model)
    if base:
        return "hosted %s (no key set)" % base
    return "local bge-small-en-v1.5"


def local_embeddings_needed() -> bool:
    base, key, key_file, _ = embedding_config()
    if not base:
        return True
    return not (key or (key_file and resolve_path(key_file).is_file()))


# --- environment ----------------------------------------------------------


def python_candidates() -> list:
    candidates = []
    if os.environ.get("PYTHON"):
        candidates.append([os.environ["PYTHON"]])
    if IS_WINDOWS:
        candidates.extend([["py", "-3"], ["python"], ["python3"]])
    else:
        candidates.extend([["python3.13"], ["python3.12"], ["python3.11"],
                           ["python3"], ["python"]])
    return candidates


def usable_python() -> tuple:
    for command in python_candidates():
        if shutil.which(command[0]) is None:
            continue
        try:
            result = subprocess.run(
                command + ["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            continue
        if result.returncode != 0:
            continue
        try:
            major, minor = (int(part) for part in result.stdout.strip().split("."))
        except ValueError:
            continue
        if (major, minor) >= (3, 11):
            return command, "%d.%d" % (major, minor)
    return None, None


def venv_ok() -> bool:
    return PY.is_file()


def ensure_venv() -> bool:
    if venv_ok():
        return True
    command, version = usable_python()
    if not command:
        note("Python 3.11 or newer is required. Install it, then run this again.")
        note("Windows: https://www.python.org/downloads/ (tick 'Add python.exe to PATH').")
        return False
    say("Creating the virtual environment with Python %s (first run only)..." % version)
    result = subprocess.run(command + ["-m", "venv", str(VENV)], cwd=str(ROOT))
    if result.returncode != 0 or not venv_ok():
        note("Could not create the virtual environment at %s" % VENV)
        return False
    return True


def missing_modules() -> list:
    if not venv_ok():
        return []
    modules = [
        "streamlit", "chromadb", "httpx", "dotenv", "requests",
        "numpy", "pydantic", "ollama",
    ]
    if local_embeddings_needed():
        modules.append("sentence_transformers")
    code = (
        "import importlib.util\n"
        "mods = %r\n"
        "print(','.join(m for m in mods if importlib.util.find_spec(m) is None))\n"
    ) % (modules,)
    try:
        result = subprocess.run(
            [str(PY), "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        note("Dependency check failed: %s" % exc)
        return modules
    if result.returncode != 0:
        note("Dependency check failed: %s" % result.stderr.strip()[:300])
        return modules
    return [name for name in result.stdout.strip().split(",") if name]


def requirements_for_install(need_sentence_transformers: bool) -> Path:
    requirements = ROOT / "requirements.txt"
    if need_sentence_transformers:
        return requirements
    lines = requirements.read_text(encoding="utf-8").splitlines()
    kept = [
        line for line in lines
        if line.strip().lower().replace("_", "-") != "sentence-transformers"
    ]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = RUN_DIR / "requirements-launcher.txt"
    trimmed.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return trimmed


def ensure_deps() -> bool:
    if not venv_ok():
        return False
    missing = missing_modules()
    if not missing:
        return True
    say("Installing dependencies (missing: %s)." % ", ".join(missing))
    say("This can take a few minutes the first time.")
    subprocess.run(
        [str(PY), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        cwd=str(ROOT),
        check=False,
    )
    requirements = requirements_for_install(
        "sentence_transformers" in missing
    )
    result = subprocess.run(
        [str(PY), "-m", "pip", "install", "-r", str(requirements)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        note("Dependency install failed. Run:")
        note("  %s -m pip install -r %s" % (PY, ROOT / "requirements.txt"))
        return False
    still = missing_modules()
    if still:
        note("Still missing after install: %s" % ", ".join(still))
        return False
    say("Dependencies installed.")
    return True


def ensure_index() -> bool:
    if CHROMA.is_file():
        return True
    note("Vector index missing at data/chroma_db/.")
    note("Rebuild it with: python launcher.py index (after downloading the PDFs).")
    return False


def ensure_ready() -> bool:
    return ensure_venv() and ensure_deps() and ensure_index()


def index_chunks() -> str:
    if not CHROMA.is_file():
        return "missing"
    try:
        manifest = json.loads(
            (ROOT / "data" / "chroma_db" / "index_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        return "{:,} chunks".format(int(manifest.get("chunks", 0)))
    except Exception:
        return "present"


def python_version() -> str:
    if venv_ok():
        try:
            result = subprocess.run(
                [str(PY), "-V"], capture_output=True, text=True, timeout=60
            )
            return result.stdout.strip().replace("Python ", "")
        except Exception:
            return "not found"
    _, version = usable_python()
    return version or "not found"


# --- app process ----------------------------------------------------------


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if IS_WINDOWS:
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def app_pid() -> int:
    try:
        return int(APP_PID.read_text(encoding="utf-8").strip())
    except Exception:
        return 0


def app_running() -> bool:
    return pid_alive(app_pid())


def port_up() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=2):
            return True
    except Exception:
        return False


def wait_for_app(timeout: int) -> bool:
    for _ in range(timeout):
        if port_up():
            return True
        time.sleep(1)
    return False


def detach_kwargs() -> dict:
    kwargs = {
        "cwd": str(ROOT),
        "stdin": subprocess.DEVNULL,
    }
    if IS_WINDOWS:
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # detached, new group
    else:
        kwargs["start_new_session"] = True
    return kwargs


def terminate(pid: int) -> None:
    if not pid_alive(pid):
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(20):
        if not pid_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def log_tail(path: Path, lines: int = 15) -> str:
    try:
        data = path.read_bytes()[-6000:]
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def open_app() -> None:
    if os.environ.get("BEEMASATHI_NO_BROWSER") == "1":
        return
    try:
        webbrowser.open(URL)
    except Exception:
        pass


# --- ollama ---------------------------------------------------------------


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def ollama_models() -> list:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [item.get("name", "") for item in data.get("models", [])]
    except Exception:
        return []


def ollama_has_model(name: str) -> bool:
    wanted = {name, name + ":latest"}
    return any(model in wanted for model in ollama_models())


def start_ollama_server() -> bool:
    if ollama_up():
        return True
    executable = shutil.which("ollama")
    if not executable:
        return False
    say("Starting the Ollama server...")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(OLLAMA_LOG, "ab") as log:
        kwargs = detach_kwargs()
        kwargs["stdout"] = log
        kwargs["stderr"] = subprocess.STDOUT
        process = subprocess.Popen([executable, "serve"], **kwargs)
    OLLAMA_PID.write_text(str(process.pid), encoding="utf-8")
    for _ in range(30):
        if ollama_up():
            return True
        time.sleep(1)
    return False


def pull_model() -> bool:
    executable = shutil.which("ollama")
    if executable:
        result = subprocess.run([executable, "pull", LOCAL_MODEL], cwd=str(ROOT))
        return result.returncode == 0 and ollama_has_model(LOCAL_MODEL)
    payload = json.dumps({"name": LOCAL_MODEL, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA + "/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            last = ""
            for raw in response:
                try:
                    event = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if event.get("error"):
                    note(str(event["error"]))
                    return False
                status = event.get("status", "")
                if status and status != last:
                    say(status)
                    last = status
    except Exception as exc:
        note("Model download failed: %s" % exc)
        return False
    return ollama_has_model(LOCAL_MODEL)


def warm_model() -> None:
    payload = json.dumps(
        {"model": LOCAL_MODEL, "prompt": "", "stream": False, "keep_alive": "10m"}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180):
            pass
    except Exception:
        pass


def unload_model() -> None:
    executable = shutil.which("ollama")
    if executable:
        subprocess.run(
            [executable, "stop", LOCAL_MODEL], capture_output=True, check=False
        )
        return
    payload = json.dumps({"model": LOCAL_MODEL, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
    except Exception:
        pass


# --- start / stop ---------------------------------------------------------


def start_app() -> bool:
    if not ensure_ready():
        return False
    if port_up() or app_running():
        say("BeemaSathi is already running at %s" % URL)
        open_app()
        return True

    state = hosted_state()
    if state == "ok":
        say("Hosted model is answering.")
    elif state == "unconfigured":
        say("No hosted model configured; using local Ollama.")
    elif state == "auth":
        say("Hosted model rejected the key; falling back to local Ollama.")
    elif state == "ca":
        say("Hosted CA certificate file is missing; falling back to local Ollama.")
        say("Fix the path with: python launcher.py config")
    else:
        say("Hosted model is not reachable; falling back to local Ollama.")

    child_env = dict(os.environ)
    child_env.setdefault("BEEMASATHI_DEBUG", "0")
    if IS_WINDOWS:
        child_env.setdefault("PYTHONUTF8", "1")

    if state != "ok":
        if not ollama_up() and not start_ollama_server():
            note("No hosted model and no Ollama server.")
            note("Run: python launcher.py config   (to set the hosted endpoint)")
            note("Or install Ollama from https://ollama.com and run: ollama pull %s"
                 % LOCAL_MODEL)
            return False
        if not ollama_has_model(LOCAL_MODEL):
            say("Local model %s is not installed." % LOCAL_MODEL)
            try:
                answer = input("Download it now (about 4.7 GB)? [y/N] ").strip()
            except EOFError:
                answer = ""
            if answer.lower() not in ("y", "yes"):
                note("Cancelled. Run: ollama pull %s" % LOCAL_MODEL)
                return False
            if not pull_model():
                return False
        warm_model()
        # The child reads .env itself, so blank these for this process only.
        child_env["BEEMASATHI_LLM_BASE"] = ""
        child_env["BEEMASATHI_LLM_KEY"] = ""

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY), "-m", "streamlit", "run", str(ROOT / "ui" / "app.py"),
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--server.headless=true",
    ]
    with open(APP_LOG, "ab") as log:
        kwargs = detach_kwargs()
        kwargs["stdout"] = log
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["env"] = child_env
        process = subprocess.Popen(command, **kwargs)
    APP_PID.write_text(str(process.pid), encoding="utf-8")

    if wait_for_app(120):
        say("BeemaSathi started: %s" % URL)
        if state == "ok":
            say("Model backend: hosted API (%s)"
                % env_value("BEEMASATHI_LLM_MODEL", "Qwen/Qwen3-8B"))
        else:
            say("Model loaded: %s" % LOCAL_MODEL)
        open_app()
        return True

    note("Streamlit failed to start. Last lines of %s:" % APP_LOG)
    tail = log_tail(APP_LOG)
    if tail:
        note(tail)
    return False


def stop_app() -> bool:
    pid = app_pid()
    if pid_alive(pid):
        terminate(pid)
        say("BeemaSathi stopped.")
    else:
        say("BeemaSathi is not running.")
    try:
        APP_PID.unlink()
    except OSError:
        pass

    base, key, _ = hosted_config()
    if not (base and key) and ollama_up():
        unload_model()
        say("Model unloaded: %s" % LOCAL_MODEL)

    ollama_pid = 0
    try:
        ollama_pid = int(OLLAMA_PID.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    if pid_alive(ollama_pid):
        terminate(ollama_pid)
        say("Ollama server stopped.")
    try:
        OLLAMA_PID.unlink()
    except OSError:
        pass
    return True


# --- report ---------------------------------------------------------------


def report() -> None:
    if venv_ok():
        missing = missing_modules()
        deps_line = "all installed" if not missing else "missing: " + ", ".join(missing)
        version = python_version()
        origin = "project venv"
    else:
        version = python_version()
        deps_line = "not installed (no venv yet)"
        origin = "no venv yet"

    say("")
    say("BeemaSathi environment")
    say("  Python          %s (%s)" % (version, origin))
    say("  Dependencies    %s" % deps_line)
    say("  Vector index    %s" % index_chunks())
    if venv_ok():
        say("  Hosted model    %s" % hosted_summary())
        say("  Embeddings      %s" % embedding_summary())
    else:
        say("  Hosted model    set in .env, checked after setup")

    if ollama_up():
        models = ", ".join(ollama_models())
        say("  Ollama          running%s" % ((" (%s)" % models) if models else ""))
    elif shutil.which("ollama"):
        say("  Ollama          installed, server not running")
    else:
        say("  Ollama          not installed")

    if port_up():
        say("  App             running at %s" % URL)
    else:
        say("  App             stopped")
    say("")


# --- configure ------------------------------------------------------------


def configure() -> bool:
    if not ensure_venv():
        return False
    env_path = ROOT / ".env"
    if not env_path.exists():
        example = ROOT / ".env.example"
        env_path.write_text(
            example.read_text(encoding="utf-8") if example.exists() else "",
            encoding="utf-8",
        )
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass

    current = read_env_file()

    def ask(label: str, name: str, secret: bool = False) -> str:
        value = current.get(name, "")
        if secret:
            shown = "(set, hidden)" if value else "(empty)"
            answer = getpass.getpass("%s [%s]: " % (label, shown))
        else:
            shown = value or "(empty)"
            answer = input("%s [%s]: " % (label, shown)).strip()
        if answer == "":
            return value
        if answer == "-":
            return ""
        return answer

    say("")
    say("Hosted OpenAI-compatible endpoint. Leave a field blank to keep it,")
    say("type - to clear it. Press Ctrl-C to cancel.")
    say("")
    base = ask("Chat base URL (none = local Ollama only)", "BEEMASATHI_LLM_BASE")
    if base.strip().lower() in {"none", "local", "off"}:
        base = ""
    key = ask("Chat API key", "BEEMASATHI_LLM_KEY", secret=True)
    model = ask("Chat model name", "BEEMASATHI_LLM_MODEL") or "Qwen/Qwen3-8B"
    ca = ask("Chat CA certificate path (optional)", "BEEMASATHI_LLM_CA")
    say("")
    say("Embedding endpoint. Leave blank to keep it; none runs the local model.")
    say("")
    embed_base = ask(
        "Embedding base URL (none = local model)", "BEEMASATHI_EMBED_BASE"
    )
    if embed_base.strip().lower() in {"none", "local", "off"}:
        embed_base = ""
    embed_key = ask("Embedding API key", "BEEMASATHI_EMBED_KEY", secret=True)
    embed_model = ask("Embedding model", "BEEMASATHI_EMBED_MODEL") or "bge-small-en-v1.5"
    embed_ca = ask(
        "Embedding CA certificate path (optional)", "BEEMASATHI_EMBED_CA"
    )
    embed_key_file = current.get("BEEMASATHI_EMBED_KEY_FILE", "")
    if embed_key:
        # An inline key wins, so drop a stale path that may not exist here.
        embed_key_file = ""

    managed = {
        "BEEMASATHI_LLM_BASE": base,
        "BEEMASATHI_LLM_KEY": key,
        "BEEMASATHI_LLM_MODEL": model,
        "BEEMASATHI_LLM_CA": ca,
        "BEEMASATHI_EMBED_BASE": embed_base,
        "BEEMASATHI_EMBED_KEY": embed_key,
        "BEEMASATHI_EMBED_KEY_FILE": embed_key_file,
        "BEEMASATHI_EMBED_MODEL": embed_model,
        "BEEMASATHI_EMBED_CA": embed_ca,
    }

    lines = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name = line.split("=", 1)[0].strip() if "=" in line else ""
        if name in MANAGED and not line.lstrip().startswith("#"):
            continue
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    lines.append("")
    lines.append("# Hosted endpoints (managed by the BeemaSathi launcher)")
    for name, value in managed.items():
        lines.append("%s=%s" % (name, value))
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(env_path, 0o600)
    except OSError:
        pass
    say("")
    say("Saved to .env (readable only by you).")

    if not (base and key):
        say("Hosted chat disabled; the app will use local Ollama.")
    else:
        try:
            request = urllib.request.Request(
                base.rstrip("/") + "/models",
                headers={"Authorization": "Bearer " + key},
            )
            with urllib.request.urlopen(
                request, timeout=8, context=ssl_context(ca)
            ) as response:
                say("Chat connection test passed." if response.status < 500
                    else "The chat endpoint answered with %d; it may be busy."
                    % response.status)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                say("The chat endpoint answered, but it rejected the key.")
            else:
                say("The chat endpoint answered with %d." % exc.code)
        except Exception as exc:
            say("Chat connection test failed (%s)." % exc.__class__.__name__)
            say("The app will fall back to local Ollama when it is available.")

    if embed_base and embed_key:
        try:
            payload = json.dumps(
                {"model": embed_model, "input": ["ping"]}
            ).encode("utf-8")
            request = urllib.request.Request(
                embed_base.rstrip("/") + "/embeddings",
                data=payload,
                headers={
                    "Authorization": "Bearer " + embed_key,
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(
                request, timeout=20, context=ssl_context(embed_ca)
            ) as response:
                say("Embedding connection test passed."
                    if response.status < 400
                    else "The embedding endpoint answered with %d." % response.status)
        except urllib.error.HTTPError as exc:
            say("The embedding endpoint answered with %d." % exc.code)
        except Exception as exc:
            say("Embedding connection test failed (%s)." % exc.__class__.__name__)
            say("Queries will fail until the endpoint is reachable.")
    elif embed_base:
        say("Embedding endpoint set without a key; the app will need one.")
    else:
        say("Hosted embeddings disabled; the local bge-small-en-v1.5 model will load.")
    return True


# --- menu and entry point -------------------------------------------------


def menu() -> int:
    ensure_ready()
    while True:
        say("")
        say("BeemaSathi")
        say("  1) Start the app")
        say("  2) Stop the app")
        say("  3) Status")
        say("  4) Configure hosted models (.env)")
        say("  5) Check the environment and install what is missing")
        say("  q) Quit")
        try:
            choice = input("Choose: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if choice == "1":
            start_app()
        elif choice == "2":
            stop_app()
        elif choice == "3":
            report()
        elif choice == "4":
            configure()
        elif choice == "5":
            if ensure_ready():
                say("Everything needed is in place.")
        elif choice.lower() == "q":
            return 0
        else:
            say("Enter 1-5 or q.")


def run_index(arguments: list) -> int:
    if not venv_ok():
        note("Virtual environment is missing. Run the launcher once to create it.")
        return 1
    return subprocess.run(
        [str(PY), str(ROOT / "rag" / "sync_knowledge.py")] + arguments,
        cwd=str(ROOT),
    ).returncode


def main(argv: list) -> int:
    command = argv[1] if len(argv) > 1 else ""
    if command == "start":
        return 0 if start_app() else 1
    if command == "stop":
        return 0 if stop_app() else 1
    if command == "restart":
        stop_app()
        return 0 if start_app() else 1
    if command in ("status", "check"):
        report()
        return 0
    if command == "config":
        return 0 if configure() else 1
    if command == "index":
        return run_index(argv[2:])
    if command in ("", "menu"):
        return menu()
    say("Usage: %s {start|stop|restart|status|config|check|index}" % Path(argv[0]).name)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
