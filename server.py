#!/usr/bin/env python3
"""
Bedrock Chat Web Server
=======================
Local web UI for AWS Bedrock.  No data leaves your machine.
Run:   python server.py
Open:  http://localhost:8000
"""

import asyncio
import base64
import datetime
import json
import mimetypes
import pathlib
import re
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Directory layout ───────────────────────────────────────────────────────────
# IMPORTANT: All paths are anchored to THIS FILE's location using __file__
# so data always saves next to server.py regardless of where Python is launched.
# Running from VS Code terminal, cmd, or double-click — all save to same place.
_HERE     = pathlib.Path(__file__).resolve().parent
DATA_DIR  = _HERE / "chat_history"          # renamed from chat_data
CONV_DIR  = DATA_DIR / "conversations"
UPL_DIR   = DATA_DIR / "uploads"
SETS_FILE = DATA_DIR / "settings.json"
STAT_FILE = DATA_DIR / "stats.json"
CRED_FILE = DATA_DIR / "credentials.json"
TMPL_FILE = DATA_DIR / "templates.json"
PRESET_FILE = DATA_DIR / "presets.json"
PROJ_DIR  = DATA_DIR / "projects"
BACKUP_DIR = DATA_DIR / "backups"
HTML_FILE = _HERE / "index.html"

for _d in [CONV_DIR, UPL_DIR, PROJ_DIR, BACKUP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
USD_TO_INR = 83

BEDROCK_REGIONS = [
    {"id": "us-east-1",      "name": "US East — N. Virginia (most models available)"},
    {"id": "us-east-2",      "name": "US East — Ohio"},
    {"id": "us-west-2",      "name": "US West — Oregon"},
    {"id": "eu-west-1",      "name": "Europe — Ireland"},
    {"id": "eu-west-3",      "name": "Europe — Paris"},
    {"id": "eu-central-1",   "name": "Europe — Frankfurt"},
    {"id": "eu-north-1",     "name": "Europe — Stockholm"},
    {"id": "ap-northeast-1", "name": "Asia Pacific — Tokyo"},
    {"id": "ap-northeast-2", "name": "Asia Pacific — Seoul"},
    {"id": "ap-southeast-1", "name": "Asia Pacific — Singapore"},
    {"id": "ap-southeast-2", "name": "Asia Pacific — Sydney"},
    {"id": "ap-south-1",     "name": "Asia Pacific — Mumbai"},
    {"id": "ca-central-1",   "name": "Canada — Central"},
    {"id": "sa-east-1",      "name": "South America — Sao Paulo"},
]

def guess_fmt(model_id: str) -> str:
    """Infer streaming body format from a model ID (handles cross-region prefix 'us.' etc.)"""
    mid = model_id.lower()
    # Strip cross-region prefix  e.g. "us.anthropic.claude-..." → still matches "anthropic"
    if "anthropic" in mid or "claude"    in mid: return "claude"
    if "amazon"    in mid and "nova"     in mid: return "nova"
    if "amazon"    in mid and "titan"    in mid: return "nova"
    if "meta"      in mid or "llama"     in mid: return "llama"
    if "moonshot"  in mid or "kimi"      in mid: return "kimi"
    if "deepseek"  in mid:                       return "deepseek"
    if "mistral"   in mid:                       return "mistral"
    if "ai21"      in mid or "jamba"     in mid: return "mistral"
    if "cohere"    in mid:                       return "mistral"
    if "writer"    in mid or "palmyra"   in mid: return "mistral"
    return "claude"  # safe default

# USD per 1 000 tokens — used as fallback when AWS doesn't provide pricing
MODEL_CATALOG: Dict[str, Dict] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "name": "Claude 3.5 Sonnet v2", "provider": "Anthropic",
        "input": 0.003, "output": 0.015, "max_tokens": 8192, "fmt": "claude",
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "name": "Claude 3.5 Sonnet", "provider": "Anthropic",
        "input": 0.003, "output": 0.015, "max_tokens": 8192, "fmt": "claude",
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "name": "Claude 3.5 Haiku", "provider": "Anthropic",
        "input": 0.0008, "output": 0.004, "max_tokens": 8192, "fmt": "claude",
    },
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "name": "Claude 3 Haiku", "provider": "Anthropic",
        "input": 0.00025, "output": 0.00125, "max_tokens": 4096, "fmt": "claude",
    },
    "anthropic.claude-3-opus-20240229-v1:0": {
        "name": "Claude 3 Opus", "provider": "Anthropic",
        "input": 0.015, "output": 0.075, "max_tokens": 4096, "fmt": "claude",
    },
    "anthropic.claude-3-7-sonnet-20250219-v1:0": {
        "name": "Claude 3.7 Sonnet", "provider": "Anthropic",
        "input": 0.003, "output": 0.015, "max_tokens": 64000, "fmt": "claude",
    },
    "amazon.nova-micro-v1:0": {
        "name": "Nova Micro",  "provider": "Amazon",
        "input": 0.000035, "output": 0.00014, "max_tokens": 5120, "fmt": "nova",
    },
    "amazon.nova-lite-v1:0": {
        "name": "Nova Lite",   "provider": "Amazon",
        "input": 0.00006,  "output": 0.00024, "max_tokens": 5120, "fmt": "nova",
    },
    "amazon.nova-pro-v1:0": {
        "name": "Nova Pro",    "provider": "Amazon",
        "input": 0.0008,   "output": 0.0032,  "max_tokens": 5120, "fmt": "nova",
    },
    "amazon.titan-text-express-v1": {
        "name": "Titan Text Express", "provider": "Amazon",
        "input": 0.0002, "output": 0.0006, "max_tokens": 8192, "fmt": "nova",
    },
    "amazon.titan-text-lite-v1": {
        "name": "Titan Text Lite",    "provider": "Amazon",
        "input": 0.00015, "output": 0.0002, "max_tokens": 4096, "fmt": "nova",
    },
    "meta.llama3-1-70b-instruct-v1:0": {
        "name": "Llama 3.1 70B",  "provider": "Meta",
        "input": 0.00072, "output": 0.00072, "max_tokens": 8192, "fmt": "llama",
    },
    "meta.llama3-1-8b-instruct-v1:0": {
        "name": "Llama 3.1 8B",   "provider": "Meta",
        "input": 0.00022, "output": 0.00022, "max_tokens": 8192, "fmt": "llama",
    },
    "meta.llama3-2-90b-instruct-v1:0": {
        "name": "Llama 3.2 90B",  "provider": "Meta",
        "input": 0.00072, "output": 0.00072, "max_tokens": 8192, "fmt": "llama",
    },
    "meta.llama3-3-70b-instruct-v1:0": {
        "name": "Llama 3.3 70B",  "provider": "Meta",
        "input": 0.00072, "output": 0.00072, "max_tokens": 8192, "fmt": "llama",
    },
    "mistral.mistral-large-2402-v1:0": {
        "name": "Mistral Large",  "provider": "Mistral",
        "input": 0.004, "output": 0.012, "max_tokens": 8192, "fmt": "mistral",
    },
    "mistral.mistral-small-2402-v1:0": {
        "name": "Mistral Small",  "provider": "Mistral",
        "input": 0.001, "output": 0.003, "max_tokens": 8192, "fmt": "mistral",
    },
    "cohere.command-r-plus-v1:0": {
        "name": "Command R+", "provider": "Cohere",
        "input": 0.003, "output": 0.015, "max_tokens": 4096, "fmt": "mistral",
    },
    "cohere.command-r-v1:0": {
        "name": "Command R",  "provider": "Cohere",
        "input": 0.0005, "output": 0.0015, "max_tokens": 4096, "fmt": "mistral",
    },
    # ── Moonshot AI / Kimi ──────────────────────────────────────────────────────
    # IDs vary by region prefix (us. / eu. / ap.); handled by inference-profile API
    "us.moonshot.kimi-k2-instruct-v1:0": {
        "name": "Kimi K2 Instruct",       "provider": "Moonshot AI",
        "input": 0.0, "output": 0.0, "max_tokens": 131072, "fmt": "kimi",
    },
    "us.moonshot.kimi-k2-thinking-instruct-v1:0": {
        "name": "Kimi K2 Thinking",        "provider": "Moonshot AI",
        "input": 0.0, "output": 0.0, "max_tokens": 131072, "fmt": "kimi",
    },
    "eu.moonshot.kimi-k2-instruct-v1:0": {
        "name": "Kimi K2 Instruct (EU)",   "provider": "Moonshot AI",
        "input": 0.0, "output": 0.0, "max_tokens": 131072, "fmt": "kimi",
    },
    "eu.moonshot.kimi-k2-thinking-instruct-v1:0": {
        "name": "Kimi K2 Thinking (EU)",   "provider": "Moonshot AI",
        "input": 0.0, "output": 0.0, "max_tokens": 131072, "fmt": "kimi",
    },
    # ── Cross-region Claude inference profiles ──────────────────────────────────
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "name": "Claude 3.5 Sonnet v2 (CR)", "provider": "Anthropic",
        "input": 0.003, "output": 0.015, "max_tokens": 8192, "fmt": "claude",
    },
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
        "name": "Claude 3.7 Sonnet (CR)",    "provider": "Anthropic",
        "input": 0.003, "output": 0.015, "max_tokens": 64000, "fmt": "claude",
    },
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "name": "Claude 3.5 Haiku (CR)",     "provider": "Anthropic",
        "input": 0.0008, "output": 0.004, "max_tokens": 8192, "fmt": "claude",
    },
    # ── Cross-region Nova ───────────────────────────────────────────────────────
    "us.amazon.nova-pro-v1:0": {
        "name": "Nova Pro (CR)",   "provider": "Amazon",
        "input": 0.0008, "output": 0.0032, "max_tokens": 5120, "fmt": "nova",
    },
    "us.amazon.nova-lite-v1:0": {
        "name": "Nova Lite (CR)",  "provider": "Amazon",
        "input": 0.00006, "output": 0.00024, "max_tokens": 5120, "fmt": "nova",
    },
    "us.amazon.nova-micro-v1:0": {
        "name": "Nova Micro (CR)", "provider": "Amazon",
        "input": 0.000035, "output": 0.00014, "max_tokens": 5120, "fmt": "nova",
    },
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "aws_region": "us-east-1",
    "main_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "summary_model": "amazon.nova-micro-v1:0",
    "temperature": 0.7,
    "max_tokens": 8192,
    "show_thinking": True,
    "context_mode": "smart",
    "max_ctx_messages": 30,
    "cost_guard_enabled": True,
    "cost_guard_inr": 5.0,
    "system_prompt": (
        "You are a helpful AI assistant. "
        "When helpful, include a concise Reasoning Summary: assumptions, evidence, "
        "reasoning summary, uncertainty, and next steps. Keep reasoning auditable."
    ),
}

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Bedrock Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=6)

# ── Settings ───────────────────────────────────────────────────────────────────
def load_settings() -> Dict:
    if SETS_FILE.exists():
        try:
            return {**DEFAULT_SETTINGS, **json.loads(SETS_FILE.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)

def save_settings(s: Dict):
    SETS_FILE.write_text(json.dumps(s, indent=2))

# ── Stats ──────────────────────────────────────────────────────────────────────
def load_stats() -> Dict:
    if STAT_FILE.exists():
        try:
            return json.loads(STAT_FILE.read_text())
        except Exception:
            pass
    return {"messages": 0, "in_tok": 0, "out_tok": 0, "cost_usd": 0.0, "cost_inr": 0.0}

def add_stats(in_tok: int, out_tok: int, cost_usd: float):
    s = load_stats()
    s["messages"] += 1
    s["in_tok"]   += in_tok
    s["out_tok"]  += out_tok
    s["cost_usd"] += cost_usd
    s["cost_inr"] += cost_usd * USD_TO_INR
    STAT_FILE.write_text(json.dumps(s, indent=2))

# ── Conversation helpers ───────────────────────────────────────────────────────
def safe_id(value: str, label: str = "id") -> str:
    value = str(value or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
        raise HTTPException(400, f"Invalid {label}")
    return value

def cpath(cid: str) -> pathlib.Path:
    cid = safe_id(cid, "conversation id")
    return CONV_DIR / f"{cid}.json"

def load_conv(cid: str) -> Dict:
    p = cpath(cid)
    if not p.exists():
        raise HTTPException(404, f"Conversation {cid} not found")
    return json.loads(p.read_text(encoding="utf-8"))

def save_conv(c: Dict):
    c["updated"] = datetime.datetime.now().isoformat()
    cpath(c["id"]).write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")

def recompute_conv_totals(c: Dict):
    c["cost_inr"] = sum(float(m.get("cost_inr", 0) or 0) for m in c.get("messages", []) if m.get("role") == "assistant")
    c["cost_usd"] = sum(float(m.get("cost_usd", 0) or 0) for m in c.get("messages", []) if m.get("role") == "assistant")
    c["in_tok"]   = sum(int(m.get("in_tok", 0) or 0) for m in c.get("messages", []) if m.get("role") == "assistant")
    c["out_tok"]  = sum(int(m.get("out_tok", 0) or 0) for m in c.get("messages", []) if m.get("role") == "assistant")

def approx_tokens(value: Any) -> int:
    if isinstance(value, list):
        return sum(approx_tokens(part.get("text", "") if isinstance(part, dict) else part) for part in value)
    return max(1, round(len(str(value or "")) / 4))

def build_context_messages(conv: Dict, model_id: str, raw_messages: List[Dict] = None) -> List[Dict]:
    s = load_settings()
    max_ctx = int(s.get("max_ctx_messages", 30) or 30)
    source = list(raw_messages if raw_messages is not None else conv.get("messages", []))[-max_ctx:]
    ctx_msgs = []
    for m in source:
        c = m.get("content", "")
        if m.get("files") and guess_fmt(model_id) == "claude":
            parts: List[Dict] = [{"type": "text", "text": c}]
            for f in m.get("files", []):
                if f.get("type") == "image":
                    parts.append({"type": "image", "source": {
                        "type": "base64", "media_type": f.get("media_type", "image/png"), "data": f.get("data", "")
                    }})
                elif f.get("type") == "text":
                    parts.append({"type": "text", "text": f"\n\n[File: {f.get('name','file')}]\n{f.get('content','')}"})
            c = parts
        ctx_msgs.append({"role": m.get("role", "user"), "content": c})
    return ensure_alternating(ctx_msgs)

def conv_system_prompt(conv: Dict, s: Dict = None) -> str:
    s = s or load_settings()
    return conv.get("project_system") or s.get("system_prompt", DEFAULT_SETTINGS["system_prompt"])

def list_convs() -> List[Dict]:
    out = []
    for p in CONV_DIR.glob("*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "id":          c["id"],
                "title":       c.get("title", "Untitled"),
                "updated":     c.get("updated", ""),
                "created":     c.get("created", ""),
                "msg_count":   len(c.get("messages", [])),
                "cost_inr":    c.get("cost_inr", 0.0),
                "pinned":      c.get("pinned", False),
                "model":       c.get("model", ""),
                "project_id":  c.get("project_id", ""),
                "project_name":c.get("project_name", ""),
                "injected_doc_ids": c.get("injected_doc_ids", []),
                "note_count":  len(c.get("notes", [])),
            })
        except Exception:
            pass
    return sorted(out, key=lambda x: (x["pinned"], x["updated"]), reverse=True)

def calc_cost(model_id: str, in_tok: int, out_tok: int) -> Dict:
    m = MODEL_CATALOG.get(model_id, {"input": 0.003, "output": 0.015})
    usd = (in_tok / 1000 * m["input"]) + (out_tok / 1000 * m["output"])
    return {"usd": usd, "inr": usd * USD_TO_INR}

def auto_title(text: str) -> str:
    clean = re.sub(r"[^\w\s]", " ", text).strip()
    words = clean.split()[:9]
    return " ".join(words) if words else "New Chat"

def ensure_alternating(msgs: List[Dict]) -> List[Dict]:
    """Bedrock requires strict user/assistant alternation."""
    out: List[Dict] = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            prev = out[-1]["content"]
            cur  = m["content"]
            if isinstance(prev, str) and isinstance(cur, str):
                out[-1]["content"] = prev + "\n\n" + cur
            # If either side is a list (multipart), just skip the duplicate safely
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# ── AWS_Chat.py pipeline bridge ────────────────────────────────────────────────
# If AWS_Chat.py is in the same folder, we use its BedrockChat engine for ALL
# AI calls.  This gives us every bug-fix and model-format handler already in
# that code for free.  Falls back to the built-in implementation if not found.

import importlib.util, io, sys as _sys

AWS_CHAT_MOD = None   # the imported module, or None

def _load_pipeline():
    global AWS_CHAT_MOD
    candidate = pathlib.Path(__file__).parent / "AWS_Chat.py"
    if not candidate.exists():
        return
    try:
        spec = importlib.util.spec_from_file_location("_awschat_engine", candidate)
        mod  = importlib.util.module_from_spec(spec)
        # Suppress any startup prints from AWS_Chat.py
        _old, _sys.stdout = _sys.stdout, io.StringIO()
        try:    spec.loader.exec_module(mod)
        finally: _sys.stdout = _old
        AWS_CHAT_MOD = mod
        print("  Pipeline: AWS_Chat.py loaded as backend engine")
    except Exception as ex:
        print(f"  AWS_Chat.py found but could not load ({ex}) - using built-in engine")

_load_pipeline()

# ── Credentials helpers ────────────────────────────────────────────────────────
def load_credentials() -> Dict:
    if CRED_FILE.exists():
        try:
            return json.loads(CRED_FILE.read_text())
        except Exception:
            pass
    return {}

def save_credentials(creds: Dict):
    # Never write empty strings — keep the file clean
    CRED_FILE.write_text(json.dumps({k: v for k, v in creds.items() if v}, indent=2))

def _make_session(region: str = None) -> "boto3.Session":
    """Build a boto3 Session using stored credentials (keys, profile, or default chain)."""
    creds  = load_credentials()
    region = region or creds.get("aws_region") or load_settings().get("aws_region", "us-east-1")

    if creds.get("profile_name"):
        return boto3.Session(profile_name=creds["profile_name"], region_name=region)
    elif creds.get("aws_access_key_id"):
        kw = dict(
            aws_access_key_id     = creds["aws_access_key_id"],
            aws_secret_access_key = creds["aws_secret_access_key"],
            region_name           = region,
        )
        if creds.get("aws_session_token"):
            kw["aws_session_token"] = creds["aws_session_token"]
        return boto3.Session(**kw)
    else:
        # Fall back to the default credential chain (~/.aws, env vars, IAM role…)
        return boto3.Session(region_name=region)

# ── Bedrock invocation ─────────────────────────────────────────────────────────
def make_client(region: str = None):
    """Runtime client — uses stored credentials automatically."""
    cfg = Config(retries={"max_attempts": 5, "mode": "adaptive"},
                 connect_timeout=15, read_timeout=300)
    return _make_session(region).client("bedrock-runtime", config=cfg)

def make_mgmt_client(region: str = None):
    """Management client (list models etc.) — uses stored credentials."""
    cfg = Config(connect_timeout=8, read_timeout=15)
    return _make_session(region).client("bedrock", config=cfg)

def build_claude_body(msgs, system, max_tok, temp):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tok,
        "temperature": temp,
        "system": system,
        "messages": msgs,
    }

def build_nova_body(msgs, system, max_tok, temp):
    nova_msgs = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, str):
            c = [{"text": c}]
        nova_msgs.append({"role": m["role"], "content": c})
    body: Dict[str, Any] = {
        "messages": nova_msgs,
        "inferenceConfig": {"maxTokens": max_tok, "temperature": temp},
    }
    if system:
        body["system"] = [{"text": system}]
    return body

def build_llama_body(msgs, system, max_tok, temp):
    parts = []
    if system:
        parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}<|eot_id|>")
    for m in msgs:
        role = "user" if m["role"] == "user" else "assistant"
        c = m["content"] if isinstance(m["content"], str) else str(m["content"])
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{c}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>")
    return {"prompt": "\n".join(parts), "max_gen_len": max_tok, "temperature": temp}

def build_mistral_body(msgs, system, max_tok, temp):
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    for m in msgs:
        c = m["content"] if isinstance(m["content"], str) else str(m["content"])
        chat.append({"role": m["role"], "content": c})
    return {"messages": chat, "max_tokens": max_tok, "temperature": temp}

def build_kimi_body(msgs, system, max_tok, temp):
    """Kimi / Moonshot AI on Bedrock — OpenAI-compatible messages format."""
    chat = []
    if system:
        chat.append({"role": "system", "content": system})
    for m in msgs:
        c = m["content"] if isinstance(m["content"], str) else str(m["content"])
        chat.append({"role": m["role"], "content": c})
    return {
        "messages":   chat,
        "max_tokens": max_tok,
        "temperature": temp,
        "stream":     True,
    }

def _pipeline_stream(msgs, model_id, system, max_tok, temp, region, queue, loop):
    """
    Use AWS_Chat.py's BedrockChat engine to stream — gets all its bug-fixes for free.
    Falls back to built-in if anything goes wrong.
    """
    mod = AWS_CHAT_MOD
    if mod is None:
        return False

    try:
        # Apply stored credentials to AWS_Chat's global state so it uses our creds
        creds = load_credentials()
        r     = region or creds.get("aws_region") or load_settings().get("aws_region","us-east-1")
        if hasattr(mod, "AWS_REGION"):
            mod.AWS_REGION = r
        if creds.get("aws_access_key_id") and hasattr(mod, "_credential_override"):
            mod._credential_override = creds
        # Set model ID on its globals
        if hasattr(mod, "EXPENSIVE_MODEL_ID"):
            mod.EXPENSIVE_MODEL_ID = model_id

        # Use BedrockChat if available
        if hasattr(mod, "BedrockChat"):
            bc = mod.BedrockChat()
            # Build context the way AWS_Chat expects
            context = [{"role": m["role"],
                        "content": m["content"] if isinstance(m["content"], str)
                                   else str(m["content"])} for m in msgs]
            # Call its internal streaming method
            for attr in ("_invoke_claude", "_invoke_expensive_model", "_stream_response"):
                if hasattr(bc, attr):
                    fn = getattr(bc, attr)
                    # Different signatures — try common ones
                    try:
                        result_iter = fn(context, system=system, max_tokens=max_tok,
                                         temperature=temp, model_id=model_id)
                    except TypeError:
                        try:
                            result_iter = fn(context)
                        except Exception:
                            continue
                    if result_iter is not None:
                        in_tok = out_tok = 0
                        for item in result_iter:
                            if isinstance(item, str) and item:
                                asyncio.run_coroutine_threadsafe(queue.put({"t":"tok","v":item}), loop)
                            elif isinstance(item, dict):
                                if item.get("text"):
                                    asyncio.run_coroutine_threadsafe(queue.put({"t":"tok","v":item["text"]}), loop)
                                if "in_tok" in item:
                                    in_tok, out_tok = item["in_tok"], item.get("out_tok",0)
                        asyncio.run_coroutine_threadsafe(queue.put({"t":"done","in":in_tok,"out":out_tok}), loop)
                        return True
    except Exception:
        pass  # fall through to built-in
    return False


def sync_stream(msgs, model_id, system, max_tok, temp, region, queue, loop):
    """Run in thread pool; push tokens into asyncio queue.
    Tries the AWS_Chat.py pipeline first; falls back to built-in streaming."""
    try:
        # ── Pipeline attempt ──────────────────────────────────────────────────
        if _pipeline_stream(msgs, model_id, system, max_tok, temp, region, queue, loop):
            return  # pipeline handled it

        # ── Built-in streaming ────────────────────────────────────────────────
        client = make_client(region)
        fmt    = guess_fmt(model_id)  # use guess_fmt, not just catalog

        if fmt == "nova":
            body = build_nova_body(msgs, system, max_tok, temp)
        elif fmt == "llama":
            body = build_llama_body(msgs, system, max_tok, temp)
        elif fmt == "kimi":
            body = build_kimi_body(msgs, system, max_tok, temp)
        elif fmt == "mistral":
            body = build_mistral_body(msgs, system, max_tok, temp)
        else:
            body = build_claude_body(msgs, system, max_tok, temp)

        resp = client.invoke_model_with_response_stream(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        in_tok = out_tok = 0

        for event in resp["body"]:
            if "chunk" not in event:
                continue
            chunk = json.loads(event["chunk"]["bytes"])

            # ── Claude ──
            if chunk.get("type") == "content_block_delta":
                text = chunk.get("delta", {}).get("text", "")
                if text:
                    asyncio.run_coroutine_threadsafe(queue.put({"t": "tok", "v": text}), loop)
            elif chunk.get("type") == "message_start":
                u = chunk.get("message", {}).get("usage", {})
                in_tok = u.get("input_tokens", in_tok)
            elif chunk.get("type") == "message_delta":
                u = chunk.get("usage", {})
                out_tok = u.get("output_tokens", out_tok)

            # ── Nova ──
            elif "contentBlockDelta" in chunk:
                text = chunk["contentBlockDelta"].get("delta", {}).get("text", "")
                if text:
                    asyncio.run_coroutine_threadsafe(queue.put({"t": "tok", "v": text}), loop)
            elif "amazon-bedrock-invocationMetrics" in chunk:
                m = chunk["amazon-bedrock-invocationMetrics"]
                in_tok  = m.get("inputTokenCount", in_tok)
                out_tok = m.get("outputTokenCount", out_tok)

            # ── Kimi / Moonshot (OpenAI streaming format) ──
            elif "choices" in chunk:
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    # Normal content
                    text = delta.get("content") or delta.get("text", "")
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put({"t": "tok", "v": text}), loop)
                    # Thinking / reasoning content — wrap in tags so UI can render it
                    think = delta.get("thinking_content") or delta.get("reasoning_content", "")
                    if think:
                        asyncio.run_coroutine_threadsafe(queue.put({"t": "tok", "v": f"<thinking>{think}</thinking>"}), loop)
                if "usage" in chunk:
                    u = chunk["usage"]
                    in_tok  = u.get("prompt_tokens",     in_tok)
                    out_tok = u.get("completion_tokens",  out_tok)

            # ── Llama / Mistral ──
            elif "generation" in chunk:
                text = chunk.get("generation", "")
                if text:
                    asyncio.run_coroutine_threadsafe(queue.put({"t": "tok", "v": text}), loop)
                if "prompt_token_count" in chunk:
                    in_tok  = chunk["prompt_token_count"]
                    out_tok = chunk.get("generation_token_count", out_tok)
            elif "outputs" in chunk:
                for o in chunk.get("outputs", []):
                    text = o.get("text", "")
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put({"t": "tok", "v": text}), loop)

        asyncio.run_coroutine_threadsafe(
            queue.put({"t": "done", "in": in_tok, "out": out_tok}), loop
        )

    except Exception as e:
        asyncio.run_coroutine_threadsafe(
            queue.put({"t": "err", "v": traceback.format_exc(limit=3)}), loop
        )

# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    message: str
    files: List[Dict] = []

class NewConv(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None   # FIX: allow frontend to pass model on creation

class SettingsBody(BaseModel):
    settings: Dict[str, Any]

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    if HTML_FILE.exists():
        return HTML_FILE.read_text(encoding="utf-8")
    return "<h1>index.html not found alongside server.py</h1>"


@app.get("/api/conversations")
async def api_list_convs():
    return list_convs()


@app.post("/api/conversations")
async def api_new_conv(body: NewConv):
    s   = load_settings()
    cid = str(uuid.uuid4())
    c   = {
        "id":       cid,
        "title":    body.title or "New Chat",
        "created":  datetime.datetime.now().isoformat(),
        "updated":  datetime.datetime.now().isoformat(),
        "model":    body.model or s["main_model"],   # FIX: honour model from frontend
        "messages": [],
        "cost_inr": 0.0,
        "cost_usd": 0.0,
        "in_tok":   0,
        "out_tok":  0,
        "pinned":   False,
    }
    save_conv(c)
    return c


@app.get("/api/conversations/{cid}")
async def api_get_conv(cid: str):
    return load_conv(cid)


@app.delete("/api/conversations/{cid}")
async def api_del_conv(cid: str):
    p = cpath(cid)
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.delete("/api/conversations/{cid}/messages/{mid}")
async def api_delete_message(cid: str, mid: str):
    conv = load_conv(cid)
    original_len = len(conv.get("messages", []))
    conv["messages"] = [m for m in conv.get("messages", []) if m.get("id") != mid]
    if len(conv["messages"]) == original_len:
        raise HTTPException(404, "Message not found")
    recompute_conv_totals(conv)
    save_conv(conv)
    return {"ok": True}


@app.patch("/api/conversations/{cid}")
async def api_patch_conv(cid: str, req: Request):
    body = await req.json()
    c    = load_conv(cid)
    for k in ("title", "pinned", "model"):
        if k in body:
            c[k] = body[k]
    save_conv(c)
    return {"ok": True}


@app.post("/api/conversations/{cid}/pin")
async def api_pin(cid: str):
    c = load_conv(cid)
    c["pinned"] = not c.get("pinned", False)
    save_conv(c)
    return {"pinned": c["pinned"]}


@app.post("/api/conversations/{cid}/stream")
async def api_stream(cid: str, body: ChatReq):
    """SSE streaming endpoint — POST with message, stream tokens back."""
    s    = load_settings()
    conv = load_conv(cid)

    # FIX: enforce cost guard server-side (frontend-only guard is bypassable)
    if s.get("cost_guard_enabled", True):
        guard_inr = s.get("cost_guard_inr", 5.0)
        model_id_check = conv.get("model") or s["main_model"]
        ctx_check = build_context_messages(conv, model_id_check)
        in_tok_est = sum(approx_tokens(m.get("content", "")) for m in ctx_check) + approx_tokens(body.message)
        out_tok_est = min(
            s.get("max_tokens", 8192),
            max(512, round(approx_tokens(body.message) * 1.5))
        )
        est_cost = calc_cost(model_id_check, in_tok_est, out_tok_est)
        if est_cost["inr"] > guard_inr:
            raise HTTPException(402, f"Estimated cost ₹{est_cost['inr']:.2f} exceeds guard limit ₹{guard_inr:.2f}. Adjust in Settings.")

    # ── Attach user message ──
    user_msg = {
        "id":        str(uuid.uuid4()),
        "role":      "user",
        "content":   body.message,
        "timestamp": datetime.datetime.now().isoformat(),
        "files":     body.files,
    }
    if not conv["messages"]:
        conv["title"] = auto_title(body.message)
    conv["messages"].append(user_msg)
    save_conv(conv)

    # ── Build context ──
    model_id = conv.get("model") or s["main_model"]
    ctx_msgs = build_context_messages(conv, model_id)

    max_tok  = min(s.get("max_tokens", 8192), MODEL_CATALOG.get(model_id, {}).get("max_tokens", 8192))
    temp     = s.get("temperature", 0.7)
    # Project conversations carry a pre-built system prompt (instructions + injected docs)
    system   = conv_system_prompt(conv, s)
    region   = s.get("aws_region", "us-east-1")

    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    loop.run_in_executor(executor, sync_stream, ctx_msgs, model_id, system, max_tok, temp, region, queue, loop)

    async def generate():
        full = ""
        in_tok = out_tok = 0
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=180)
                if item["t"] == "tok":
                    full += item["v"]
                    yield f"data: {json.dumps({'type':'tok','v':item['v']})}\n\n"
                elif item["t"] == "done":
                    in_tok, out_tok = item["in"], item["out"]
                    break
                elif item["t"] == "err":
                    yield f"data: {json.dumps({'type':'err','v':item['v']})}\n\n"
                    return
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type':'err','v':'Request timed out after 3 minutes'})}\n\n"
            # FIX: save stub error message so conversation is not left in broken half-state
            stub = {
                "id":        str(uuid.uuid4()),
                "role":      "assistant",
                "content":   "[Response timed out after 3 minutes. Please try again.]",
                "timestamp": datetime.datetime.now().isoformat(),
                "in_tok":    0, "out_tok": 0, "cost_usd": 0.0, "cost_inr": 0.0,
                "model":     model_id, "error": True,
            }
            conv["messages"].append(stub)
            save_conv(conv)
            return

        cost  = calc_cost(model_id, in_tok, out_tok)
        ai_id = str(uuid.uuid4())
        ai_msg = {
            "id":        ai_id,
            "role":      "assistant",
            "content":   full,
            "timestamp": datetime.datetime.now().isoformat(),
            "in_tok":    in_tok,
            "out_tok":   out_tok,
            "cost_usd":  cost["usd"],
            "cost_inr":  cost["inr"],
            "model":     model_id,
        }
        conv["messages"].append(ai_msg)
        conv["cost_inr"] = conv.get("cost_inr", 0) + cost["inr"]
        conv["cost_usd"] = conv.get("cost_usd", 0) + cost["usd"]
        conv["in_tok"]   = conv.get("in_tok",   0) + in_tok
        conv["out_tok"]  = conv.get("out_tok",  0) + out_tok
        save_conv(conv)
        add_stats(in_tok, out_tok, cost["usd"])

        yield f"data: {json.dumps({'type':'done','id':ai_id,'in_tok':in_tok,'out_tok':out_tok,'cost_inr':cost['inr'],'title':conv['title']})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def make_chat_stream(conv: Dict, raw_messages: List[Dict], on_done, model_id: str = None):
    """Shared SSE streamer for regenerate, continue, and edited-message reruns."""
    s        = load_settings()
    model_id = model_id or conv.get("model") or s["main_model"]
    ctx_msgs = build_context_messages(conv, model_id, raw_messages)
    max_tok  = min(s.get("max_tokens", 8192), MODEL_CATALOG.get(model_id, {}).get("max_tokens", 8192))
    temp     = s.get("temperature", 0.7)
    system   = conv_system_prompt(conv, s)
    region   = s.get("aws_region", "us-east-1")
    loop     = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    loop.run_in_executor(executor, sync_stream, ctx_msgs, model_id, system, max_tok, temp, region, queue, loop)

    async def generate():
        full = ""
        in_tok = out_tok = 0
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=180)
                if item["t"] == "tok":
                    full += item["v"]
                    yield f"data: {json.dumps({'type':'tok','v':item['v']})}\n\n"
                elif item["t"] == "done":
                    in_tok, out_tok = item.get("in", 0), item.get("out", 0)
                    break
                elif item["t"] == "err":
                    yield f"data: {json.dumps({'type':'err','v':item.get('v','Unknown error')})}\n\n"
                    return
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type':'err','v':'Request timed out after 3 minutes'})}\n\n"
            # FIX: save stub so conversation stays consistent after timeout
            stub = {
                "id":        str(uuid.uuid4()),
                "role":      "assistant",
                "content":   "[Response timed out after 3 minutes. Please try again.]",
                "timestamp": datetime.datetime.now().isoformat(),
                "in_tok":    0, "out_tok": 0, "cost_usd": 0.0, "cost_inr": 0.0,
                "model":     model_id, "error": True,
            }
            on_done(stub)
            save_conv(conv)
            return

        cost = calc_cost(model_id, in_tok, out_tok)
        ai_msg = {
            "id":        str(uuid.uuid4()),
            "role":      "assistant",
            "content":   full,
            "timestamp": datetime.datetime.now().isoformat(),
            "in_tok":    in_tok,
            "out_tok":   out_tok,
            "cost_usd":  cost["usd"],
            "cost_inr":  cost["inr"],
            "model":     model_id,
        }
        extra = on_done(ai_msg) or {}
        recompute_conv_totals(conv)
        save_conv(conv)
        add_stats(in_tok, out_tok, cost["usd"])
        payload = {
            "type": "done",
            "id": ai_msg["id"],
            "in_tok": in_tok,
            "out_tok": out_tok,
            "cost_inr": cost["inr"],
            "title": conv.get("title", "New Chat"),
        }
        payload.update(extra)
        yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/conversations/{cid}/messages/{mid}/regenerate")
async def api_regenerate_message(cid: str, mid: str):
    conv = load_conv(cid)
    msgs = conv.get("messages", [])
    idx = next((i for i, m in enumerate(msgs) if m.get("id") == mid), None)
    if idx is None or msgs[idx].get("role") != "assistant":
        raise HTTPException(404, "Assistant message not found")
    old = dict(msgs[idx])
    raw_messages = msgs[:idx]

    def persist(ai_msg: Dict):
        ai_msg["id"] = old.get("id", ai_msg["id"])
        conv["messages"] = msgs[:idx] + [ai_msg]
        return {"replaced_id": ai_msg["id"]}

    return make_chat_stream(conv, raw_messages, persist, old.get("model") or conv.get("model"))


@app.post("/api/conversations/{cid}/messages/{mid}/respond")
async def api_respond_from_message(cid: str, mid: str):
    conv = load_conv(cid)
    msgs = conv.get("messages", [])
    idx = next((i for i, m in enumerate(msgs) if m.get("id") == mid), None)
    if idx is None or msgs[idx].get("role") != "user":
        raise HTTPException(404, "User message not found")
    raw_messages = msgs[:idx + 1]

    def persist(ai_msg: Dict):
        conv["messages"] = raw_messages + [ai_msg]

    return make_chat_stream(conv, raw_messages, persist)


@app.post("/api/conversations/{cid}/messages/{mid}/continue")
async def api_continue_message(cid: str, mid: str):
    conv = load_conv(cid)
    msgs = conv.get("messages", [])
    idx = next((i for i, m in enumerate(msgs) if m.get("id") == mid), None)
    if idx is None:
        raise HTTPException(404, "Message not found")
    continue_msg = {
        "id":        str(uuid.uuid4()),
        "role":      "user",
        "content":   "Continue from the previous answer. Do not repeat what you already wrote.",
        "timestamp": datetime.datetime.now().isoformat(),
        "files":     [],
    }
    raw_messages = msgs[:idx + 1] + [continue_msg]

    def persist(ai_msg: Dict):
        conv["messages"] = raw_messages + [ai_msg]

    return make_chat_stream(conv, raw_messages, persist)


@app.get("/api/models")
async def api_models():
    return [
        {
            "id":       mid,
            "name":     m["name"],
            "provider": m["provider"],
            "in_inr":   round(m["input"]  * USD_TO_INR, 6),
            "out_inr":  round(m["output"] * USD_TO_INR, 6),
            "max_tok":  m["max_tokens"],
        }
        for mid, m in MODEL_CATALOG.items()
    ]


@app.get("/api/settings")
async def api_get_settings():
    return load_settings()


@app.post("/api/settings")
async def api_save_settings(body: SettingsBody):
    cur = load_settings()
    cur.update(body.settings)
    save_settings(cur)
    return {"ok": True}


@app.get("/api/stats")
async def api_stats():
    return load_stats()


@app.get("/api/search")
async def api_search(q: str):
    ql  = q.lower()
    out = []
    for p in CONV_DIR.glob("*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            snips = []
            for m in c.get("messages", []):
                txt = m.get("content", "")
                if not isinstance(txt, str):
                    continue
                if ql in txt.lower():
                    idx   = txt.lower().find(ql)
                    start = max(0, idx - 60)
                    end   = min(len(txt), idx + 120)
                    snips.append({"role": m["role"], "snip": txt[start:end]})
            if snips:
                out.append({"conv_id": c["id"], "title": c.get("title",""), "snips": snips[:3]})
        except Exception:
            pass
    return out


@app.get("/api/conversations/{cid}/export")
async def api_export(cid: str):
    c     = load_conv(cid)
    lines = [f"# {c['title']}", f"*Exported {datetime.datetime.now():%Y-%m-%d %H:%M}*\n"]
    for m in c["messages"]:
        label = "**You**" if m["role"] == "user" else "**Assistant**"
        lines += [f"### {label}", m["content"], ""]
        if m["role"] == "assistant" and m.get("cost_inr"):
            lines.append(f"*↑{m.get('in_tok',0)} ↓{m.get('out_tok',0)} tokens | ₹{m['cost_inr']:.5f}*\n")
    md   = "\n".join(lines)
    name = re.sub(r"[^\w\s-]", "", c["title"])[:50] + ".md"
    return StreamingResponse(iter([md]), media_type="text/markdown",
                             headers={"Content-Disposition": f"attachment; filename=\"{name}\""})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    data = await file.read()
    mime = file.content_type or mimetypes.guess_type(file.filename or "")[0] or ""
    if mime.startswith("image/"):
        return {"type": "image", "name": file.filename, "media_type": mime,
                "data": base64.b64encode(data).decode()}
    exts = {".py",".js",".ts",".jsx",".tsx",".json",".md",".txt",".csv",".html",".css",
            ".java",".cpp",".c",".h",".go",".rs",".rb",".sh",".yaml",".yml",".toml",".xml"}
    if mime.startswith("text/") or pathlib.Path(file.filename or "").suffix in exts:
        return {"type": "text", "name": file.filename,
                "content": data.decode("utf-8", errors="replace")}
    raise HTTPException(400, f"Unsupported file type: {mime or file.filename}")


@app.patch("/api/conversations/{cid}/messages/{mid}")
async def api_annotate(cid: str, mid: str, req: Request):
    body = await req.json()
    c    = load_conv(cid)
    for i, m in enumerate(c["messages"]):
        if m["id"] == mid:
            if "annotation" in body:
                m["annotation"] = body.get("annotation", "")
            if "content" in body:
                m["content"] = body.get("content", "")
            if body.get("truncate_after"):
                c["messages"] = c["messages"][:i + 1]
                recompute_conv_totals(c)
            break
    save_conv(c)
    return {"ok": True}


# ── New endpoints added for UI improvements ─────────────────────────────────

@app.get("/api/aws-regions")
async def api_regions():
    """Return the list of AWS regions where Bedrock is available."""
    return BEDROCK_REGIONS


@app.get("/api/aws-models")
async def api_aws_models(region: str = None):
    """
    Fetch ALL models available in the user's AWS account via the Bedrock API.
    Falls back to the hardcoded catalog on credential / permission errors.
    """
    s = load_settings()
    r = region or s.get("aws_region", "us-east-1")

    def _build_entry(mid: str, name: str, provider: str) -> Dict:
        cat     = MODEL_CATALOG.get(mid, {})
        in_usd  = cat.get("input",  0.0)
        out_usd = cat.get("output", 0.0)
        return {
            "id":          mid,
            "name":        name,
            "provider":    provider,
            "in_inr":      round(in_usd  * USD_TO_INR, 6),
            "out_inr":     round(out_usd * USD_TO_INR, 6),
            "max_tok":     cat.get("max_tokens", 4096),
            "fmt":         cat.get("fmt", guess_fmt(mid)),
            "has_pricing": mid in MODEL_CATALOG,
        }

    try:
        client  = make_mgmt_client(r)
        models  = []
        seen    = set()

        # ── 1. Base foundation models ─────────────────────────────────────────
        resp = client.list_foundation_models(byOutputModality="TEXT")
        for m in resp.get("modelSummaries", []):
            mid   = m.get("modelId", "")
            modes = m.get("inferenceTypesSupported", [])
            if not mid or (modes and "ON_DEMAND" not in modes):
                continue
            if mid not in seen:
                seen.add(mid)
                models.append(_build_entry(
                    mid,
                    m.get("modelName", mid),
                    m.get("providerName", mid.split(".")[-2].capitalize() if "." in mid else mid.split(":")[0].capitalize()),
                ))

        # ── 2. Cross-region inference profiles (Kimi, newer Claude, Nova CR…) ─
        try:
            prof_resp = client.list_inference_profiles(typeEquals="SYSTEM_DEFINED")
            for p in prof_resp.get("inferenceProfileSummaries", []):
                pid  = p.get("inferenceProfileId", "") or p.get("inferenceProfileArn", "")
                if not pid:
                    continue
                # Normalise ARN → ID  (arn:aws:bedrock:us-east-1::foundation-model/us.moon... )
                if pid.startswith("arn:"):
                    pid = pid.split("/", 1)[-1]
                if pid in seen:
                    continue
                seen.add(pid)
                # Provider from ID  "us.moonshot.kimi-k2..." → "Moonshot"
                parts    = pid.split(".")
                prov_raw = parts[1] if len(parts) > 2 else parts[0]
                provider = prov_raw.replace("-", " ").title()
                name     = p.get("inferenceProfileName", pid)
                models.append(_build_entry(pid, name, provider))
        except Exception:
            pass  # list_inference_profiles may not be available in all regions

        if models:
            return sorted(models, key=lambda x: (x["provider"], x["name"]))
    except Exception:
        pass  # credentials missing or Bedrock not reachable → fall through

    # Fallback: hardcoded catalog
    return sorted(
        [_build_entry(mid, m["name"], m["provider"]) for mid, m in MODEL_CATALOG.items()],
        key=lambda x: (x["provider"], x["name"]),
    )


@app.get("/api/credentials")
async def api_get_credentials():
    """Return credential metadata — never exposes the secret key value."""
    creds = load_credentials()
    ak    = creds.get("aws_access_key_id", "")
    return {
        "has_keys":           bool(ak),
        "has_profile":        bool(creds.get("profile_name")),
        "has_session_token":  bool(creds.get("aws_session_token")),
        "aws_access_key_id":  (ak[:4] + "…" + ak[-4:]) if len(ak) > 8 else ("*"*len(ak) if ak else ""),
        "profile_name":       creds.get("profile_name", ""),
        "aws_region":         creds.get("aws_region", ""),
        "method":             "profile" if creds.get("profile_name") else ("keys" if ak else "default-chain"),
    }


@app.post("/api/credentials")
async def api_save_credentials(req: Request):
    """
    Save AWS credentials to disk.
    If secret_key is the placeholder "****", keep the existing stored value.
    """
    body     = await req.json()
    existing = load_credentials()

    sk = body.get("aws_secret_access_key", "")
    st = body.get("aws_session_token", "")

    creds = {
        "aws_access_key_id":     body.get("aws_access_key_id", "").strip(),
        "aws_secret_access_key": sk if sk not in ("****", "") else existing.get("aws_secret_access_key", ""),
        "aws_session_token":     st if st not in ("****", "") else existing.get("aws_session_token", ""),
        "profile_name":          body.get("profile_name", "").strip(),
        "aws_region":            body.get("aws_region", "").strip(),
    }
    save_credentials(creds)
    return {"ok": True}


@app.get("/api/test-connection")
async def api_test_connection(region: str = None):
    """Attempt a real AWS API call and report success / error."""
    import traceback as _tb
    try:
        r      = region or load_credentials().get("aws_region") or load_settings().get("aws_region","us-east-1")
        client = make_mgmt_client(r)
        resp   = client.list_foundation_models(byOutputModality="TEXT")
        count  = len(resp.get("modelSummaries", []))
        creds  = load_credentials()
        method = "profile" if creds.get("profile_name") else ("access keys" if creds.get("aws_access_key_id") else "default credential chain")
        return {"ok": True, "model_count": count, "region": r, "method": method}
    except Exception as e:
        return {"ok": False, "error": str(e), "detail": _tb.format_exc(limit=2)}



# ══════════════════════════════════════════════════════════════════════════════
# ── Branching, Templates, Compare, Draft endpoints ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Template helpers ──────────────────────────────────────────────────────────
def load_templates() -> list:
    if TMPL_FILE.exists():
        try: return json.loads(TMPL_FILE.read_text(encoding="utf-8"))
        except Exception: pass
    return []

def save_templates(t: list):
    TMPL_FILE.write_text(json.dumps(t, indent=2, ensure_ascii=False), encoding="utf-8")

def default_presets() -> list:
    s = load_settings()
    return [
        {"id": "preset_fast", "name": "Fast Draft", "model": s.get("summary_model"), "temperature": 0.5, "max_tokens": 2048, "system_prompt": s.get("system_prompt", "")},
        {"id": "preset_deep", "name": "Deep Dive", "model": s.get("main_model"), "temperature": 0.7, "max_tokens": 8192, "system_prompt": s.get("system_prompt", "")},
        {"id": "preset_precise", "name": "Precise Review", "model": s.get("main_model"), "temperature": 0.2, "max_tokens": 4096, "system_prompt": "Be precise, cite uncertainty, and separate facts from assumptions."},
    ]

def load_presets() -> list:
    if PRESET_FILE.exists():
        try:
            return json.loads(PRESET_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default_presets()

def save_presets(presets: list):
    PRESET_FILE.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")


@app.get("/api/templates")
async def api_get_templates():
    return load_templates()


@app.post("/api/templates")
async def api_upsert_template(req: Request):
    body = await req.json()
    templates = load_templates()
    if body.get("id"):
        templates = [body if t["id"] == body["id"] else t for t in templates]
    else:
        body["id"] = str(uuid.uuid4())
        templates.append(body)
    save_templates(templates)
    return body


@app.delete("/api/templates/{tid}")
async def api_delete_template(tid: str):
    save_templates([t for t in load_templates() if t["id"] != tid])
    return {"ok": True}


@app.get("/api/presets")
async def api_get_presets():
    return load_presets()


@app.post("/api/presets")
async def api_upsert_preset(req: Request):
    body = await req.json()
    presets = load_presets()
    if body.get("id"):
        presets = [body if p["id"] == body["id"] else p for p in presets]
    else:
        body["id"] = str(uuid.uuid4())
        presets.append(body)
    save_presets(presets)
    return body


@app.delete("/api/presets/{pid}")
async def api_delete_preset(pid: str):
    save_presets([p for p in load_presets() if p["id"] != pid])
    return {"ok": True}


# ── Draft endpoints ───────────────────────────────────────────────────────────
@app.post("/api/conversations/{cid}/draft")
async def api_save_draft(cid: str, req: Request):
    """Save an in-progress message draft alongside the conversation file."""
    body = await req.json()
    p = cpath(cid)
    if p.exists():
        c = json.loads(p.read_text(encoding="utf-8"))
        c["_draft"] = {"text": body.get("text",""), "files": body.get("files",[]),
                       "saved": datetime.datetime.now().isoformat()}
        p.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


@app.delete("/api/conversations/{cid}/draft")
async def api_clear_draft(cid: str):
    p = cpath(cid)
    if p.exists():
        c = json.loads(p.read_text(encoding="utf-8"))
        c.pop("_draft", None)
        p.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


# ── Branch endpoint ───────────────────────────────────────────────────────────
@app.post("/api/conversations/{cid}/branch")
async def api_branch(cid: str, req: Request):
    """Create a new conversation forked from a specific message in this one."""
    body     = await req.json()
    msg_id   = body.get("from_message_id")
    conv     = load_conv(cid)
    msgs     = conv.get("messages", [])

    # Slice up to and including the chosen message
    cut = len(msgs)
    for i, m in enumerate(msgs):
        if m["id"] == msg_id:
            cut = i + 1
            break

    s = load_settings()
    new_id = str(uuid.uuid4())
    now    = datetime.datetime.now().isoformat()
    branch = {
        "id":          new_id,
        "title":       "Branch: " + conv.get("title","Chat")[:35],
        "created":     now,
        "updated":     now,
        "model":       conv.get("model", s["main_model"]),
        "messages":    msgs[:cut],
        "cost_inr":    0.0,
        "cost_usd":    0.0,
        "in_tok":      0,
        "out_tok":     0,
        "pinned":      False,
        "parent_conv": cid,
        "branch_msg":  msg_id,
    }
    for k in ("project_id", "project_name", "injected_doc_ids", "project_system"):
        if k in conv:
            branch[k] = conv[k]
    save_conv(branch)
    return branch


# ── Multi-model comparison streaming ─────────────────────────────────────────
@app.post("/api/compare-stream")
async def api_compare_stream(req: Request):
    """
    Send one prompt to 2-3 models simultaneously.
    Returns a merged SSE stream tagged with model IDs.
    """
    body    = await req.json()
    models  = body.get("model_ids", [])[:3]
    msg     = body.get("message", "")
    s       = load_settings()
    region  = s.get("aws_region", "us-east-1")
    # FIX: use conversation's system prompt and settings if conv_id is provided
    conv_id = body.get("conv_id")
    if conv_id:
        try:
            conv   = load_conv(conv_id)
            system = conv_system_prompt(conv, s)
            temp   = s.get("temperature", 0.7)
        except Exception:
            system = s.get("system_prompt", "")
            temp   = s.get("temperature", 0.7)
    else:
        system = s.get("system_prompt", "")
        temp   = s.get("temperature", 0.7)
    max_t   = min(s.get("max_tokens", 4096), 4096)   # cap for compare to keep costs down
    context = [{"role": "user", "content": msg}]
    loop    = asyncio.get_running_loop()   # FIX: use get_running_loop (3.10+ safe)

    async def generate():
        queues: Dict[str, asyncio.Queue] = {mid: asyncio.Queue() for mid in models}
        done:   Dict[str, bool]          = {mid: False           for mid in models}

        for mid in models:
            loop.run_in_executor(
                executor, sync_stream,
                context, mid, system, max_t, temp, region, queues[mid], loop
            )

        timed_out = 0
        while not all(done.values()):
            changed = False
            for mid in models:
                if done[mid]: continue
                try:
                    item = queues[mid].get_nowait()
                    changed = True
                    SEP = "\n\n"
                    if item["t"] == "tok":
                        payload = json.dumps({"mid":mid,"t":"tok","v":item["v"]})
                        yield f"data: {payload}" + SEP
                    elif item["t"] == "done":
                        done[mid] = True
                        payload = json.dumps({"mid":mid,"t":"done","in":item.get("in",0),"out":item.get("out",0)})
                        yield f"data: {payload}" + SEP
                    elif item["t"] == "err":
                        done[mid] = True
                        payload = json.dumps({"mid":mid,"t":"err","v":item.get("v","")})
                        yield f"data: {payload}" + SEP
                except asyncio.QueueEmpty:
                    pass
            if not changed:
                await asyncio.sleep(0.015)
                timed_out += 1
                if timed_out > 12000:   # 3-minute hard timeout
                    break
        for mid, is_done in done.items():
            if not is_done:
                payload = json.dumps({"mid": mid, "t": "err", "v": "Compare timed out"})
                yield f"data: {payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── PROJECT endpoints  (chat_data/projects/)  ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def proj_path(pid: str) -> pathlib.Path:
    pid = safe_id(pid, "project id")
    return PROJ_DIR / f"{pid}.json"

def load_project(pid: str) -> Dict:
    p = proj_path(pid)
    if not p.exists():
        raise HTTPException(404, f"Project {pid} not found")
    return json.loads(p.read_text(encoding="utf-8"))

def save_project(proj: Dict):
    proj["updated"] = datetime.datetime.now().isoformat()
    proj_path(proj["id"]).write_text(
        json.dumps(proj, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def list_projects() -> List[Dict]:
    out = []
    for p in PROJ_DIR.glob("*.json"):
        try:
            proj = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "id":        proj["id"],
                "name":      proj.get("name", "Untitled Project"),
                "updated":   proj.get("updated", ""),
                "doc_count": len(proj.get("documents", [])),
                "model":     proj.get("model", ""),   # FIX: include model so sidebar can display it
            })
        except Exception:
            pass
    return sorted(out, key=lambda x: x["updated"], reverse=True)

def build_project_system(proj: Dict, doc_ids: List[str], base_system: str = "") -> str:
    parts: List[str] = []
    if base_system:
        parts.append(base_system.strip())
    if proj.get("instructions"):
        parts.append(proj["instructions"].strip())
    for doc in proj.get("documents", []):
        if doc["id"] in doc_ids:
            parts.append(f"---\n**{doc['name']}**\n\n{doc['content']}")
    if not parts:
        return base_system
    return "\n\n".join(parts)


@app.get("/api/projects")
async def api_list_projects():
    return list_projects()


@app.post("/api/projects")
async def api_create_project(req: Request):
    body = await req.json()
    pid  = str(uuid.uuid4())
    now  = datetime.datetime.now().isoformat()
    s    = load_settings()   # FIX: load settings to get default model
    proj = {
        "id":           pid,
        "name":         body.get("name", "New Project"),
        "instructions": body.get("instructions", ""),
        "model":        body.get("model", s["main_model"]),   # FIX: store model on project
        "created":      now,
        "updated":      now,
        "documents":    [],
    }
    save_project(proj)
    return proj


@app.get("/api/projects/{pid}")
async def api_get_project(pid: str):
    return load_project(pid)


@app.patch("/api/projects/{pid}")
async def api_update_project(pid: str, req: Request):
    body = await req.json()
    proj = load_project(pid)
    for k in ("name", "instructions", "model"):   # FIX: allow model to be updated
        if k in body:
            proj[k] = body[k]
    save_project(proj)
    return {"ok": True}


@app.delete("/api/projects/{pid}")
async def api_delete_project(pid: str):
    p = proj_path(pid)
    if p.exists():
        p.unlink()
    for cp in CONV_DIR.glob("*.json"):
        try:
            conv = json.loads(cp.read_text(encoding="utf-8"))
            if conv.get("project_id") == pid:
                conv.pop("project_id", None)
                conv.pop("project_name", None)
                conv.pop("injected_doc_ids", None)
                conv.pop("project_system", None)
                cp.write_text(json.dumps(conv, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/projects/{pid}/documents")
async def api_add_doc(pid: str, req: Request):
    body = await req.json()
    proj = load_project(pid)
    doc  = {
        "id":      str(uuid.uuid4()),
        "name":    body.get("name", "document.md"),
        "content": body.get("content", ""),
        "type":    body.get("type", "text"),   # text | summary | upload
        "created": datetime.datetime.now().isoformat(),
    }
    proj.setdefault("documents", []).append(doc)
    save_project(proj)
    return doc


@app.patch("/api/projects/{pid}/documents/{did}")
async def api_update_doc(pid: str, did: str, req: Request):
    body = await req.json()
    proj = load_project(pid)
    for doc in proj.get("documents", []):
        if doc["id"] == did:
            for k in ("name", "content"):
                if k in body:
                    doc[k] = body[k]
            break
    save_project(proj)
    # FIX: rebuild project_system in all linked conversations so they see updated doc content
    base_system = load_settings().get("system_prompt", "")
    for cp in CONV_DIR.glob("*.json"):
        try:
            conv = json.loads(cp.read_text(encoding="utf-8"))
            if conv.get("project_id") == pid:
                conv["project_system"] = build_project_system(proj, conv.get("injected_doc_ids", []), base_system)
                cp.write_text(json.dumps(conv, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True}


@app.delete("/api/projects/{pid}/documents/{did}")
async def api_delete_doc(pid: str, did: str):
    proj = load_project(pid)
    proj["documents"] = [d for d in proj.get("documents", []) if d["id"] != did]
    save_project(proj)
    # FIX: rebuild project_system in all linked conversations so deleted doc is no longer injected
    base_system = load_settings().get("system_prompt", "")
    for cp in CONV_DIR.glob("*.json"):
        try:
            conv = json.loads(cp.read_text(encoding="utf-8"))
            if conv.get("project_id") == pid:
                # Also remove the deleted doc from injected_doc_ids
                conv["injected_doc_ids"] = [d for d in conv.get("injected_doc_ids", []) if d != did]
                conv["project_system"] = build_project_system(proj, conv["injected_doc_ids"], base_system)
                cp.write_text(json.dumps(conv, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/projects/{pid}/conversations")
async def api_new_project_conv(pid: str, req: Request):
    """Create a new conversation inside a project with selected documents injected."""
    body      = await req.json()
    proj      = load_project(pid)
    s         = load_settings()
    doc_ids   = body.get("document_ids", [])
    cid       = str(uuid.uuid4())
    now       = datetime.datetime.now().isoformat()
    sys_text  = build_project_system(proj, doc_ids, s.get("system_prompt",""))
    conv = {
        "id":               cid,
        "title":            body.get("title", "New Chat"),
        "created":          now,
        "updated":          now,
        "model":            body.get("model") or proj.get("model") or s["main_model"],   # FIX: inherit from project
        "messages":         [],
        "cost_inr":         0.0,
        "cost_usd":         0.0,
        "in_tok":           0,
        "out_tok":          0,
        "pinned":           False,
        "project_id":       pid,
        "project_name":     proj.get("name", ""),
        "injected_doc_ids": doc_ids,
        "project_system":   sys_text,
    }
    save_conv(conv)
    return conv


@app.get("/api/projects/{pid}/conversations")
async def api_project_convs(pid: str):
    return [c for c in list_convs() if c.get("project_id") == pid]


@app.patch("/api/conversations/{cid}/project-context")
async def api_update_project_context(cid: str, req: Request):
    body = await req.json()
    conv = load_conv(cid)
    pid = body.get("project_id") or conv.get("project_id")
    if not pid:
        raise HTTPException(400, "Conversation is not linked to a project")
    proj = load_project(pid)
    doc_ids = body.get("document_ids", [])
    conv["project_id"] = pid
    conv["project_name"] = proj.get("name", "")
    conv["injected_doc_ids"] = doc_ids
    conv["project_system"] = build_project_system(proj, doc_ids, load_settings().get("system_prompt", ""))
    save_conv(conv)
    return {"ok": True, "conversation": conv}


@app.get("/api/conversations/{cid}/notes")
async def api_get_notes(cid: str):
    return load_conv(cid).get("notes", [])


@app.post("/api/conversations/{cid}/notes")
async def api_add_note(cid: str, req: Request):
    body = await req.json()
    conv = load_conv(cid)
    note = {
        "id": str(uuid.uuid4()),
        "text": body.get("text", ""),
        "source": body.get("source", "manual"),
        "pinned": bool(body.get("pinned", True)),
        "created": datetime.datetime.now().isoformat(),
    }
    conv.setdefault("notes", []).append(note)
    save_conv(conv)
    return note


@app.patch("/api/conversations/{cid}/notes/{nid}")
async def api_update_note(cid: str, nid: str, req: Request):
    body = await req.json()
    conv = load_conv(cid)
    for note in conv.setdefault("notes", []):
        if note.get("id") == nid:
            for k in ("text", "source", "pinned"):
                if k in body:
                    note[k] = body[k]
            note["updated"] = datetime.datetime.now().isoformat()
            break
    save_conv(conv)
    return {"ok": True}


@app.delete("/api/conversations/{cid}/notes/{nid}")
async def api_delete_note(cid: str, nid: str):
    conv = load_conv(cid)
    conv["notes"] = [n for n in conv.get("notes", []) if n.get("id") != nid]
    save_conv(conv)
    return {"ok": True}


@app.post("/api/conversations/{cid}/compare-save")
async def api_save_compare_response(cid: str, req: Request):
    body = await req.json()
    conv = load_conv(cid)
    prompt = body.get("prompt", "").strip()
    response = body.get("response", "").strip()
    if not prompt or not response:
        raise HTTPException(400, "Prompt and response are required")
    model_id = body.get("model_id") or conv.get("model") or load_settings()["main_model"]
    in_tok = int(body.get("in_tok", approx_tokens(prompt)) or 0)
    out_tok = int(body.get("out_tok", approx_tokens(response)) or 0)
    cost = calc_cost(model_id, in_tok, out_tok)
    now = datetime.datetime.now().isoformat()
    if not conv.get("messages") or conv["messages"][-1].get("content") != prompt:
        conv.setdefault("messages", []).append({
            "id": str(uuid.uuid4()), "role": "user", "content": prompt,
            "timestamp": now, "files": [], "from_compare": True,
        })
    conv["messages"].append({
        "id": str(uuid.uuid4()), "role": "assistant", "content": response,
        "timestamp": now, "model": model_id, "in_tok": in_tok, "out_tok": out_tok,
        "cost_usd": cost["usd"], "cost_inr": cost["inr"], "from_compare": True,
    })
    recompute_conv_totals(conv)
    save_conv(conv)
    return {"ok": True}


@app.post("/api/conversations/{cid}/estimate")
async def api_estimate_cost(cid: str, req: Request):
    body = await req.json()
    conv = load_conv(cid)
    s = load_settings()
    model_id = conv.get("model") or s["main_model"]
    draft = {
        "id": "estimate",
        "role": "user",
        "content": body.get("message", ""),
        "timestamp": datetime.datetime.now().isoformat(),
        "files": body.get("files", []),
    }
    ctx = build_context_messages(conv, model_id, conv.get("messages", []) + [draft])
    in_tok = sum(approx_tokens(m.get("content", "")) for m in ctx) + approx_tokens(conv_system_prompt(conv, s))
    max_out = min(s.get("max_tokens", 8192), MODEL_CATALOG.get(model_id, {}).get("max_tokens", 8192))
    est_out = min(max_out, max(512, round(approx_tokens(body.get("message", "")) * 1.5)))
    cost = calc_cost(model_id, in_tok, est_out)
    return {
        "model_id": model_id,
        "input_tokens": in_tok,
        "estimated_output_tokens": est_out,
        "estimated_cost_inr": cost["inr"],
        "guard_enabled": s.get("cost_guard_enabled", True),
        "guard_inr": s.get("cost_guard_inr", 5.0),
    }


# ── Summary generation endpoint ───────────────────────────────────────────────
@app.post("/api/summarize")
async def api_summarize(req: Request):
    """
    Receives curated snippets + a model ID.
    Returns a structured markdown summary.
    Temperature is kept low (0.2) to reduce hallucination.
    """
    body     = await req.json()
    snippets = body.get("snippets", [])
    model_id = body.get("model_id") or load_settings().get("summary_model", "amazon.nova-micro-v1:0")

    if not snippets:
        raise HTTPException(400, "No snippets provided")

    snippet_text = "\n\n---\n".join(
        f"[{s.get('source','')}]\n{s['text']}" for s in snippets if s.get("text","").strip()
    )

    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt = f"""You are a precise documentation assistant.
Below are HAND-SELECTED excerpts from a session. The user chose these specifically.

RULES:
- Do NOT add, infer, or hallucinate anything not in the excerpts
- Preserve numbers, formulas, and technical terms EXACTLY as written
- If a section has nothing in the excerpts, write: _(none)_
- Output ONLY the markdown below, nothing else

EXCERPTS:
{snippet_text}

---
## Session Summary — {today}

### Established Facts
(confirmed findings — keep exact wording for critical items)

### Key Numbers / Data
(exact figures, scores, formulas — zero paraphrasing)

### Decisions Made
(what was concluded and why)

### Open Questions
(unexplored or partially explored threads)

### Corrections / Dead Ends
(what was tried and ruled out)

### Next Steps
(what to investigate in the next session)
"""

    s      = load_settings()
    region = s.get("aws_region", "us-east-1")
    loop   = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    msgs = [{"role": "user", "content": prompt}]
    loop.run_in_executor(executor, sync_stream, msgs, model_id, "",
                         min(4096, MODEL_CATALOG.get(model_id,{}).get("max_tokens",4096)),
                         0.2, region, queue, loop)

    full = ""
    try:
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=120)
            if item["t"] == "tok":
                full += item["v"]
            elif item["t"] in ("done", "err"):
                break
    except asyncio.TimeoutError:
        pass

    return {"summary": full.strip(), "model_id": model_id}


@app.post("/api/conversations/{cid}/summary")
async def api_conversation_summary(cid: str, req: Request):
    body = await req.json()
    conv = load_conv(cid)
    s = load_settings()
    model_id = body.get("model_id") or s.get("summary_model", "amazon.nova-micro-v1:0")
    transcript = []
    for m in conv.get("messages", []):
        role = "User" if m.get("role") == "user" else "Assistant"
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        transcript.append(f"{role}: {content}")
    if not transcript:
        raise HTTPException(400, "Conversation has no messages")
    prompt = f"""Create a concise summary for this conversation.
Return markdown with these sections:
- Established facts
- Decisions / conclusions
- Important numbers or code details
- Open questions
- Next steps

Do not invent details outside the transcript.

Conversation title: {conv.get('title','New Chat')}

Transcript:
{chr(10).join(transcript)}
"""
    region = s.get("aws_region", "us-east-1")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    msgs = [{"role": "user", "content": prompt}]
    loop.run_in_executor(
        executor, sync_stream, msgs, model_id, "",
        min(4096, MODEL_CATALOG.get(model_id, {}).get("max_tokens", 4096)),
        0.2, region, queue, loop
    )
    full = ""
    try:
        while True:
            item = await asyncio.wait_for(queue.get(), timeout=180)
            if item["t"] == "tok":
                full += item["v"]
            elif item["t"] == "err":
                raise HTTPException(500, item.get("v", "Summary failed"))
            elif item["t"] == "done":
                break
    except asyncio.TimeoutError:
        raise HTTPException(504, "Summary timed out")
    return {"summary": full.strip(), "model_id": model_id}


@app.get("/api/backup")
async def api_backup():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = BACKUP_DIR / f"bedrock_chat_backup_{ts}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in DATA_DIR.rglob("*"):
            if p.is_file() and BACKUP_DIR not in p.parents:
                z.write(p, p.relative_to(DATA_DIR))
    return FileResponse(out, filename=out.name, media_type="application/zip")


@app.post("/api/restore")
async def api_restore(file: UploadFile = File(...)):
    data = await file.read()
    tmp = BACKUP_DIR / f"restore_{uuid.uuid4()}.zip"
    tmp.write_bytes(data)
    restored = 0
    try:
        with zipfile.ZipFile(tmp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                target = (DATA_DIR / info.filename).resolve()
                try:
                    target.relative_to(DATA_DIR.resolve())
                except ValueError:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src:
                    target.write_bytes(src.read())
                restored += 1
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass
    return {"ok": True, "restored_files": restored}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import socket
    import uvicorn

    host = "0.0.0.0"
    port = 8000
    local_ip = socket.gethostbyname(socket.gethostname())

    print("\n" + "=" * 58)
    print("  Bedrock Chat - Local Web UI")
    print("=" * 58)
    print(f"  Local :  http://localhost:{port}")
    print(f"  LAN   :  http://{local_ip}:{port}")
    print(f"  Data  :  {DATA_DIR.resolve()}")
    print("  All chats, credentials, projects, and backups save here")
    print("  Press Ctrl+C to stop")
    print("=" * 58 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")
