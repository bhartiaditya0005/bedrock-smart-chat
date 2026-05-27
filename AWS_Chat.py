#!/usr/bin/env python3
"""
============================================================================
Persistent Bedrock Chat with Cost Optimization & Smart Context Management
============================================================================

This script provides an interactive chat interface using AWS Bedrock with:
- Smart cost optimization: uses a cheap model for summarization and an
  expensive model for high-quality responses
- Persistent conversation history saved to disk
- Automatic context compression for long conversations
- Token usage tracking and cost estimation
- Session statistics and export capabilities

Requirements:
    pip install boto3

AWS Setup:
    - Configure AWS credentials (aws configure) or use environment variables
    - Ensure your IAM role has bedrock:InvokeModel permission
    - Enable the models in your AWS Bedrock console for your region

Author: [Your Name]
Version: 2.1
============================================================================
"""

import json
import os
import re
import select
import shutil
import sys
import textwrap
import time
import datetime
import copy
import hashlib
from pathlib import Path
from typing import Optional, Dict, List

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
)

# ============================================================================
# COLOR SUPPORT - ANSI codes with vibrant, varied colors (no external library needed)
# ============================================================================

class Colors:
    """ANSI color codes for terminal output with vibrant theme."""
    # Reset
    RESET = '\033[0m'
    
    # Bright/Vibrant Colors (Primary palette)
    BRIGHT_CYAN = '\033[96m'      # Vibrant cyan
    BRIGHT_MAGENTA = '\033[95m'   # Hot magenta
    BRIGHT_GREEN = '\033[92m'     # Lime green
    BRIGHT_YELLOW = '\033[93m'    # Vibrant yellow
    BRIGHT_BLUE = '\033[94m'      # Royal blue
    BRIGHT_RED = '\033[91m'       # Crimson red
    BRIGHT_WHITE = '\033[97m'     # Pure white
    
    # Extended Vibrant Colors (Secondary palette)
    NEON_PINK = '\033[38;5;206m'   # Neon pink
    NEON_PURPLE = '\033[38;5;135m' # Neon purple
    NEON_ORANGE = '\033[38;5;214m' # Neon orange
    NEON_LIME = '\033[38;5;118m'   # Neon lime
    NEON_AZURE = '\033[38;5;51m'   # Neon azure
    NEON_VIOLET = '\033[38;5;177m' # Neon violet
    
    # Regular Colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background Colors (Vibrant)
    BG_CYAN = '\033[106m'       # Bright cyan background
    BG_MAGENTA = '\033[105m'    # Bright magenta background
    BG_BLUE = '\033[104m'       # Bright blue background
    BG_GREEN = '\033[102m'      # Bright green background
    BG_YELLOW = '\033[103m'     # Bright yellow background
    BG_RED = '\033[101m'        # Bright red background
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    
    # Helpers with vibrant themes
    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ {text}{Colors.RESET}"
    
    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.BRIGHT_RED}{Colors.BOLD}✗ {text}{Colors.RESET}"
    
    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.NEON_ORANGE}{Colors.BOLD}⚠ {text}{Colors.RESET}"
    
    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.NEON_AZURE}{Colors.BOLD}ℹ {text}{Colors.RESET}"
    
    @staticmethod
    def header(text: str) -> str:
        return f"{Colors.BOLD}{Colors.NEON_PURPLE}{text}{Colors.RESET}"
    
    @staticmethod
    def user_msg(text: str) -> str:
        return f"{Colors.NEON_LIME}{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def ai_msg(text: str) -> str:
        return f"{Colors.NEON_AZURE}{text}{Colors.RESET}"
    
    @staticmethod
    def section(text: str) -> str:
        return f"{Colors.NEON_PINK}{Colors.BOLD}{text}{Colors.RESET}"


def terminal_width(default: int = 100) -> int:
    """Return a practical terminal width for readable wrapping."""
    return max(72, min(120, shutil.get_terminal_size((default, 24)).columns))


def strip_reasoning_blocks(text: str) -> str:
    """Remove provider reasoning/thinking markup from user-visible answers.
    No-op when SHOW_THINKING is True so the full chain-of-thought is visible."""
    if not text:
        return ""
    if SHOW_THINKING:
        return text  # User wants to see the thinking — leave it untouched.

    cleaned = str(text)
    hidden_patterns = (
        r"<reasoning\b[^>]*>.*?</reasoning>",
        r"<think\b[^>]*>.*?</think>",
    )
    for pattern in hidden_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove stray tags without deleting nearby final-answer text.
    cleaned = re.sub(r"</?(?:reasoning|think)\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_answer_text(text: str) -> str:
    """Clean model output before display, recovery saves, and history saves."""
    cleaned = strip_reasoning_blocks(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned.strip()


class ReasoningStreamFilter:
    """Streaming filter that hides <reasoning>/<think> blocks as they arrive."""

    OPEN_TAGS = ("<reasoning", "<think")
    CLOSE_TAGS = ("</reasoning>", "</think>")

    def __init__(self):
        self.buffer = ""
        self.hidden = False

    def feed(self, text: str) -> str:
        data = self.buffer + (text or "")
        self.buffer = ""
        visible = []

        while data:
            lower = data.lower()

            if self.hidden:
                close_positions = [(lower.find(tag), tag) for tag in self.CLOSE_TAGS if lower.find(tag) != -1]
                if not close_positions:
                    self.buffer = data[-32:]
                    return "".join(visible)
                close_index, close_tag = min(close_positions, key=lambda item: item[0])
                data = data[close_index + len(close_tag):]
                self.hidden = False
                continue

            open_positions = [(lower.find(tag), tag) for tag in self.OPEN_TAGS if lower.find(tag) != -1]
            if not open_positions:
                safe_text, self.buffer = self._split_possible_partial_tag(data)
                visible.append(self._remove_stray_reasoning_tags(safe_text))
                break

            open_index, _ = min(open_positions, key=lambda item: item[0])
            visible.append(self._remove_stray_reasoning_tags(data[:open_index]))

            tag_end = data.find(">", open_index)
            if tag_end == -1:
                self.buffer = data[open_index:]
                break

            data = data[tag_end + 1:]
            self.hidden = True

        return "".join(visible)

    def flush(self) -> str:
        if self.hidden:
            self.buffer = ""
            self.hidden = False
            return ""
        remaining = self._remove_stray_reasoning_tags(self.buffer)
        self.buffer = ""
        return remaining

    @staticmethod
    def _remove_stray_reasoning_tags(text: str) -> str:
        return re.sub(r"</?(?:reasoning|think)\b[^>]*>", "", text, flags=re.IGNORECASE)

    @staticmethod
    def _split_possible_partial_tag(text: str) -> tuple[str, str]:
        last_lt = text.rfind("<")
        if last_lt == -1:
            return text, ""

        tail = text[last_lt:].lower()
        if ">" in tail:
            return text, ""
        possible_prefixes = (
            "<r", "<re", "<rea", "<reas", "<reaso", "<reason", "<reasoni",
            "<reasonin", "<reasoning", "</r", "</re", "</rea", "</reas",
            "</reaso", "</reason", "</reasoni", "</reasonin", "</reasoning",
            "<t", "<th", "<thi", "<thin", "<think", "</t", "</th", "</thi",
            "</thin", "</think",
        )
        if any(prefix.startswith(tail) or tail.startswith(prefix) for prefix in possible_prefixes):
            return text[:last_lt], text[last_lt:]
        return text, ""

def atomic_write_json(path: str | Path, data: dict):
    """Write JSON via temp file + replace so sudden shutdown rarely corrupts files."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True) if target.parent != Path(".") else None
    temp = target.with_suffix(target.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(temp, target)


def bedrock_client_config() -> Config:
    """Retry and timeout config for unstable internet connections."""
    return Config(
        connect_timeout=AWS_CONNECT_TIMEOUT,
        read_timeout=AWS_READ_TIMEOUT,
        retries={"max_attempts": AWS_MAX_ATTEMPTS, "mode": "adaptive"},
    )


# ============================================================================
# CONFIGURATION - These are just DEFAULTS. You can change them at runtime!
# ============================================================================

# Get the script's directory for saving all files
SCRIPT_DIR = Path(__file__).parent

# AWS Region - Default to us-east-1 (most models available here)
AWS_REGION = "us-east-1"

# Default models (optimized for us-east-1 availability)
EXPENSIVE_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
CHEAP_MODEL_ID = "amazon.nova-micro-v1:0"  # Changed from nova-lite (more available)

# --- Context Management Settings ---
RECENT_MESSAGES_COUNT = 5
MAX_MESSAGES_BEFORE_SUMMARY = 10
CONTEXT_MODE = "smart"  # "smart" summarizes old context, "full" sends full conversation
STREAM_RESPONSES = True

# --- File Settings (all save to script directory) ---
HISTORY_FILE = str(SCRIPT_DIR / "bedrock_chat_history.json")
EXPORT_DIR = str(SCRIPT_DIR / "chat_exports")
SETTINGS_FILE = str(SCRIPT_DIR / "bedrock_chat_settings.json")
CHECKPOINT_DIR = str(SCRIPT_DIR / "chat_checkpoints")
REQUEST_STATE_FILE = str(SCRIPT_DIR / "bedrock_chat_in_progress.json")
AUTOSAVE_CHECKPOINT_EVERY = 8

# --- Reliability Settings ---
AWS_MAX_ATTEMPTS = 8
AWS_CONNECT_TIMEOUT = 10
AWS_READ_TIMEOUT = 300
APP_RETRY_ATTEMPTS = 4
APP_RETRY_BASE_DELAY = 2
PARTIAL_SAVE_EVERY_CHARS = 160

# --- Model Parameters ---
TEMPERATURE = 0.7
# Output token limits – set to each model's documented maximum.
# Claude 3.5/3.7 Sonnet → 8 192 | Nova Micro/Lite/Pro → 5 120 | Haiku/Opus → 4 096
# The actual cap per call is further constrained by the model catalog value (see
# _get_model_max_output_tokens helper), so you never exceed what the model allows.
MAX_TOKENS_EXPENSIVE = 8192
MAX_TOKENS_CHEAP = 5120

# --- Thinking / Reasoning Visibility ---
# Set True  → show the model's <thinking>/<reasoning> blocks during streaming and in
#              non-streamed responses so you can audit the model's logic step-by-step.
# Set False → suppress reasoning blocks (original behaviour).
# Can also be toggled at runtime via the /think command.
SHOW_THINKING = True

# ============================================================================
# DYNAMIC MODEL CATALOG - Loads models from AWS Bedrock
# ============================================================================

class ModelCatalog:
    """Dynamically loads ALL available models from AWS Bedrock."""
    
    def __init__(self, client=None):
        self.client = client
        self.catalog = {}
        self.last_updated = None
        
    def refresh(self, region=None):
        """Fetch all available models from AWS Bedrock for the current region."""
        try:
            if not self.client:
                # Create a Bedrock control-plane client just for listing models.
                session = boto3.Session(region_name=region or AWS_REGION)
                bedrock_client = session.client('bedrock')
            else:
                bedrock_client = self.client
            
            print("🔄 Fetching available models from AWS Bedrock...")
            
            models = []
            request = {"byInferenceType": "ON_DEMAND"}
            
            # list_foundation_models is not pageable in every boto3 version, so
            # prefer a paginator when available and fall back to a plain call.
            try:
                paginator = bedrock_client.get_paginator('list_foundation_models')
                for page in paginator.paginate(**request):
                    models.extend(page.get('modelSummaries', []))
            except Exception:
                try:
                    response = bedrock_client.list_foundation_models(**request)
                except Exception:
                    response = bedrock_client.list_foundation_models()
                models.extend(response.get('modelSummaries', []))
            
            print(f"✅ Found {len(models)} models in {region or AWS_REGION}")
            
            # Build catalog with pricing
            self.catalog = {}
            for model in models:
                model_id = model['modelId']
                
                # Extract short key from model ID
                short_key = self._create_short_key(model_id)
                
                provider = model.get('providerName', 'Unknown')
                model_name = model.get('modelName', model_id)
                
                # Determine pricing category based on model name and provider.
                pricing = self._estimate_pricing(model_id, model_name, provider)
                
                self.catalog[short_key] = {
                    "model_id": model_id,
                    "display_name": model_name,
                    "provider": provider,
                    "category": pricing['category'],
                    "input_cost_per_1k": pricing['input_cost'],
                    "output_cost_per_1k": pricing['output_cost'],
                    "max_tokens": self._estimate_max_tokens(model_id, model_name),
                    "description": self._get_description(model_id, model_name),
                    "api_format": self._detect_api_format(model_id, provider),
                }
            
            self.last_updated = datetime.datetime.now().isoformat()
            self._sync_global_catalog()
            return True
            
        except Exception as e:
            print(f"⚠️  Could not fetch models: {e}")
            # Fallback to a minimal catalog
            self._load_fallback_catalog()
            return False
    
    def _create_short_key(self, model_id: str) -> str:
        """Create a user-friendly short key from model ID."""
        # Remove common prefixes
        short = model_id.lower()
        if short == "deepseek.v3.2":
            return "deepseek-v3.2"
        if short == "deepseek.v3-v1:0":
            return "deepseek-v3.1"
        if "deepseek.r1" in short or "deepseek.r1" in short.replace("-", "."):
            return "deepseek-r1"
        short = short.replace("anthropic.", "").replace("amazon.", "").replace("meta.", "")
        short = short.replace("mistral.", "").replace("cohere.", "").replace("ai21.", "")
        short = short.replace("deepseek.", "")
        short = short.replace(":0", "").replace(":1", "").replace("-v1", "").replace("-v2", "")
        short = short.replace("_", "-").replace(".", "-")
        return short
    
    def _estimate_pricing(self, model_id: str, model_name: str, provider: str = "") -> dict:
        """Estimate pricing based on model name and provider."""
        model_lower = model_id.lower()
        name_lower = model_name.lower()
        provider_lower = provider.lower()
        
        # DeepSeek pricing from the AWS Bedrock pricing page. Values are per 1K tokens.
        if "deepseek" in model_lower or "deepseek" in name_lower or "deepseek" in provider_lower:
            if "ap-southeast-2" in AWS_REGION:
                return {"input_cost": 0.0006386, "output_cost": 0.0019055, "category": "mid"}
            if AWS_REGION in {"ap-south-1", "ap-northeast-1", "ap-southeast-3", "eu-north-1", "sa-east-1"}:
                return {"input_cost": 0.00074, "output_cost": 0.00222, "category": "mid"}
            return {"input_cost": 0.00062, "output_cost": 0.00185, "category": "mid"}
        
        # Claude pricing
        if "claude-3-opus" in model_lower:
            return {"input_cost": 0.015, "output_cost": 0.075, "category": "premium"}
        elif "claude-3.5-sonnet" in model_lower:
            return {"input_cost": 0.003, "output_cost": 0.015, "category": "premium"}
        elif "claude-3-sonnet" in model_lower:
            return {"input_cost": 0.003, "output_cost": 0.015, "category": "mid"}
        elif "claude-3-haiku" in model_lower or "claude-3.5-haiku" in model_lower:
            return {"input_cost": 0.00025, "output_cost": 0.00125, "category": "budget"}
        
        # Nova pricing
        elif "nova-pro" in model_lower:
            return {"input_cost": 0.0008, "output_cost": 0.0032, "category": "mid"}
        elif "nova-lite" in model_lower:
            return {"input_cost": 0.00006, "output_cost": 0.00024, "category": "budget"}
        elif "nova-micro" in model_lower:
            return {"input_cost": 0.000035, "output_cost": 0.00014, "category": "budget"}
        
        # Titan pricing
        elif "titan-text-premier" in model_lower:
            return {"input_cost": 0.0012, "output_cost": 0.0048, "category": "premium"}
        elif "titan-text-express" in model_lower:
            return {"input_cost": 0.0008, "output_cost": 0.0016, "category": "mid"}
        elif "titan-text-lite" in model_lower:
            return {"input_cost": 0.0003, "output_cost": 0.0004, "category": "budget"}
        
        # Llama pricing
        elif "llama3-2-90b" in model_lower:
            return {"input_cost": 0.002, "output_cost": 0.002, "category": "premium"}
        elif "llama3-1-70b" in model_lower:
            return {"input_cost": 0.00099, "output_cost": 0.00099, "category": "mid"}
        elif "llama3-1-8b" in model_lower or "llama3-2-11b" in model_lower:
            return {"input_cost": 0.00022, "output_cost": 0.00022, "category": "budget"}
        
        # Mistral pricing
        elif "mistral-large" in model_lower:
            return {"input_cost": 0.004, "output_cost": 0.012, "category": "premium"}
        elif "mixtral" in model_lower:
            return {"input_cost": 0.00045, "output_cost": 0.0007, "category": "mid"}
        elif "mistral-7b" in model_lower:
            return {"input_cost": 0.00015, "output_cost": 0.0002, "category": "budget"}
        
        # Cohere pricing
        elif "command-r-plus" in model_lower:
            return {"input_cost": 0.003, "output_cost": 0.015, "category": "premium"}
        elif "command-r" in model_lower:
            return {"input_cost": 0.0005, "output_cost": 0.0015, "category": "mid"}
        
        # AI21 pricing
        elif "jamba-large" in model_lower:
            return {"input_cost": 0.002, "output_cost": 0.008, "category": "mid"}
        elif "jamba-mini" in model_lower:
            return {"input_cost": 0.0002, "output_cost": 0.0004, "category": "budget"}
        
        # Default pricing for unknown models
        elif "premier" in model_name.lower() or "max" in model_name.lower():
            return {"input_cost": 0.001, "output_cost": 0.004, "category": "premium"}
        elif "pro" in model_name.lower() or "express" in model_name.lower():
            return {"input_cost": 0.0008, "output_cost": 0.003, "category": "mid"}
        elif "lite" in model_name.lower() or "mini" in model_name.lower():
            return {"input_cost": 0.0001, "output_cost": 0.0004, "category": "budget"}
        else:
            return {"input_cost": 0.0005, "output_cost": 0.0015, "category": "mid"}
    
    def _estimate_max_tokens(self, model_id: str, model_name: str = "") -> int:
        """Return the documented max *output* tokens for a Bedrock model."""
        model_lower = model_id.lower()
        name_lower = model_name.lower()
        # DeepSeek / Moonshot / Kimi
        if "deepseek" in model_lower or "deepseek" in name_lower:
            return 8192
        if "moonshot" in model_lower or "kimi" in model_lower or "moonshot" in name_lower or "kimi" in name_lower:
            return 16384
        # Claude – order matters: most-specific first
        if "claude-3-7" in model_lower or "claude-3.7" in model_lower:
            return 64000          # Claude 3.7 Sonnet (AWS Bedrock documented max)
        if "claude-3-5" in model_lower or "claude-3.5" in model_lower:
            return 8192           # Claude 3.5 Sonnet / Haiku
        if "claude-3-opus" in model_lower:
            return 4096
        if "claude-3-haiku" in model_lower:
            return 4096
        if "claude" in model_lower:
            return 4096           # generic Claude fallback
        # Amazon Nova (all tiers: Micro / Lite / Pro / Premier)
        if "nova" in model_lower:
            if "premier" in model_lower:
                return 5120
            return 5120
        # Amazon Titan
        if "titan-text-premier" in model_lower:
            return 3072
        if "titan-text-express" in model_lower:
            return 8192
        if "titan-text-lite" in model_lower:
            return 4096
        # Meta Llama
        if "llama3-2-90b" in model_lower or "llama3-2-11b" in model_lower:
            return 8192
        if "llama3-1-70b" in model_lower or "llama3-1-405b" in model_lower:
            return 8192
        if "llama" in model_lower:
            return 4096
        # Mistral
        if "mistral-large" in model_lower:
            return 8192
        if "mixtral" in model_lower or "mistral-7b" in model_lower:
            return 4096
        # Cohere
        if "command-r-plus" in model_lower:
            return 4096
        if "command-r" in model_lower:
            return 4096
        # AI21 Jamba
        if "jamba" in model_lower:
            return 4096
        # Conservative default for anything unrecognised
        return 4096
    
    def _get_description(self, model_id: str, model_name: str) -> str:
        """Generate a description for the model."""
        if "claude" in model_id.lower():
            return "Anthropic's advanced AI assistant with strong reasoning capabilities."
        elif "titan" in model_id.lower():
            return "Amazon's Titan family of large language models."
        elif "nova" in model_id.lower():
            return "Amazon's Nova models optimized for specific tasks."
        elif "llama" in model_id.lower():
            return "Meta's open-source large language model."
        elif "mistral" in model_id.lower():
            return "Mistral AI's efficient language models."
        elif "cohere" in model_id.lower():
            return "Cohere's models optimized for enterprise use cases."
        elif "jamba" in model_id.lower():
            return "AI21's Jamba models with efficient architecture."
        elif "deepseek" in model_id.lower() or "deepseek" in model_name.lower():
            return "DeepSeek's reasoning, coding, and instruction-following model."
        elif "moonshot" in model_id.lower() or "kimi" in model_id.lower() or "kimi" in model_name.lower():
            return "Moonshot AI's Kimi model for reasoning, coding, and long-context work."
        else:
            return "AWS Bedrock foundation model."
    
    def _detect_api_format(self, model_id: str, provider: str = "") -> str:
        """Detect the API format needed for the model."""
        model_lower = model_id.lower()
        provider_lower = provider.lower()
        if "anthropic" in model_lower:
            return "claude"
        elif "deepseek" in model_lower or "deepseek" in provider_lower:
            return "deepseek"
        elif "moonshot" in model_lower or "kimi" in model_lower or "moonshot" in provider_lower:
            return "openai_chat"
        elif "amazon.titan-text" in model_lower:
            return "titan"
        elif "amazon.nova" in model_lower:
            return "nova"
        elif "meta" in model_lower:
            return "llama"
        elif "mistral" in model_lower:
            return "mistral"
        elif "cohere" in model_lower:
            return "cohere"
        elif "ai21" in model_lower:
            return "ai21"
        else:
            return "unknown"
    
    def _sync_global_catalog(self):
        """Keep the module-level catalog in sync after a live AWS refresh."""
        if "MODEL_CATALOG" in globals():
            MODEL_CATALOG.clear()
            MODEL_CATALOG.update(self.catalog)
        if "_rebuild_cost_lookup" in globals():
            _rebuild_cost_lookup()
    
    def _load_fallback_catalog(self):
        """Load a fallback catalog if AWS fetch fails."""
        # (Keep a small subset of models as fallback)
        self.catalog = {
            "claude-3.5-sonnet": {
                "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "display_name": "Claude 3.5 Sonnet",
                "provider": "Anthropic",
                "category": "premium",
                "input_cost_per_1k": 0.003,
                "output_cost_per_1k": 0.015,
                "max_tokens": 8192,
                "description": "Best balance of quality & speed.",
                "api_format": "claude",
            },
            "llama3-2-90b": {
                "model_id": "meta.llama3-2-90b-instruct-v1:0",
                "display_name": "Llama 3.2 90B",
                "provider": "Meta",
                "category": "premium",
                "input_cost_per_1k": 0.002,
                "output_cost_per_1k": 0.002,
                "max_tokens": 4096,
                "description": "Large open-source model.",
                "api_format": "llama",
            },
            "nova-micro": {
                "model_id": "amazon.nova-micro-v1:0",
                "display_name": "Amazon Nova Micro",
                "provider": "Amazon",
                "category": "budget",
                "input_cost_per_1k": 0.000035,
                "output_cost_per_1k": 0.00014,
                "max_tokens": 5120,
                "description": "Cheapest model available.",
                "api_format": "nova",
            },
            "deepseek-v3.2": {
                "model_id": "deepseek.v3.2",
                "display_name": "DeepSeek V3.2",
                "provider": "DeepSeek",
                "category": "mid",
                "input_cost_per_1k": 0.00062,
                "output_cost_per_1k": 0.00185,
                "max_tokens": 8192,
                "description": "DeepSeek's mixture-of-experts model for reasoning, coding, and instruction following.",
                "api_format": "deepseek",
            },
        }
        self._sync_global_catalog()
    
    def get_max_output_tokens(self, model_id: str) -> int:
        """Return the smallest of the global MAX_TOKENS_EXPENSIVE and the model's
        documented output-token ceiling so we never send an illegal value to the API."""
        for info in self.catalog.values():
            if info.get("model_id") == model_id:
                catalog_max = info.get("max_tokens", MAX_TOKENS_EXPENSIVE)
                return min(MAX_TOKENS_EXPENSIVE, catalog_max)
        # Model not in catalog yet – honour the global setting (safe fallback)
        return MAX_TOKENS_EXPENSIVE

    def get_all_models(self) -> dict:
        """Return the complete catalog."""
        if not self.catalog or (self.last_updated and datetime.datetime.now() - datetime.datetime.fromisoformat(self.last_updated) > datetime.timedelta(hours=24)):
            self.refresh()
        return self.catalog
    
    def get_model_by_id(self, model_id: str):
        """Get model info by exact model ID."""
        for key, info in self.catalog.items():
            if info["model_id"] == model_id:
                return info
        return None

# Global instance for backward compatibility
MODEL_CATALOG_INSTANCE = ModelCatalog()

# Build fallback static catalog for initial use (before dynamic loading)
MODEL_CATALOG = {
    "claude-3.5-sonnet-v2": {
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "display_name": "Claude 3.5 Sonnet v2",
        "provider": "Anthropic",
        "category": "premium",
        "input_cost_per_1k": 0.003,
        "output_cost_per_1k": 0.015,
        "max_tokens": 8192,
        "description": "Best balance of quality & speed.",
        "api_format": "claude",
    },
    "claude-3-haiku": {
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "display_name": "Claude 3 Haiku",
        "provider": "Anthropic",
        "category": "budget",
        "input_cost_per_1k": 0.00025,
        "output_cost_per_1k": 0.00125,
        "max_tokens": 4096,
        "description": "Fast and cheap Claude.",
        "api_format": "claude",
    },
    "nova-micro": {
        "model_id": "amazon.nova-micro-v1:0",
        "display_name": "Amazon Nova Micro",
        "provider": "Amazon",
        "category": "budget",
        "input_cost_per_1k": 0.000035,
        "output_cost_per_1k": 0.00014,
        "max_tokens": 5120,
        "description": "Cheapest model available.",
        "api_format": "nova",
    },
    "llama3-2-90b": {
        "model_id": "meta.llama3-2-90b-instruct-v1:0",
        "display_name": "Llama 3.2 90B",
        "provider": "Meta",
        "category": "premium",
        "input_cost_per_1k": 0.002,
        "output_cost_per_1k": 0.002,
        "max_tokens": 4096,
        "description": "Large open-source model.",
        "api_format": "llama",
    },
    "deepseek-v3.2": {
        "model_id": "deepseek.v3.2",
        "display_name": "DeepSeek V3.2",
        "provider": "DeepSeek",
        "category": "mid",
        "input_cost_per_1k": 0.00062,
        "output_cost_per_1k": 0.00185,
        "max_tokens": 8192,
        "description": "DeepSeek's mixture-of-experts model for reasoning, coding, and instruction following.",
        "api_format": "deepseek",
    },
}

# --- Available AWS Regions with Bedrock ---
AVAILABLE_REGIONS = {
    "us-east-1": "US East (N. Virginia) - Most models available ★ RECOMMENDED",
    "us-west-2": "US West (Oregon) - Most models available",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "eu-central-1": "Europe (Frankfurt)",
    "eu-west-1": "Europe (Ireland)",
    "eu-west-2": "Europe (London)",
    "eu-west-3": "Europe (Paris)",
    "ca-central-1": "Canada (Central)",
    "sa-east-1": "South America (São Paulo)",
}

# --- System Prompt ---
SYSTEM_PROMPT = """You are a helpful, knowledgeable, and friendly AI assistant. 
You provide clear, accurate, and well-structured responses. 
When given a conversation summary for context, use it to maintain continuity 
but focus on answering the user's current question thoroughly.
If you're unsure about something, say so honestly.
For complex work, include a concise rationale when useful: key assumptions,
evidence used, decision logic, uncertainty, and suggested checks. Keep this
rationale clear and reviewable without exposing private scratchpad text."""

# --- Cost lookup helper ---
COST_PER_1K = {}


def _rebuild_cost_lookup():
    """Refresh cost lookup after static or dynamic catalog changes."""
    COST_PER_1K.clear()
    for model_info in MODEL_CATALOG.values():
        COST_PER_1K[model_info["model_id"]] = {
            "input": model_info["input_cost_per_1k"],
            "output": model_info["output_cost_per_1k"],
        }


_rebuild_cost_lookup()

# ============================================================================
# FILE MANAGER CLASS - Handle file uploads and storage
# ============================================================================

class FileManager:
    """Manages file uploads, storage, and processing."""
    
    def __init__(self, base_dir: str = "uploaded_files"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load file metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {"files": [], "total_size": 0}
        return {"files": [], "total_size": 0}
    
    def _save_metadata(self):
        """Save file metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def upload_file(self, file_path: str, description: str = "") -> Optional[str]:
        """
        Upload a file from local path to the chat.
        
        Args:
            file_path: Absolute or relative path to the file
            description: Optional description of the file
            
        Returns:
            File ID if successful, None otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"❌ File not found: {file_path}")
                return None
            
            # Check file size (limit to 10MB)
            if path.stat().st_size > 10 * 1024 * 1024:
                print("❌ File too large (max 10MB)")
                return None
            
            # Generate unique ID
            file_id = hashlib.md5(str(path).encode() + str(time.time()).encode()).hexdigest()[:12]
            
            # Copy file to upload directory
            destination = self.base_dir / f"{file_id}_{path.name}"
            
            # Handle different file types
            ext = path.suffix.lower()
            
            if ext in ['.txt', '.py', '.json', '.md', '.csv', '.html', '.xml']:
                # Text files: copy as-is
                import shutil
                shutil.copy2(path, destination)
                content_preview = self._read_text_preview(path)
                
            elif ext in ['.pdf']:
                # PDF: Extract text if possible
                try:
                    import PyPDF2
                    content_preview = self._extract_pdf_text(path)
                    with open(destination.with_suffix('.txt'), 'w') as f:
                        f.write(content_preview)
                except:
                    import shutil
                    shutil.copy2(path, destination)
                    content_preview = f"[PDF file: {path.name}]"
                    
            elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
                # Images: Store and create description
                import shutil
                shutil.copy2(path, destination)
                content_preview = f"[Image file: {path.name}, size: {path.stat().st_size} bytes]"
                
            else:
                # Binary files: store with note
                import shutil
                shutil.copy2(path, destination)
                content_preview = f"[Binary file: {path.name}, size: {path.stat().st_size} bytes]"
            
            # Update metadata
            file_info = {
                "id": file_id,
                "original_name": path.name,
                "stored_name": destination.name,
                "path": str(destination),
                "size": path.stat().st_size,
                "upload_time": datetime.datetime.now().isoformat(),
                "description": description,
                "type": ext[1:] if ext else "unknown",
                "content_preview": content_preview[:500] + "..." if len(content_preview) > 500 else content_preview
            }
            
            self.metadata["files"].append(file_info)
            self.metadata["total_size"] += path.stat().st_size
            self._save_metadata()
            
            print(f"✅ File uploaded: {path.name} (ID: {file_id})")
            return file_id
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None
    
    def _read_text_preview(self, path: Path) -> str:
        """Read first 2000 chars of text file."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(2000)
        except:
            return "[Could not read file content]"
    
    def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF."""
        try:
            import PyPDF2
            text = ""
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages[:5]:  # First 5 pages only
                    text += page.extract_text() + "\n"
            return text[:2000]  # Limit text
        except:
            return "[PDF content extraction failed]"
    
    def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get information about an uploaded file."""
        for file in self.metadata["files"]:
            if file["id"] == file_id:
                return file
        return None
    
    def get_file_content(self, file_id: str, max_chars: int = 10000) -> Optional[str]:
        """Get the content of a file (for sending to AI)."""
        file_info = self.get_file_info(file_id)
        if not file_info:
            return None
        
        path = Path(file_info["path"])
        ext = path.suffix.lower()
        
        try:
            if ext in ['.txt', '.py', '.json', '.md', '.csv']:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(max_chars)
                    
            elif ext == '.pdf':
                return self._extract_pdf_text(path)[:max_chars]
                
            else:
                return f"File: {file_info['original_name']}\nType: {file_info['type']}\nSize: {file_info['size']} bytes"
                
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    def list_files(self) -> List[Dict]:
        """List all uploaded files."""
        return self.metadata.get("files", [])
    
    def delete_file(self, file_id: str) -> bool:
        """Delete an uploaded file."""
        for i, file in enumerate(self.metadata["files"]):
            if file["id"] == file_id:
                # Remove file from disk
                try:
                    Path(file["path"]).unlink(missing_ok=True)
                except:
                    pass
                
                # Remove from metadata
                self.metadata["files"].pop(i)
                self.metadata["total_size"] -= file["size"]
                self._save_metadata()
                
                print(f"🗑️  Deleted file: {file['original_name']}")
                return True
        
        print(f"❌ File not found: {file_id}")
        return False

# ============================================================================
# HELPER CLASSES
# ============================================================================


class TokenTracker:
    """
    Tracks token usage and estimates costs across the session.
    This helps you understand how much your conversations cost.
    """

    def __init__(self):
        # Track tokens separately for each model type
        self.expensive_input_tokens = 0
        self.expensive_output_tokens = 0
        self.cheap_input_tokens = 0
        self.cheap_output_tokens = 0
        self.total_requests = 0
        self.summary_requests = 0

    def add_expensive_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage for the expensive (main) model."""
        self.expensive_input_tokens += input_tokens
        self.expensive_output_tokens += output_tokens
        self.total_requests += 1

    def add_cheap_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage for the cheap (summary) model."""
        self.cheap_input_tokens += input_tokens
        self.cheap_output_tokens += output_tokens
        self.summary_requests += 1

    def estimate_cost(self) -> dict:
        """
        Calculate estimated costs based on token usage.
        Returns a dictionary with cost breakdown.
        """
        # Get cost rates for the configured models (with safe defaults)
        expensive_rates = COST_PER_1K.get(
            EXPENSIVE_MODEL_ID, {"input": 0.003, "output": 0.015}
        )
        cheap_rates = COST_PER_1K.get(
            CHEAP_MODEL_ID, {"input": 0.0001, "output": 0.0004}
        )

        # Calculate costs (tokens / 1000 * cost_per_1k)
        expensive_cost = (
            self.expensive_input_tokens / 1000 * expensive_rates["input"]
            + self.expensive_output_tokens / 1000 * expensive_rates["output"]
        )

        cheap_cost = (
            self.cheap_input_tokens / 1000 * cheap_rates["input"]
            + self.cheap_output_tokens / 1000 * cheap_rates["output"]
        )

        return {
            "expensive_model_cost": expensive_cost,
            "cheap_model_cost": cheap_cost,
            "total_cost": expensive_cost + cheap_cost,
            "total_input_tokens": self.expensive_input_tokens + self.cheap_input_tokens,
            "total_output_tokens": self.expensive_output_tokens
            + self.cheap_output_tokens,
        }

    def get_summary(self) -> str:
        """Return a formatted string summarizing token usage and costs."""
        costs = self.estimate_cost()
        return (
            f"\n{'='*50}\n"
            f"📊 Session Statistics\n"
            f"{'='*50}\n"
            f"Total Requests:          {self.total_requests} (main) + {self.summary_requests} (summaries)\n"
            f"Expensive Model Tokens:  {self.expensive_input_tokens:,} in / {self.expensive_output_tokens:,} out\n"
            f"Cheap Model Tokens:      {self.cheap_input_tokens:,} in / {self.cheap_output_tokens:,} out\n"
            f"{'─'*50}\n"
            f"Estimated Cost:\n"
            f"  Main model:     ${costs['expensive_model_cost']:.6f}\n"
            f"  Summary model:  ${costs['cheap_model_cost']:.6f}\n"
            f"  Total:          ${costs['total_cost']:.6f}\n"
            f"{'='*50}"
        )


class ConversationHistory:
    """
    Manages the full conversation history with persistence.
    Handles saving/loading from disk and organizing messages.
    """

    def __init__(self, filepath: str = HISTORY_FILE):
        self.filepath = filepath
        # Each message is a dict: {"role": "user"|"assistant", "content": str, "timestamp": str}
        self.messages: list[dict] = []
        # Store summaries of older conversations
        self.summaries: list[dict] = []
        # Metadata about the conversation
        self.metadata: dict = {
            "created": None,
            "last_updated": None,
            "total_messages": 0,
            "expensive_model": EXPENSIVE_MODEL_ID,
            "cheap_model": CHEAP_MODEL_ID,
        }

    def _normalize_state(self):
        """Keep loaded/saved history internally consistent."""
        cleaned_messages: list[dict] = []
        for raw in self.messages:
            if not isinstance(raw, dict):
                continue

            role = raw.get("role")
            if role not in ("user", "assistant"):
                continue

            content = str(raw.get("content", ""))
            if role == "assistant":
                content = normalize_answer_text(content)
            if role == "assistant" and not content.strip():
                continue

            cleaned_messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": raw.get("timestamp") or datetime.datetime.now().isoformat(),
                }
            )

        self.messages = cleaned_messages
        if not isinstance(self.summaries, list):
            self.summaries = []
        if not isinstance(self.metadata, dict):
            self.metadata = {}

        if self.messages:
            self.metadata.setdefault("created", self.messages[0].get("timestamp"))
            self.metadata["last_updated"] = self.messages[-1].get("timestamp")
        else:
            self.metadata.setdefault("created", None)
            self.metadata["last_updated"] = None

        self.metadata["total_messages"] = len(self.messages)
        self.metadata.setdefault("expensive_model", EXPENSIVE_MODEL_ID)
        self.metadata.setdefault("cheap_model", CHEAP_MODEL_ID)

    def add_message(self, role: str, content: str):
        """
        Add a new message to the conversation history.

        Args:
            role: Either "user" or "assistant"
            content: The message text
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Unsupported message role: {role}")
        if role == "assistant":
            content = normalize_answer_text(content)
        if role == "assistant" and not str(content).strip():
            raise ValueError("Refusing to save an empty assistant response.")

        timestamp = datetime.datetime.now().isoformat()

        message = {"role": role, "content": content, "timestamp": timestamp}

        self.messages.append(message)
        self.metadata["total_messages"] += 1
        self.metadata["last_updated"] = timestamp

        if self.metadata["created"] is None:
            self.metadata["created"] = timestamp

        # Auto-save after each message to prevent data loss
        self.save()

    def add_summary(self, summary: str, messages_summarized: int):
        """Store a summary of older messages."""
        self.summaries.append(
            {
                "summary": summary,
                "messages_summarized": messages_summarized,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        self.save()

    def get_recent_messages(self, count: int = RECENT_MESSAGES_COUNT) -> list[dict]:
        """Get the most recent N messages."""
        return self.messages[-count:] if len(self.messages) >= count else self.messages[:]

    def get_older_messages(self, recent_count: int = RECENT_MESSAGES_COUNT) -> list[dict]:
        """Get all messages except the most recent N."""
        if len(self.messages) <= recent_count:
            return []
        return self.messages[:-recent_count]

    def get_latest_summary(self) -> Optional[str]:
        """Get the most recent conversation summary, if any."""
        if self.summaries:
            return self.summaries[-1]["summary"]
        return None

    def needs_summarization(self) -> bool:
        """
        Check if the conversation is long enough to benefit from summarization.
        Returns True if we have more messages than our threshold.
        """
        return len(self.messages) > MAX_MESSAGES_BEFORE_SUMMARY

    def _payload(self) -> dict:
        """Return the complete serializable conversation state."""
        self._normalize_state()
        return {
            "metadata": self.metadata,
            "messages": self.messages,
            "summaries": self.summaries,
        }

    def save(self):
        """Save the entire conversation history to a JSON file."""
        data = self._payload()

        try:
            # Write to a temp file first, then rename (atomic write to prevent corruption)
            temp_file = self.filepath + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            # Replace the old file with the new one
            os.replace(temp_file, self.filepath)

        except IOError as e:
            print(f"⚠️  Warning: Could not save history: {e}")

    def create_checkpoint(self, reason: str = "manual", keep: int = 30) -> Optional[str]:
        """Create a timestamped safety checkpoint for important project chats."""
        if not self.messages:
            return None

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in reason.strip())
        safe_reason = safe_reason[:40] or "manual"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = Path(CHECKPOINT_DIR) / f"checkpoint_{timestamp}_{safe_reason}.json"

        try:
            atomic_write_json(checkpoint, self._payload())

            checkpoints = sorted(Path(CHECKPOINT_DIR).glob("checkpoint_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in checkpoints[keep:]:
                try:
                    old.unlink()
                except OSError:
                    pass

            return str(checkpoint)
        except IOError as e:
            print(f"⚠️  Warning: Could not create checkpoint: {e}")
            return None

    def list_checkpoints(self, limit: int = 30) -> list[Path]:
        """Return recent checkpoints, newest first."""
        checkpoint_dir = Path(CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return []
        return sorted(checkpoint_dir.glob("checkpoint_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    def describe_checkpoint(self, checkpoint_path: Path) -> dict:
        """Return human-friendly information about a checkpoint/past chat."""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            metadata = data.get("metadata", {})
            last_updated = metadata.get("last_updated") or metadata.get("created") or ""
            snippet = ""
            for msg in reversed(messages):
                content = str(msg.get("content", "")).replace("\n", " ").strip()
                if content:
                    role = "You" if msg.get("role") == "user" else "Assistant"
                    snippet = f"{role}: {content[:90]}"
                    break
            return {
                "path": checkpoint_path,
                "name": checkpoint_path.name,
                "messages": len(messages),
                "summaries": len(data.get("summaries", [])),
                "last_updated": last_updated,
                "snippet": snippet or "(empty chat)",
            }
        except (IOError, json.JSONDecodeError, KeyError) as e:
            return {
                "path": checkpoint_path,
                "name": checkpoint_path.name,
                "messages": 0,
                "summaries": 0,
                "last_updated": "",
                "snippet": f"(could not read: {e})",
            }

    def list_past_chats(self, limit: int = 30) -> list[dict]:
        """List checkpoint-backed past chats with previews."""
        return [self.describe_checkpoint(path) for path in self.list_checkpoints(limit)]

    def restore_checkpoint(self, checkpoint_path: Path) -> bool:
        """Restore conversation state from a checkpoint file."""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.messages = data.get("messages", [])
            self.summaries = data.get("summaries", [])
            self.metadata = data.get("metadata", self.metadata)
            self._normalize_state()
            self.save()
            return True
        except (IOError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Could not restore checkpoint: {e}")
            return False

    def load(self) -> bool:
        """
        Load conversation history from the JSON file.
        Returns True if history was loaded successfully.
        """
        if not os.path.exists(self.filepath):
            print("📝 No previous conversation found. Starting fresh!")
            return False

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.messages = data.get("messages", [])
            self.summaries = data.get("summaries", [])
            self.metadata = data.get("metadata", self.metadata)
            self._normalize_state()

            msg_count = len(self.messages)
            if msg_count > 0:
                print(f"✅ Loaded {msg_count} messages from previous conversation.")
                last_updated = self.metadata.get("last_updated", "unknown")
                print(f"   Last active: {last_updated}")
                return True
            else:
                print("📝 History file exists but is empty. Starting fresh!")
                return False

        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Warning: History file is corrupted ({e}). Starting fresh!")
            # Back up the corrupted file
            backup_name = self.filepath + ".backup"
            try:
                os.rename(self.filepath, backup_name)
                print(f"   Corrupted file backed up to: {backup_name}")
            except OSError:
                pass
            return False

    def clear(self):
        """Clear all conversation history."""
        self.messages = []
        self.summaries = []
        self.metadata = {
            "created": None,
            "last_updated": None,
            "total_messages": 0,
            "expensive_model": EXPENSIVE_MODEL_ID,
            "cheap_model": CHEAP_MODEL_ID,
        }

        # Delete the file
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

        print("🗑️  Conversation history cleared!")

    def export(self, format_type: str = "markdown") -> Optional[str]:
        """
        Export conversation to a readable file.

        Args:
            format_type: "markdown" or "text"

        Returns:
            The filepath of the exported file, or None if export failed.
        """
        if not self.messages:
            print("Nothing to export - conversation is empty.")
            return None

        # Create export directory if it doesn't exist
        os.makedirs(EXPORT_DIR, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if format_type == "markdown" else "txt"
        filepath = os.path.join(EXPORT_DIR, f"chat_export_{timestamp}.{ext}")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                if format_type == "markdown":
                    f.write("# Chat Conversation Export\n\n")
                    f.write(
                        f"**Exported:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"**Messages:** {len(self.messages)}\n")
                    f.write(f"**Main Model:** {EXPENSIVE_MODEL_ID}\n\n")
                    f.write("---\n\n")

                    for msg in self.messages:
                        role = "🧑 **You**" if msg["role"] == "user" else "🤖 **Assistant**"
                        ts = msg.get("timestamp", "")
                        f.write(f"### {role}\n")
                        f.write(f"*{ts}*\n\n")
                        f.write(f"{msg['content']}\n\n---\n\n")
                else:
                    f.write("Chat Conversation Export\n")
                    f.write(f"{'='*50}\n\n")
                    for msg in self.messages:
                        role = "YOU" if msg["role"] == "user" else "ASSISTANT"
                        f.write(f"[{role}] ({msg.get('timestamp', '')})\n")
                        f.write(f"{msg['content']}\n\n")
                        f.write(f"{'-'*50}\n\n")

            return filepath

        except IOError as e:
            print(f"⚠️  Export failed: {e}")
            return None


# ============================================================================
# SETTINGS MANAGER - Handles runtime configuration changes
# ============================================================================


class SettingsManager:
    """
    Manages all runtime-configurable settings.
    Settings persist across restarts via a JSON file.
    """

    def __init__(self):
        self.settings = {
            "aws_region": AWS_REGION,
            "expensive_model_id": EXPENSIVE_MODEL_ID,
            "cheap_model_id": CHEAP_MODEL_ID,
            "expensive_model_key": "claude-3.5-sonnet-v2",
            "cheap_model_key": "nova-lite",
            "temperature": TEMPERATURE,
            "max_tokens_expensive": MAX_TOKENS_EXPENSIVE,
            "max_tokens_cheap": MAX_TOKENS_CHEAP,
            "recent_messages_count": RECENT_MESSAGES_COUNT,
            "max_messages_before_summary": MAX_MESSAGES_BEFORE_SUMMARY,
            "context_mode": CONTEXT_MODE,
            "stream_responses": STREAM_RESPONSES,
            "show_thinking": SHOW_THINKING,
            "system_prompt": SYSTEM_PROMPT,
        }

    def load(self) -> bool:
        """Load settings from file if it exists."""
        if not os.path.exists(SETTINGS_FILE):
            return False

        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)

            # Merge saved settings with defaults (in case new settings were added)
            for key, value in saved.items():
                if key in self.settings:
                    self.settings[key] = value

            # Apply settings to global variables
            self._apply_globals()
            print(f"✅ Settings loaded from {SETTINGS_FILE}")
            return True

        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Could not load settings: {e}. Using defaults.")
            return False

    def save(self):
        """Save current settings to file."""
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            print(f"✅ Settings saved to {SETTINGS_FILE}")
        except IOError as e:
            print(f"⚠️  Could not save settings: {e}")

    def _apply_globals(self):
        """Apply settings to the global configuration variables."""
        global AWS_REGION, EXPENSIVE_MODEL_ID, CHEAP_MODEL_ID
        global TEMPERATURE, MAX_TOKENS_EXPENSIVE, MAX_TOKENS_CHEAP
        global RECENT_MESSAGES_COUNT, MAX_MESSAGES_BEFORE_SUMMARY, CONTEXT_MODE, STREAM_RESPONSES, SHOW_THINKING, SYSTEM_PROMPT

        AWS_REGION = self.settings["aws_region"]
        EXPENSIVE_MODEL_ID = self.settings["expensive_model_id"]
        CHEAP_MODEL_ID = self.settings["cheap_model_id"]
        TEMPERATURE = self.settings["temperature"]
        MAX_TOKENS_EXPENSIVE = self.settings["max_tokens_expensive"]
        MAX_TOKENS_CHEAP = self.settings["max_tokens_cheap"]
        RECENT_MESSAGES_COUNT = self.settings["recent_messages_count"]
        MAX_MESSAGES_BEFORE_SUMMARY = self.settings["max_messages_before_summary"]
        CONTEXT_MODE = self.settings.get("context_mode", "smart")
        STREAM_RESPONSES = bool(self.settings.get("stream_responses", True))
        SHOW_THINKING = bool(self.settings.get("show_thinking", True))
        SYSTEM_PROMPT = self._with_research_rationale(self.settings["system_prompt"])

    def _with_research_rationale(self, prompt: str) -> str:
        """Keep saved prompts research-friendly without overwriting user settings."""
        rationale_instruction = (
            "For complex tasks, include a concise Rationale section when useful: "
            "assumptions, evidence used, reasoning summary, uncertainty, and next checks. "
            "Keep it reviewable and do not include raw hidden scratchpad text."
        )
        prompt = str(prompt or "").strip()
        if "Research Rationale" in prompt or "research rationale" in prompt.lower():
            return prompt
        return f"{prompt}\n\n{rationale_instruction}".strip()

    def get(self, key: str):
        """Get a setting value."""
        return self.settings.get(key)

    def set(self, key: str, value):
        """Set a setting value and apply globally."""
        self.settings[key] = value
        self._apply_globals()

    def display_model_catalog(self, filter_category: str = None):
        """
        Display all available models in a nice formatted table.

        Args:
            filter_category: None for all, or "budget", "mid", "premium"
        """
        print(f"\n{'═'*80}")
        print("📋 AVAILABLE MODELS CATALOG")
        print(f"{'═'*80}")

        # Group by provider
        providers = {}
        for key, info in MODEL_CATALOG.items():
            provider = info["provider"]
            if provider not in providers:
                providers[provider] = []
            providers[provider].append((key, info))

        # Category emoji mapping
        cat_emoji = {"budget": "💚", "mid": "💛", "premium": "💎"}

        for provider, models in providers.items():
            print(f"\n┌{'─'*78}┐")
            print(f"│ 🏢 {provider:<74}│")
            print(f"├{'─'*78}┤")
            print(
                f"│ {'#':<3} {'Short Key':<20} {'Name':<28} {'Cat':<4} {'Input/1K':<10} {'Output/1K':<10} │"
            )
            print(f"├{'─'*78}┤")

            for idx, (key, info) in enumerate(models, 1):
                if filter_category and info["category"] != filter_category:
                    continue

                cat = cat_emoji.get(info["category"], "  ")
                input_cost = f"${info['input_cost_per_1k']:.6f}"
                output_cost = f"${info['output_cost_per_1k']:.6f}"

                print(
                    f"│ {idx:<3} {key:<20} {info['display_name']:<28} {cat:<4} {input_cost:<10} {output_cost:<10} │"
                )
                # Print description on next line
                print(f"│     └─ {info['description']:<70}│")

            print(f"└{'─'*78}┘")

        print(f"\n{'─'*80}")
        print("Legend: 💚 Budget (<$0.001/1K)  💛 Mid ($0.001-$0.005/1K)  💎 Premium (>$0.005/1K)")
        print(f"{'─'*80}")

    def choose_model(self, purpose: str = "main") -> Optional[tuple]:
        """
        Interactive model selection with fuzzy matching.

        Args:
            purpose: "main" for expensive model, "summary" for cheap model

        Returns:
            Tuple of (short_key, model_id) or None if cancelled
        """
        if purpose == "summary":
            print("\n🔍 Choose a model for SUMMARIZATION (cheaper is better):")
            print("   Recommended: nova-lite, nova-micro, claude-3-haiku, llama3.1-8b")
        else:
            print("\n🔍 Choose a model for MAIN RESPONSES (quality matters):")
            print("   Recommended: claude-3.5-sonnet-v2, claude-3-opus, mistral-large")

        # Show catalog
        self.display_model_catalog()

        print("\n💡 Type the short key (e.g., 'nova-lite', 'claude-3.5-sonnet-v2')")
        print("   Or type part of the name and I'll try to match it.")
        print("   Type 'cancel' to cancel.\n")

        while True:
            user_input = input(f"   Select model for {purpose}: ").strip()

            if not user_input:
                continue

            # Check for cancel
            if user_input.lower() in ("cancel", "c", "quit", "q", "back", "b"):
                print("   Cancelled.")
                return None

            # Try exact match first
            match = self._fuzzy_match_model(user_input)

            if match:
                key, info = match
                print(f"\n   ✅ Selected: {info['display_name']}")
                print(f"      Model ID: {info['model_id']}")
                print(f"      Cost: ${info['input_cost_per_1k']:.6f}/1K input, ${info['output_cost_per_1k']:.6f}/1K output")
                print(f"      Category: {info['category']}")

                confirm = input(f"\n   Confirm this selection? (y/n): ").strip().lower()
                if confirm in ("y", "yes", ""):
                    return (key, info["model_id"])
                else:
                    print("   Let's try again...")
                    continue
            else:
                print(f"   ❌ Could not find a model matching '{user_input}'")
                print(f"   💡 Try one of these: {', '.join(list(MODEL_CATALOG.keys())[:5])}...")

    def _fuzzy_match_model(self, user_input: str) -> Optional[tuple]:
        """
        Try to match user input to a model key using fuzzy matching.
        Handles typos, partial matches, case insensitivity.

        Returns:
            Tuple of (key, model_info) or None
        """
        user_input = user_input.lower().strip()

        # 1. Exact key match (case-insensitive)
        for key, info in MODEL_CATALOG.items():
            if key.lower() == user_input:
                return (key, info)

        # 2. Key contains the input
        matches = []
        for key, info in MODEL_CATALOG.items():
            if user_input in key.lower():
                matches.append((key, info))

        if len(matches) == 1:
            return matches[0]

        # 3. Model ID contains the input
        for key, info in MODEL_CATALOG.items():
            if user_input in info["model_id"].lower():
                return (key, info)

        # 4. Display name contains the input
        for key, info in MODEL_CATALOG.items():
            if user_input in info["display_name"].lower():
                matches.append((key, info))

        if len(matches) == 1:
            return matches[0]

        # 5. Try with common typo fixes
        typo_map = {
            "sonnet": "sonnet",
            "sonet": "sonnet",
            "sonnnet": "sonnet",
            "claude": "claude",
            "cluade": "claude",
            "calud": "claude",
            "calude": "claude",
            "nova": "nova",
            "noav": "nova",
            "llama": "llama",
            "lama": "llama",
            "llamma": "llama",
            "mistral": "mistral",
            "mistrl": "mistral",
            "mistal": "mistral",
            "haiku": "haiku",
            "hiku": "haiku",
            "haiki": "haiku",
            "opus": "opus",
            "opsu": "opus",
            "micro": "micro",
            "mirco": "micro",
            "mciro": "micro",
            "lite": "lite",
            "liet": "lite",
            "large": "large",
            "larg": "large",
            "cohere": "cohere",
            "cohre": "cohere",
            "jamba": "jamba",
            "jmba": "jamba",
        }

        # Apply typo corrections
        corrected = user_input
        for typo, correct in typo_map.items():
            if typo in corrected:
                corrected = corrected.replace(typo, correct)

        if corrected != user_input:
            # Try matching with corrected input
            for key, info in MODEL_CATALOG.items():
                if corrected in key.lower() or corrected in info["display_name"].lower():
                    print(f"   💡 Did you mean '{key}'? (auto-corrected from '{user_input}')")
                    return (key, info)

        # 6. If multiple matches, show them and let user pick
        if len(matches) > 1:
            print(f"   Found {len(matches)} matches:")
            for i, (key, info) in enumerate(matches, 1):
                print(f"     {i}. {key} - {info['display_name']}")

            try:
                pick = input(f"   Pick a number (1-{len(matches)}): ").strip()
                idx = int(pick) - 1
                if 0 <= idx < len(matches):
                    return matches[idx]
            except (ValueError, IndexError):
                pass

        return None

    def interactive_settings_menu(self):
        """
        Full interactive settings configuration menu.
        Lists all options with current values and available choices.
        """
        while True:
            print(f"\n{'═'*60}")
            print("⚙️  SETTINGS CONFIGURATION")
            print(f"{'═'*60}")

            # Get current model display names
            exp_name = "Unknown"
            cheap_name = "Unknown"
            for key, info in MODEL_CATALOG.items():
                if info["model_id"] == self.settings["expensive_model_id"]:
                    exp_name = info["display_name"]
                if info["model_id"] == self.settings["cheap_model_id"]:
                    cheap_name = info["display_name"]

            settings_display = [
                ("1", "Main Model (expensive)", f"{exp_name}", "For main chat responses"),
                ("2", "Summary Model (cheap)", f"{cheap_name}", "For summarizing old context"),
                ("3", "AWS Region", f"{self.settings['aws_region']}", "Where Bedrock runs"),
                ("4", "Temperature", f"{self.settings['temperature']}", "0.0=precise, 1.0=creative"),
                ("5", "Max Tokens (main)", f"{self.settings['max_tokens_expensive']}", "Max response length"),
                ("6", "Max Tokens (summary)", f"{self.settings['max_tokens_cheap']}", "Max summary length"),
                ("7", "Recent Messages Count", f"{self.settings['recent_messages_count']}", "Messages sent in full"),
                ("8", "Summary Threshold", f"{self.settings['max_messages_before_summary']}", "When to start summarizing"),
                ("9", "System Prompt", f"{self.settings['system_prompt'][:50]}...", "AI personality/behavior"),
                ("10", "Context Mode", f"{self.settings.get('context_mode', 'smart')}", "smart summary or full conversation"),
                ("11", "Streaming", "on" if self.settings.get("stream_responses", True) else "off", "Show response while it is generated"),
                ("12", "View Model Catalog", "---", "See all available models & prices"),
                ("13", "Reset to Defaults", "---", "Restore original settings"),
                ("0", "Back to Main Menu", "---", "Save and return"),
            ]

            print(f"\n{'#':<4} {'Setting':<28} {'Current Value':<30} {'Description'}")
            print(f"{'─'*4} {'─'*28} {'─'*30} {'─'*30}")

            for num, name, value, desc in settings_display:
                # Truncate long values
                val_display = value if len(value) <= 28 else value[:25] + "..."
                print(f"{num:<4} {name:<28} {val_display:<30} {desc}")

            print(f"\n{'─'*60}")
            choice = input("Enter setting number to change (0 to go back): ").strip()

            # ─── Handle each setting ─────────────────────────────
            if choice == "0" or choice.lower() in ("back", "b", "quit", "q", "exit"):
                self.save()
                break

            elif choice == "1":
                # Change main model
                result = self.choose_model("main")
                if result:
                    key, model_id = result
                    self.settings["expensive_model_key"] = key
                    self.settings["expensive_model_id"] = model_id
                    self._apply_globals()
                    print(f"   ✅ Main model changed to: {MODEL_CATALOG[key]['display_name']}")

            elif choice == "2":
                # Change summary model
                result = self.choose_model("summary")
                if result:
                    key, model_id = result
                    self.settings["cheap_model_key"] = key
                    self.settings["cheap_model_id"] = model_id
                    self._apply_globals()
                    print(f"   ✅ Summary model changed to: {MODEL_CATALOG[key]['display_name']}")

            elif choice == "3":
                # Change region
                print("\n   Available AWS Regions:")
                region_list = list(AVAILABLE_REGIONS.items())
                for i, (region_code, description) in enumerate(region_list, 1):
                    marker = " ◀ current" if region_code == self.settings["aws_region"] else ""
                    print(f"     {i:2}. {region_code:<20} {description}{marker}")

                pick = input("\n   Enter region number or code: ").strip()

                # Try as number
                try:
                    idx = int(pick) - 1
                    if 0 <= idx < len(region_list):
                        new_region = region_list[idx][0]
                        self.settings["aws_region"] = new_region
                        self._apply_globals()
                        print(f"   ✅ Region changed to: {new_region}")
                        print(f"   ⚠️  Note: You'll need to restart for region change to take effect on the client.")
                    else:
                        print("   ❌ Invalid number.")
                except ValueError:
                    # Try as region code (fuzzy match)
                    pick_lower = pick.lower().replace(" ", "").replace("_", "-")
                    matched_region = None
                    for rc in AVAILABLE_REGIONS:
                        if pick_lower in rc or rc in pick_lower:
                            matched_region = rc
                            break
                    if matched_region:
                        self.settings["aws_region"] = matched_region
                        self._apply_globals()
                        print(f"   ✅ Region changed to: {matched_region}")
                        print(f"   ⚠️  Note: Restart needed for region change.")
                    else:
                        print(f"   ❌ Unknown region: {pick}")

            elif choice == "4":
                # Change temperature
                print(f"\n   Current temperature: {self.settings['temperature']}")
                print("   Range: 0.0 (deterministic) to 1.0 (very creative)")
                print("   Recommended: 0.3 for code, 0.7 for general, 0.9 for creative")

                try:
                    val = input("   Enter new temperature (0.0 - 1.0): ").strip()
                    val = float(val)
                    if 0.0 <= val <= 1.0:
                        self.settings["temperature"] = val
                        self._apply_globals()
                        print(f"   ✅ Temperature set to: {val}")
                    else:
                        print("   ❌ Must be between 0.0 and 1.0")
                except ValueError:
                    print("   ❌ Invalid number. Please enter a decimal like 0.7")

            elif choice == "5":
                # Change max tokens expensive
                print(f"\n   Current max tokens (main): {self.settings['max_tokens_expensive']}")
                print("   Range: 256 - 8192 (higher = longer responses, more cost)")
                print("   Recommended: 2048 for short replies, 4096 for detailed, 8192 for very long")

                try:
                    val = input("   Enter new max tokens: ").strip()
                    val = int(val)
                    if 256 <= val <= 8192:
                        self.settings["max_tokens_expensive"] = val
                        self._apply_globals()
                        print(f"   ✅ Max tokens (main) set to: {val}")
                    else:
                        print("   ❌ Must be between 256 and 8192")
                except ValueError:
                    print("   ❌ Invalid number.")

            elif choice == "6":
                # Change max tokens cheap
                print(f"\n   Current max tokens (summary): {self.settings['max_tokens_cheap']}")
                print("   Range: 128 - 4096")
                print("   Recommended: 512 for brief summaries, 1024 for detailed")

                try:
                    val = input("   Enter new max tokens: ").strip()
                    val = int(val)
                    if 128 <= val <= 4096:
                        self.settings["max_tokens_cheap"] = val
                        self._apply_globals()
                        print(f"   ✅ Max tokens (summary) set to: {val}")
                    else:
                        print("   ❌ Must be between 128 and 4096")
                except ValueError:
                    print("   ❌ Invalid number.")

            elif choice == "7":
                # Change recent messages count
                print(f"\n   Current recent messages count: {self.settings['recent_messages_count']}")
                print("   This is how many recent messages are ALWAYS sent in full to the main model.")
                print("   Range: 2 - 20")
                print("   Recommended: 4-6 (higher = more context but more cost)")

                try:
                    val = input("   Enter new count: ").strip()
                    val = int(val)
                    if 2 <= val <= 20:
                        self.settings["recent_messages_count"] = val
                        self._apply_globals()
                        print(f"   ✅ Recent messages count set to: {val}")
                    else:
                        print("   ❌ Must be between 2 and 20")
                except ValueError:
                    print("   ❌ Invalid number.")

            elif choice == "8":
                # Change summary threshold
                print(f"\n   Current summary threshold: {self.settings['max_messages_before_summary']}")
                print("   After this many messages, older ones get summarized instead of sent in full.")
                print("   Range: 5 - 50")
                print("   Recommended: 8-12 (lower = more summaries = cheaper, but less detail)")

                try:
                    val = input("   Enter new threshold: ").strip()
                    val = int(val)
                    if 5 <= val <= 50:
                        self.settings["max_messages_before_summary"] = val
                        self._apply_globals()
                        print(f"   ✅ Summary threshold set to: {val}")
                    else:
                        print("   ❌ Must be between 5 and 50")
                except ValueError:
                    print("   ❌ Invalid number.")

            elif choice == "9":
                # Change system prompt
                print(f"\n   Current system prompt:")
                print(f"   {self.settings['system_prompt']}")
                print(f"\n   Enter new system prompt (or 'cancel' to keep current):")
                print("   (You can type multiple lines. Enter an empty line to finish)")

                lines = []
                while True:
                    try:
                        line = input("   > ")
                        if line.lower() == "cancel":
                            lines = []
                            break
                        if line == "" and lines:
                            break
                        lines.append(line)
                    except EOFError:
                        break

                if lines:
                    new_prompt = "\n".join(lines)
                    self.settings["system_prompt"] = new_prompt
                    self._apply_globals()
                    print(f"   ✅ System prompt updated!")
                else:
                    print("   Cancelled - keeping current prompt.")

            elif choice == "10":
                # Change context mode
                print("\n   Context modes:")
                print("   1. smart - summarize older messages with the cheap model")
                print("   2. full  - send the full conversation, no summaries")
                mode_choice = input("   Choose mode: ").strip().lower()
                if mode_choice in ("1", "smart", "summary"):
                    self.settings["context_mode"] = "smart"
                    self._apply_globals()
                    print("   Context mode set to smart summaries.")
                elif mode_choice in ("2", "full", "pure"):
                    self.settings["context_mode"] = "full"
                    self._apply_globals()
                    print("   Context mode set to full pure conversation.")
                else:
                    print("   Invalid context mode.")

            elif choice == "11":
                # Toggle streaming
                self.settings["stream_responses"] = not self.settings.get("stream_responses", True)
                self._apply_globals()
                state = "on" if self.settings["stream_responses"] else "off"
                print(f"   Streaming is now {state}.")

            elif choice == "12":
                # View model catalog
                print("\n   Filter by category?")
                print("   1. All models")
                print("   2. Budget only")
                print("   3. Mid-tier only")
                print("   4. Premium only")

                filter_choice = input("   Choice (default: all): ").strip()
                filter_map = {"2": "budget", "3": "mid", "4": "premium"}
                self.display_model_catalog(filter_map.get(filter_choice))

            elif choice == "13":
                # Reset to defaults
                confirm = input("   Reset ALL settings to defaults? (yes/no): ").strip().lower()
                if confirm in ("yes", "y"):
                    self.settings = {
                        "aws_region": "ap-south-1",
                        "expensive_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "cheap_model_id": "amazon.nova-lite-v1:0",
                        "expensive_model_key": "claude-3.5-sonnet-v2",
                        "cheap_model_key": "nova-lite",
                        "temperature": 0.7,
                        "max_tokens_expensive": 4096,
                        "max_tokens_cheap": 1024,
                        "recent_messages_count": 5,
                        "max_messages_before_summary": 10,
                        "context_mode": "smart",
                        "stream_responses": True,
                        "system_prompt": SYSTEM_PROMPT,
                    }
                    self._apply_globals()
                    self.save()
                    print("   All settings reset to defaults!")
                else:
                    print("   Cancelled.")

            else:
                print("   ❌ Invalid choice. Enter a number from the list above.")


# ============================================================================
# CHAT COMMANDS PROCESSOR
# ============================================================================

class ChatCommands:
    """Handle special chat commands like /file, /summarize, etc."""
    
    def __init__(self, chat_instance):
        self.chat = chat_instance
        self.commands = {
            "/help": self.cmd_help,
            "/file": self.cmd_file,
            "/clear": self.cmd_clear,
            "/stats": self.cmd_stats,
            "/cost": self.cmd_cost,
            "/export": self.cmd_export,
            "/recent": self.cmd_recent,
            "/summary": self.cmd_summary,
            "/search": self.cmd_search,
            "/models": self.cmd_models,
            "/model": self.cmd_model,
            "/settings": self.cmd_settings,
            "/context": self.cmd_context,
            "/stream": self.cmd_stream,
            "/think": self.cmd_think,      # toggle thinking/reasoning visibility
            "/checkpoint": self.cmd_checkpoint,
            "/recover": self.cmd_recover,
            "/chats": self.cmd_chats,
            "/continue": self.cmd_chats,
            "/menu": self.cmd_menu,
            "/multi": self.cmd_multi_hint,
            "/paste": self.cmd_multi_hint,  # alias
            "/p": self.cmd_multi_hint,      # short alias
            "/quit": self.cmd_exit,
            "/exit": self.cmd_exit,
        }
    
    def process(self, user_input: str) -> Optional[str]:
        """Process chat commands, returns None if not a command."""
        if not user_input.startswith("/"):
            return None
        
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            return f"❌ Unknown command: {cmd}. Type /help for available commands."
    
    def cmd_help(self, args: str) -> str:
        """Show help for commands."""
        return """
Available commands:

Chat:
  /p  or  /paste      Paste or type a long message (END to send)
  /recent [n]         Show recent messages
  /summary            Summarize the conversation
  /search <text>      Search conversation history
  /chats              List past chats you can continue
  /chats continue N   Restore and continue past chat N
  /chats save <name>  Save current chat as a named checkpoint
  /clear              Clear conversation history
  /context [smart|full] Show or change context mode
  /stream [on|off]    Show or change streaming
  /think [on|off]     Show or hide model reasoning/thinking blocks
  /quit               Exit safely

Project safety:
  /checkpoint [note]  Save a manual checkpoint
  /recover list       Show recent checkpoints
  /recover restore N  Restore checkpoint N from the list
  /export [md|txt]    Export conversation

Files:
  /file upload <path> [description]
  /file list
  /file show <id>
  /file delete <id>

Models/settings:
  /models [budget|mid|premium]
  /model main         Change main response model
  /model summary      Change summary model
  /settings           Open settings
  /cost or /stats     Show usage
        """
    
    def cmd_file(self, args: str) -> str:
        """Handle file operations."""
        subcmd = args.split(maxsplit=1)[0] if args else ""
        
        if subcmd == "upload" and len(args.split()) > 1:
            remainder = args.split(maxsplit=1)[1].strip()
            file_path, description = self._split_file_upload_args(remainder)
            file_id = self.chat.upload_file(file_path, description)
            return f"✅ File uploaded and added to context. ID: {file_id}" if file_id else "❌ File upload failed."

        if subcmd == "list":
            files = self.chat.file_manager.list_files()
            if not files:
                return "No files uploaded."
            
            result = "Uploaded files:\n"
            for f in files:
                result += f"  {f['id']}: {f['original_name']} ({f['type']}, {f['size']} bytes)\n"
            return result
            
        elif subcmd == "show" and len(args.split()) > 1:
            file_id = args.split(maxsplit=1)[1]
            content = self.chat.file_manager.get_file_content(file_id, max_chars=2000)
            if content:
                return f"📄 **File Content:**\n```\n{content[:2000]}\n```"
            else:
                return f"❌ File not found: {file_id}"
                
        elif subcmd == "delete" and len(args.split()) > 1:
            file_id = args.split(maxsplit=1)[1]
            success = self.chat.file_manager.delete_file(file_id)
            return "✅ File deleted." if success else "❌ File not found."
            
        else:
            return "Usage: /file upload <path> [description] | /file list | /file show <id> | /file delete <id>"

    def _split_file_upload_args(self, text: str) -> tuple[str, str]:
        """Parse /file upload arguments, allowing quoted file paths."""
        text = text.strip()
        if not text:
            return "", ""
        if text[0] in ('"', "'"):
            quote = text[0]
            end = text.find(quote, 1)
            if end != -1:
                return text[1:end], text[end + 1:].strip()
        parts = text.split(maxsplit=1)
        return parts[0], parts[1] if len(parts) > 1 else ""
    
    def cmd_clear(self, args: str) -> str:
        """Clear history."""
        self.chat.history.clear()
        return "🗑️  Conversation cleared!"
    
    def cmd_stats(self, args: str) -> str:
        """Show conversation stats."""
        if not self.chat.history.messages:
            return "No messages yet."
        
        user_msgs = sum(1 for m in self.chat.history.messages if m["role"] == "user")
        asst_msgs = sum(1 for m in self.chat.history.messages if m["role"] == "assistant")
        
        return f"""
📊 **Conversation Stats:**
  Total messages: {len(self.chat.history.messages)}
  Your messages: {user_msgs}
  AI responses: {asst_msgs}
  Files uploaded: {len(self.chat.file_manager.list_files())}
        """
    
    def cmd_cost(self, args: str) -> str:
        """Show cost information."""
        return self.chat.tracker.get_summary()
    
    def cmd_export(self, args: str) -> str:
        """Export conversation."""
        fmt = "markdown" if not args or "mark" in args.lower() else "text"
        filepath = self.chat.history.export(fmt)
        if filepath:
            return f"✅ Exported to: {filepath}"
        else:
            return "❌ Export failed."

    def cmd_recent(self, args: str) -> str:
        """Show recent messages."""
        try:
            count = int(args.strip()) if args.strip() else 8
        except ValueError:
            count = 8
        self.chat.view_recent(max(1, min(count, 50)))
        return ""

    def cmd_summary(self, args: str) -> str:
        """Show conversation summary."""
        self.chat.view_full_summary()
        return ""

    def cmd_search(self, args: str) -> str:
        """Search history."""
        if not args.strip():
            return "Usage: /search <text>"
        self.chat.search_history(args.strip())
        return ""

    def cmd_models(self, args: str) -> str:
        """Show model catalog."""
        category = args.strip().lower() or None
        if category not in (None, "budget", "mid", "premium"):
            return "Usage: /models [budget|mid|premium]"
        self.chat.settings_manager.display_model_catalog(category)
        return ""

    def cmd_model(self, args: str) -> str:
        """Change main or summary model."""
        target = args.strip().lower()
        if target not in ("main", "summary"):
            return "Usage: /model main OR /model summary"

        result = self.chat.settings_manager.choose_model(target)
        if not result:
            return "Model change cancelled."

        key, model_id = result
        if target == "main":
            self.chat.settings_manager.settings["expensive_model_key"] = key
            self.chat.settings_manager.settings["expensive_model_id"] = model_id
        else:
            self.chat.settings_manager.settings["cheap_model_key"] = key
            self.chat.settings_manager.settings["cheap_model_id"] = model_id

        self.chat.settings_manager._apply_globals()
        self.chat.settings_manager.save()
        return f"✅ {target.title()} model changed to {MODEL_CATALOG[key]['display_name']}"

    def cmd_settings(self, args: str) -> str:
        """Open settings menu."""
        self.chat.settings_manager.interactive_settings_menu()
        self.chat.reinitialize_client()
        return "Settings updated."

    def cmd_context(self, args: str) -> str:
        """Show or change context mode."""
        mode = args.strip().lower()
        if not mode:
            return f"Context mode: {CONTEXT_MODE}. Use /context smart or /context full."
        if mode in ("smart", "summary", "summaries"):
            self.chat.settings_manager.settings["context_mode"] = "smart"
            self.chat.settings_manager._apply_globals()
            self.chat.settings_manager.save()
            return "Context mode set to smart summaries."
        if mode in ("full", "pure", "conversation"):
            self.chat.settings_manager.settings["context_mode"] = "full"
            self.chat.settings_manager._apply_globals()
            self.chat.settings_manager.save()
            return "Context mode set to full pure conversation. No summary model will be used for chat context."
        return "Usage: /context smart OR /context full"

    def cmd_stream(self, args: str) -> str:
        """Show or change streaming mode."""
        value = args.strip().lower()
        if not value:
            return f"Streaming: {'on' if STREAM_RESPONSES else 'off'}. Use /stream on or /stream off."
        if value in ("on", "true", "yes", "1"):
            self.chat.settings_manager.settings["stream_responses"] = True
        elif value in ("off", "false", "no", "0"):
            self.chat.settings_manager.settings["stream_responses"] = False
        else:
            return "Usage: /stream on OR /stream off"
        self.chat.settings_manager._apply_globals()
        self.chat.settings_manager.save()
        return f"Streaming is now {'on' if STREAM_RESPONSES else 'off'}."

    def cmd_think(self, args: str) -> str:
        """Toggle visibility of model reasoning/thinking blocks."""
        global SHOW_THINKING
        value = args.strip().lower()
        if not value:
            state = "ON (reasoning visible)" if SHOW_THINKING else "OFF (reasoning hidden)"
            return (
                f"Thinking display: {state}\n"
                "Use /think on  → show <thinking>/<reasoning> blocks in amber\n"
                "Use /think off → hide them (only final answer shown)"
            )
        if value in ("on", "true", "yes", "1", "show"):
            SHOW_THINKING = True
            self.chat.settings_manager.settings["show_thinking"] = True
        elif value in ("off", "false", "no", "0", "hide"):
            SHOW_THINKING = False
            self.chat.settings_manager.settings["show_thinking"] = False
        else:
            return "Usage: /think on  OR  /think off"
        self.chat.settings_manager._apply_globals()
        self.chat.settings_manager.save()
        return f"Thinking display is now {'ON — reasoning blocks will appear in amber' if SHOW_THINKING else 'OFF — only the final answer is shown'}."

    def cmd_checkpoint(self, args: str) -> str:
        """Create a manual checkpoint."""
        checkpoint = self.chat.history.create_checkpoint(args.strip() or "manual")
        if checkpoint:
            self.chat._last_checkpoint_msg_count = len(self.chat.history.messages)
            return f"✅ Checkpoint saved: {checkpoint}"
        return "No messages to checkpoint yet."

    def _format_past_chats(self, chats: list[dict]) -> str:
        """Format checkpoint-backed past chats for display."""
        if not chats:
            return "No past chats found yet. Checkpoints are created on exit, errors, and every few messages."

        result = "Past chats you can continue:\n"
        for i, chat in enumerate(chats, 1):
            updated = chat["last_updated"][:19].replace("T", " ") if chat["last_updated"] else "unknown time"
            result += f"  {i}. {updated} | {chat['messages']} messages | {chat['name']}\n"
            result += f"     {chat['snippet']}\n"
        result += "\nUse: /chats continue N"
        return result

    def _restore_past_chat_by_index(self, index_text: str) -> str:
        """Restore a checkpoint by visible list number."""
        chats = self.chat.history.list_past_chats()
        if not index_text.isdigit():
            return "Usage: /chats continue N"
        index = int(index_text) - 1
        if not 0 <= index < len(chats):
            return "Invalid chat number. Use /chats first."

        if self.chat.history.messages:
            self.chat.history.create_checkpoint("before_switching_chat")

        target = chats[index]["path"]
        if self.chat.history.restore_checkpoint(target):
            self.chat._last_checkpoint_msg_count = len(self.chat.history.messages)
            return f"Restored past chat: {chats[index]['name']}\nYou can continue typing now."
        return "Could not restore that past chat."

    def cmd_chats(self, args: str) -> str:
        """List, save, or continue previous chats."""
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if action in ("", "list", "show"):
            return self._format_past_chats(self.chat.history.list_past_chats())

        if action in ("continue", "load", "restore", "open"):
            return self._restore_past_chat_by_index(rest.strip())

        if action == "save":
            checkpoint = self.chat.history.create_checkpoint(rest.strip() or "saved_chat")
            if checkpoint:
                self.chat._last_checkpoint_msg_count = len(self.chat.history.messages)
                return f"Saved current chat: {checkpoint}"
            return "No messages to save yet."

        return "Usage: /chats | /chats continue N | /chats save <name>"

    def cmd_recover(self, args: str) -> str:
        """Backward-compatible recovery alias for /chats."""
        parts = args.split(maxsplit=1)
        if not parts or parts[0].lower() == "list":
            return self.cmd_chats("list")
        if parts[0].lower() in ("restore", "continue", "load", "open"):
            number = parts[1] if len(parts) > 1 else ""
            return self.cmd_chats(f"continue {number}")
        return "Usage: /recover list OR /recover restore N"

    def cmd_menu(self, args: str) -> str:
        """Show the old menu as a reference."""
        print_menu(self.chat)
        return ""

    def cmd_multi_hint(self, args: str) -> str:
        """Handled by the chat loop before command processing."""
        return "Type /multi at the prompt to open multi-line input."

    def cmd_exit(self, args: str) -> str:
        """Exit the chat loop."""
        checkpoint = self.chat.history.create_checkpoint("exit")
        if checkpoint:
            return f"__EXIT__::{checkpoint}"
        return "__EXIT__"

# ============================================================================
# PROJECT SESSION MANAGER
# ============================================================================

class ProjectSession:
    """Save and restore complete chat sessions."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.sessions_dir = Path("project_sessions")
        self.sessions_dir.mkdir(exist_ok=True)
    
    def save_session(self, chat_instance, notes: str = ""):
        """Save current chat state as a project session."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = self.sessions_dir / f"{self.project_name}_{timestamp}.json"
        
        session_data = {
            "project": self.project_name,
            "timestamp": timestamp,
            "notes": notes,
            "chat_history": chat_instance.history.messages[-100:],  # Last 100 messages
            "settings": {
                "expensive_model": EXPENSIVE_MODEL_ID,
                "cheap_model": CHEAP_MODEL_ID,
                "temperature": TEMPERATURE,
                "region": AWS_REGION,
            },
            "files": [f["id"] for f in chat_instance.file_manager.list_files()],
            "token_usage": {
                "expensive_input": chat_instance.tracker.expensive_input_tokens,
                "expensive_output": chat_instance.tracker.expensive_output_tokens,
                "cheap_input": chat_instance.tracker.cheap_input_tokens,
                "cheap_output": chat_instance.tracker.cheap_output_tokens,
            }
        }
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"✅ Session saved: {session_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save session: {e}")
            return False
    
    def list_sessions(self):
        """List all saved sessions for this project."""
        sessions = list(self.sessions_dir.glob(f"{self.project_name}_*.json"))
        if not sessions:
            return []
        
        sessions_info = []
        for session_file in sorted(sessions, reverse=True):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                sessions_info.append({
                    "file": session_file.name,
                    "timestamp": data.get("timestamp", ""),
                    "message_count": len(data.get("chat_history", [])),
                    "notes": data.get("notes", "")[:100]
                })
            except:
                continue
        
        return sessions_info

# ============================================================================
# MAIN CHAT CLASS
# ============================================================================


class BedrockChat:
    """
    Main chat class that orchestrates the conversation with AWS Bedrock.
    Handles model invocation, context management, and cost optimization.
    """

    def __init__(self):
        # Initialize the conversation history manager
        self.history = ConversationHistory()

        # Initialize the token/cost tracker
        self.tracker = TokenTracker()

        # Initialize the file manager for uploads
        self.file_manager = FileManager()

        # Initialize the chat commands processor
        self.commands = ChatCommands(self)

        # Initialize project session manager (optional)
        self.project_session = None

        # Initialize the AWS Bedrock client
        self.client = None

        # Track if we've already created a summary/checkpoint for the current batch
        self._last_summary_msg_count = 0
        self._last_checkpoint_msg_count = 0
        self._last_response_was_streamed = False
        
        # Track last response metadata for display
        self._last_response_tokens = 0
        self._last_response_cost = 0.0
        self._last_response_model = EXPENSIVE_MODEL_ID

    def initialize(self) -> bool:
        """
        Set up the AWS Bedrock client, load settings, and load any existing history.
        Returns True if initialization was successful.
        """
        print("\n🚀 Initializing Bedrock Chat...")

        # Step 0: Load saved settings
        self.settings_manager = SettingsManager()
        self.settings_manager.load()

        print(f"   Region:          {AWS_REGION}")
        print(f"   Main Model:      {EXPENSIVE_MODEL_ID}")
        print(f"   Summary Model:   {CHEAP_MODEL_ID}")
        print(f"   History File:    {HISTORY_FILE}")
        print()

        # Step 1: Get credentials interactively
        self.client = None
        auth_success = self._get_credentials_interactive()
        if not auth_success:
            return False

        # Step 2: Refresh the model catalog now that credentials are available.
        try:
            MODEL_CATALOG_INSTANCE.refresh(AWS_REGION)
        except Exception as e:
            print(f"⚠️  Model catalog refresh skipped: {e}")

        # Step 3: Load existing conversation history
        self.history.load()
        self._last_checkpoint_msg_count = len(self.history.messages)

        # Step 4: Recover any answer that was interrupted by shutdown/network failure.
        self._recover_interrupted_request()

        print("\n✅ Initialization complete! Ready to chat.\n")
        return True

    def _request_state_path(self) -> Path:
        """Path to the in-progress request journal."""
        return Path(REQUEST_STATE_FILE)

    def _save_request_state(self, state: dict):
        """Persist in-progress request state for reboot/crash recovery."""
        state["updated_at"] = datetime.datetime.now().isoformat()
        try:
            atomic_write_json(self._request_state_path(), state)
        except IOError as e:
            print(f"⚠️  Could not save request recovery state: {e}")

    def _load_request_state(self) -> Optional[dict]:
        """Load in-progress request state if present."""
        path = self._request_state_path()
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError):
            backup = path.with_suffix(path.suffix + ".corrupt")
            try:
                os.replace(path, backup)
            except OSError:
                pass
            return None

    def _clear_request_state(self):
        """Remove completed in-progress request state."""
        try:
            self._request_state_path().unlink(missing_ok=True)
        except OSError:
            pass

    def _recover_interrupted_request(self):
        """Offer recovery when the prior run died mid-answer."""
        state = self._load_request_state()
        if not state:
            return

        user_message = state.get("user_message", "")
        partial = state.get("partial_response", "")
        updated_at = state.get("updated_at", "unknown time")

        print("\n" + "=" * 72)
        print("Interrupted Work Found")
        print("=" * 72)
        print(f"Last update: {updated_at}")
        print(f"User message: {user_message[:180]}")
        if partial:
            print(f"Partial assistant answer saved ({len(partial)} chars).")
            print(partial[:600] + ("..." if len(partial) > 600 else ""))
        else:
            print("No assistant text was saved yet, but your user message was preserved.")

        print("\n1. Continue/retry this request now")
        print("2. Save partial answer and continue later")
        print("3. Discard recovery state")
        choice = safe_input("Recovery choice (default 2): ").strip() or "2"

        if choice == "1":
            self._clear_request_state()
            # Remove the already-saved trailing user message so retry does not duplicate it.
            if self.history.messages and self.history.messages[-1].get("role") == "user" and self.history.messages[-1].get("content") == user_message:
                self.history.messages.pop()
                self.history.metadata["total_messages"] = max(0, self.history.metadata.get("total_messages", 1) - 1)
                self.history.save()
            print("Retrying interrupted request...")
            self.send_message(user_message, stream=STREAM_RESPONSES)
            return

        if choice == "3":
            self._clear_request_state()
            print("Recovery state discarded. Conversation history remains saved.")
            return

        if partial:
            self.history.add_message("assistant", "[Recovered partial answer after interruption]\n" + partial)
            self.history.create_checkpoint("recovered_partial")
            self._clear_request_state()
            print("Partial answer saved into the conversation. You can ask the model to continue from it.")
        else:
            print("Recovery state kept so you can retry it later.")

    def _get_credentials_interactive(self):
        """Interactive credential collection with persistence."""
        import os
        import json
        
        CREDS_FILE = str(SCRIPT_DIR / "aws_credentials.json")
        LEGACY_CREDS_FILE = "aws_credentials.json"

        def normalize_credentials(creds: dict) -> dict:
            """Handle older/partial credential files without crashing startup."""
            if not isinstance(creds, dict):
                creds = {}
            if not creds.get("region"):
                creds["region"] = creds.get("aws_region") or AWS_REGION
            return creds
        
        # Try to load saved credentials first
        saved_creds = {}
        creds_load_path = CREDS_FILE
        if not os.path.exists(creds_load_path) and os.path.exists(LEGACY_CREDS_FILE):
            creds_load_path = LEGACY_CREDS_FILE

        if os.path.exists(creds_load_path):
            try:
                with open(creds_load_path, 'r') as f:
                    saved_creds = normalize_credentials(json.load(f))
                print(f"📁 Loaded saved credentials for region: {saved_creds.get('region', 'not set')}")
            except:
                saved_creds = {}
        else:
            saved_creds = normalize_credentials(saved_creds)
        
        while True:
            print(f"\n{'═'*60}")
            print("🔐 AWS CREDENTIALS SETUP")
            print(f"{'═'*60}")
            
            # Show current values
            access_key = saved_creds.get('aws_access_key_id', '')
            secret_key = saved_creds.get('aws_secret_access_key', '')
            region = saved_creds.get('region', AWS_REGION)
            
            print(f"1. AWS Access Key ID: {access_key if access_key else '[NOT SET]'}")
            print(f"2. AWS Secret Access Key: {'*' * len(secret_key) if secret_key else '[NOT SET]'}")
            print(f"3. AWS Region: {region}")
            print(f"4. Test Connection")
            print(f"5. Clear Saved Credentials")
            print(f"0. Continue (use current)")
            print(f"{'═'*60}")
            
            choice = input("\nSelect option (0-5): ").strip()
            
            if choice == "0":
                # Use saved credentials if they exist
                saved_creds = normalize_credentials(saved_creds)
                if saved_creds.get('aws_access_key_id') and saved_creds.get('aws_secret_access_key'):
                    os.environ['AWS_ACCESS_KEY_ID'] = saved_creds['aws_access_key_id']
                    os.environ['AWS_SECRET_ACCESS_KEY'] = saved_creds['aws_secret_access_key']
                    os.environ['AWS_DEFAULT_REGION'] = saved_creds.get('region', AWS_REGION)
                break
            
            elif choice == "1":
                new_key = input("Enter AWS Access Key ID: ").strip()
                if new_key:
                    saved_creds['aws_access_key_id'] = new_key
            
            elif choice == "2":
                new_secret = input("Enter AWS Secret Access Key: ").strip()
                if new_secret:
                    saved_creds['aws_secret_access_key'] = new_secret
            
            elif choice == "3":
                print("\nAvailable Regions:")
                region_list = list(AVAILABLE_REGIONS.items())
                for i, (rc, desc) in enumerate(region_list[:10], 1):
                    print(f"   {i}. {rc:<15} - {desc}")
                print("   ... (see all in settings)")
                
                new_region = input(f"Enter region (current: {region}): ").strip()
                if new_region:
                    saved_creds['region'] = new_region
            
            elif choice == "4":
                if not saved_creds.get('aws_access_key_id') or not saved_creds.get('aws_secret_access_key'):
                    print("❌ Please set Access Key ID and Secret Key first")
                    continue
                
                # Test with current credentials
                saved_creds = normalize_credentials(saved_creds)
                os.environ['AWS_ACCESS_KEY_ID'] = saved_creds['aws_access_key_id']
                os.environ['AWS_SECRET_ACCESS_KEY'] = saved_creds['aws_secret_access_key']
                os.environ['AWS_DEFAULT_REGION'] = saved_creds.get('region', AWS_REGION)
                
                try:
                    # Test with STS
                    test_session = boto3.Session(
                        aws_access_key_id=saved_creds['aws_access_key_id'],
                        aws_secret_access_key=saved_creds['aws_secret_access_key'],
                        region_name=saved_creds.get('region', AWS_REGION)
                    )
                    sts = test_session.client('sts')
                    identity = sts.get_caller_identity()
                    print(f"✅ AWS Identity: {identity['UserId']}")
                    
                    # Test Bedrock access
                    bedrock = test_session.client('bedrock')
                    models = bedrock.list_foundation_models()
                    print(f"✅ Bedrock Access: {len(models['modelSummaries'])} models available")
                    
                    # Save credentials since test succeeded
                    with open(CREDS_FILE, 'w') as f:
                        json.dump(saved_creds, f, indent=2)
                    print(f"💾 Credentials saved to: {CREDS_FILE}")
                    
                except Exception as e:
                    print(f"❌ Connection failed: {e}")
                    print("   Check your credentials and region.")
            
            elif choice == "5":
                removed_any = False
                for path in {CREDS_FILE, LEGACY_CREDS_FILE}:
                    if os.path.exists(path):
                        os.remove(path)
                        removed_any = True
                if removed_any:
                    saved_creds = {}
                    print("🗑️  Saved credentials cleared")
                else:
                    print("No saved credentials found")
            
            else:
                print("Invalid choice")
        
        # Create client with final credentials
        try:
            saved_creds = normalize_credentials(saved_creds)
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=saved_creds.get('region', AWS_REGION),
                aws_access_key_id=saved_creds.get('aws_access_key_id'),
                aws_secret_access_key=saved_creds.get('aws_secret_access_key'),
                config=bedrock_client_config(),
            )
            print("✅ AWS Bedrock client created successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to create client: {e}")
            return False

    def reinitialize_client(self):
        """Reinitialize the Bedrock client (needed after region change)."""
        try:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=AWS_REGION,
                config=bedrock_client_config(),
            )
            print(f"✅ Client reconnected to region: {AWS_REGION}")
            return True
        except Exception as e:
            print(f"❌ Failed to reconnect: {e}")
            return False

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return True for transient network/service errors."""
        if isinstance(error, (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError, ConnectionClosedError)):
            return True
        if isinstance(error, ClientError):
            code = error.response.get("Error", {}).get("Code", "")
            retryable_codes = {
                "ThrottlingException",
                "TooManyRequestsException",
                "RequestTimeout",
                "RequestTimeoutException",
                "ServiceUnavailableException",
                "InternalServerException",
                "ModelTimeoutException",
                "ModelNotReadyException",
                "ModelStreamErrorException",
                "NetworkingError",
            }
            return code in retryable_codes or "timeout" in code.lower() or "throttl" in code.lower()
        return isinstance(error, BotoCoreError)

    def _call_with_retries(self, label: str, func, *args, **kwargs):
        """App-level retry wrapper for flaky internet and transient Bedrock errors."""
        last_error = None
        for attempt in range(1, APP_RETRY_ATTEMPTS + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if not self._is_retryable_error(e) or attempt >= APP_RETRY_ATTEMPTS:
                    raise
                delay = min(45, APP_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                print(f"\n⚠️  {label} failed temporarily ({e}). Retry {attempt}/{APP_RETRY_ATTEMPTS - 1} in {delay}s...")
                time.sleep(delay)
                try:
                    self.reinitialize_client()
                except Exception:
                    pass
        raise last_error

    def _invoke_model_with_retries(self, **kwargs):
        """Call invoke_model with network resilience."""
        return self._call_with_retries("Bedrock invoke", self.client.invoke_model, **kwargs)

    def _invoke_stream_with_retries(self, **kwargs):
        """Start invoke_model_with_response_stream with network resilience."""
        return self._call_with_retries("Bedrock stream", self.client.invoke_model_with_response_stream, **kwargs)

    def _get_model_info_by_id(self, model_id: str) -> dict:
        """Resolve model info, inferring format for dynamic models missing from local catalog."""
        for key, info in MODEL_CATALOG.items():
            if info["model_id"] == model_id:
                return info
        return {
            "model_id": model_id,
            "display_name": model_id,
            "provider": "Unknown",
            "api_format": MODEL_CATALOG_INSTANCE._detect_api_format(model_id),
            "category": "mid",
            "input_cost_per_1k": 0.0005,
            "output_cost_per_1k": 0.0015,
            "max_tokens": 4096,
            "description": "Dynamically selected Bedrock model.",
        }

    def _build_openai_chat_messages(self, messages: list[dict], system: str = "") -> list[dict]:
        """Build OpenAI-compatible chat messages used by Moonshot/Kimi on Bedrock."""
        converted = []
        research_rationale_guard = (
            "For complex tasks, include a visible 'Reasoning Summary' section when useful. "
            "Make it concise and audit-friendly: assumptions, evidence used, reasoning summary, "
            "uncertainty, and next checks. Do not output raw hidden chain-of-thought, scratchpad "
            "text, or <reasoning>/<think> tags. For long answers, use short headings, bullets, "
            "numbered steps, and fenced code blocks when useful."
        )
        system = f"{system.strip()}\n\n{research_rationale_guard}" if system else research_rationale_guard
        if system:
            converted.append({"role": "system", "content": system})
        for msg in messages:
            role = msg.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content)
            converted.append({"role": role, "content": str(content)})
        return converted

    def _build_openai_chat_body(self, messages: list[dict], system: str = "", max_tokens: int = 4096, temperature: float = 0.7) -> dict:
        """Build an InvokeModel body for OpenAI-compatible Bedrock models."""
        return {
            "messages": self._build_openai_chat_messages(messages, system),
            "max_tokens": max(1, min(max_tokens, 16384)),
            "temperature": max(0, min(temperature, 1)),
        }

    def _parse_openai_chat_response(self, result: dict) -> dict:
        """Parse OpenAI-compatible chat responses such as Moonshot/Kimi."""
        choices = result.get("choices", [])
        content = ""
        if choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            message = choice.get("message", {}) if isinstance(choice.get("message", {}), dict) else {}
            content = message.get("content") or choice.get("text") or ""
        if not content and "content" in result:
            content = result["content"]
            if isinstance(content, list):
                content = content[0].get("text", str(content[0])) if content else ""
        if not content and "generation" in result:
            content = result["generation"]

        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("inputTokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("outputTokens") or max(1, len(str(content)) // 4) if content else 0
        content = normalize_answer_text(content)
        return {
            "content": str(content),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _build_deepseek_messages(self, messages: list[dict], system: str = "") -> list[dict]:
        """Build DeepSeek chat messages for Bedrock's native Invoke API."""
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            if role not in {"user", "assistant"}:
                role = "user"
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content)
            converted.append({"role": role, "content": str(content)})

        if system and converted:
            converted[0]["content"] = f"System instructions:\n{system}\n\nUser message:\n{converted[0]['content']}"
        elif system:
            converted.append({"role": "user", "content": f"System instructions:\n{system}"})

        return converted

    def _build_deepseek_body(self, model_id: str, max_tokens: int, temperature: float, prompt: str = None, messages: list[dict] = None, system: str = "") -> dict:
        """Build a DeepSeek request body using AWS-documented native shapes."""
        safe_max_tokens = max(1, min(max_tokens, 8192))
        safe_temperature = max(0, min(temperature, 1))

        if "r1" in model_id.lower():
            if messages:
                prompt_text = "\n".join(f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages)
            else:
                prompt_text = prompt or ""
            if system:
                prompt_text = f"{system}\n\n{prompt_text}"
            formatted_prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt_text}<｜Assistant｜><think>\n"
            return {
                "prompt": formatted_prompt,
                "max_tokens": safe_max_tokens,
                "temperature": safe_temperature,
                "top_p": 0.9,
            }

        deepseek_messages = self._build_deepseek_messages(
            messages if messages is not None else [{"role": "user", "content": prompt or ""}],
            system,
        )
        return {
            "messages": deepseek_messages,
            "max_tokens": safe_max_tokens,
            "temperature": safe_temperature,
            "top_p": 0.9,
        }

    def _extract_prompt_from_body(self, body: dict) -> str:
        """Extract readable text from a non-Claude body for fallback prompts."""
        if not isinstance(body, dict):
            return str(body)
        if body.get("prompt"):
            return str(body["prompt"])
        if body.get("inputText"):
            return str(body["inputText"])
        if body.get("messages"):
            parts = []
            for msg in body["messages"]:
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                if isinstance(content, list):
                    content = " ".join(part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in content)
                parts.append(str(content))
            return "\n".join(parts)
        return json.dumps(body)

    def _invoke_with_inference_profile(self, model_id: str, body: dict, model_info: dict) -> dict:
        """
        Try to invoke model with inference profile support.
        Some models require inference profiles (like Llama 3.2 90B).
        """
        api_format = model_info.get("api_format", "unknown")
        
        # First try direct invocation
        try:
            response = self._invoke_model_with_retries(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            result = self._parse_response(response, api_format)
            return result
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            
            # Check if we need an inference profile
            if "inference profile" in error_msg.lower() or "on-demand throughput" in error_msg.lower():
                print(f"⚠️  Model requires inference profile. Trying to find one...")
                
                # Try to find inference profiles for this model
                inference_profile_id = self._find_inference_profile(model_id)
                
                if inference_profile_id:
                    print(f"🔄 Using inference profile: {inference_profile_id}")
                    try:
                        response = self._invoke_model_with_retries(
                            modelId=inference_profile_id,
                            contentType="application/json",
                            accept="application/json",
                            body=json.dumps(body),
                        )
                        return self._parse_response(response, api_format)
                    except Exception as profile_error:
                        print(f"❌ Inference profile also failed: {profile_error}")
                
                # Fallback: try Claude as backup
                print(f"🔄 Falling back to Claude 3 Haiku...")
                return self._invoke_claude_fallback(body)
            
            raise  # Re-raise other errors
        except Exception as e:
            raise

    def _find_inference_profile(self, model_id: str) -> Optional[str]:
        """Try to find an inference profile for a model."""
        try:
            # List inference profiles
            bedrock = boto3.client('bedrock', region_name=AWS_REGION)
            response = bedrock.list_inference_profiles()
            
            for profile in response.get('inferenceProfileSummaries', []):
                # Check if profile contains our model
                if model_id in str(profile.get('modelConfigs', [])):
                    return profile['inferenceProfileId']
                    
        except Exception as e:
            print(f"⚠️  Could not find inference profile: {e}")
        
        return None

    def _parse_response(self, response, api_format: str) -> dict:
        """Parse model response based on API format."""
        result = json.loads(response["body"].read())

        if api_format == "claude":
            content = result.get("content", [{}])[0].get("text", "No response")
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        elif api_format == "llama":
            content = result.get("generation", "No response")
            input_tokens = result.get("prompt_token_count", 0)
            output_tokens = result.get("generation_token_count", 0)
        elif api_format == "titan":
            content = result.get("results", [{}])[0].get("outputText", "No response")
            input_tokens = len(json.dumps(result)) // 4
            output_tokens = len(content) // 4
        elif api_format in ("deepseek", "openai_chat"):
            parsed = self._parse_openai_chat_response(result)
            content = parsed["content"]
            input_tokens = parsed["input_tokens"]
            output_tokens = parsed["output_tokens"]
        else:
            if "choices" in result:
                parsed = self._parse_openai_chat_response(result)
                content = parsed["content"]
                input_tokens = parsed["input_tokens"]
                output_tokens = parsed["output_tokens"]
            else:
                content = result.get("content", [{}])[0].get("text", "No response")
                usage = result.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        content = normalize_answer_text(content)
        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _streaming_supported_for_model(self, model_id: str, api_format: str) -> bool:
        """Return whether we should attempt Bedrock response streaming."""
        model_lower = model_id.lower()
        if "deepseek" in model_lower or api_format == "deepseek":
            return False
        return True

    def _extract_stream_text_delta(self, chunk: dict) -> str:
        """Extract text delta from common Bedrock streaming chunk formats."""
        if not isinstance(chunk, dict):
            return ""

        if "contentBlockDelta" in chunk:
            delta = chunk.get("contentBlockDelta", {}).get("delta", {})
            return delta.get("text", "")

        if chunk.get("type") == "content_block_delta":
            return chunk.get("delta", {}).get("text", "")

        if "delta" in chunk and isinstance(chunk["delta"], dict):
            return chunk["delta"].get("text", "")

        if "completion" in chunk:
            return chunk.get("completion", "")

        choices = chunk.get("choices", [])
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                return delta.get("content") or delta.get("text") or ""
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return message.get("content", "")

        return ""

    def _extract_stream_usage(self, chunk: dict) -> dict:
        """Extract token usage from stream metadata when a provider sends it."""
        usage = {}
        if not isinstance(chunk, dict):
            return usage

        if "metadata" in chunk:
            usage.update(chunk.get("metadata", {}).get("usage", {}))
        if "usage" in chunk:
            usage.update(chunk.get("usage", {}))
        if chunk.get("type") == "message_delta":
            usage.update(chunk.get("usage", {}))
        if "amazon-bedrock-invocationMetrics" in chunk:
            metrics = chunk.get("amazon-bedrock-invocationMetrics", {})
            usage.setdefault("inputTokens", metrics.get("inputTokenCount", 0))
            usage.setdefault("outputTokens", metrics.get("outputTokenCount", 0))

        return usage

    def _invoke_with_response_stream(self, model_id: str, body: dict, model_info: dict, on_text) -> dict:
        """Invoke a model with streaming and call on_text for each text delta."""
        response = self._invoke_stream_with_retries(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        content_parts = []
        input_tokens = 0
        output_tokens = 0

        for event in response.get("body", []):
            if "chunk" not in event:
                for error_key in (
                    "internalServerException",
                    "modelStreamErrorException",
                    "validationException",
                    "throttlingException",
                    "modelTimeoutException",
                    "serviceUnavailableException",
                ):
                    if error_key in event:
                        message = event[error_key].get("message", str(event[error_key]))
                        raise RuntimeError(f"{error_key}: {message}")
                continue

            raw = event["chunk"].get("bytes", b"")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not raw:
                continue

            chunk = json.loads(raw)
            text = self._extract_stream_text_delta(chunk)
            if text:
                content_parts.append(text)
                on_text(text)

            usage = self._extract_stream_usage(chunk)
            input_tokens = (
                usage.get("input_tokens")
                or usage.get("inputTokens")
                or usage.get("prompt_tokens")
                or input_tokens
            )
            output_tokens = (
                usage.get("output_tokens")
                or usage.get("outputTokens")
                or usage.get("completion_tokens")
                or output_tokens
            )

        content = normalize_answer_text("".join(content_parts))
        if not output_tokens:
            output_tokens = max(1, len(content) // 4) if content else 0
        if not input_tokens:
            input_tokens = max(1, len(json.dumps(body)) // 4)

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _invoke_claude_fallback(self, body: dict) -> dict:
        """Fall back to Claude 3 Haiku when other models fail."""
        fallback_model = "anthropic.claude-3-haiku-20240307-v1:0"
        
        try:
            # Adjust body for Claude format if needed
            if body.get("anthropic_version") != "bedrock-2023-05-31":
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS_EXPENSIVE,
                    "temperature": TEMPERATURE,
                    "messages": [{"role": "user", "content": self._extract_prompt_from_body(body)}],
                }
            
            response = self._invoke_model_with_retries(
                modelId=fallback_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            
            result = json.loads(response["body"].read())
            content = result.get("content", [{}])[0].get("text", "No response")
            usage = result.get("usage", {})
            
            return {
                "content": content,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "model_used": fallback_model,
                "original_model_failed": True
            }
            
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            return {
                "content": "Sorry, I'm unable to process your request at the moment. Please try again or contact support.",
                "input_tokens": 0,
                "output_tokens": 0,
                "model_used": "none",
                "original_model_failed": True
            }

    def _invoke_claude(self, messages: list[dict], system: str = SYSTEM_PROMPT, stream_callback=None) -> dict:
        """
        Send a request to the main model (expensive) via Bedrock.
        Handles all model types with proper inference profile support.

        Args:
            messages: List of message dicts with "role" and "content" keys
            system: System prompt to set behavior

        Returns:
            Dict with "content", "input_tokens", and "output_tokens"
        """
        try:
            # Get model info to determine format and capabilities
            model_info = self._get_model_info_by_id(EXPENSIVE_MODEL_ID)
            
            api_format = model_info.get("api_format", "claude")
            model_lower = EXPENSIVE_MODEL_ID.lower()
            
            # Honour per-model output-token ceiling (never ask for more than AWS allows)
            effective_max_tokens = MODEL_CATALOG_INSTANCE.get_max_output_tokens(EXPENSIVE_MODEL_ID)
            
            # ─── Build body for the specific model type ─────────────────
            if "anthropic" in model_lower or "claude" in model_lower or api_format == "claude":
                # Claude format
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": effective_max_tokens,
                    "temperature": TEMPERATURE,
                    "system": system,
                    "messages": messages,
                }
            
            elif "deepseek" in model_lower or api_format == "deepseek":
                # DeepSeek format
                body = self._build_deepseek_body(
                    model_id=EXPENSIVE_MODEL_ID,
                    messages=messages,
                    system=system,
                    max_tokens=effective_max_tokens,
                    temperature=TEMPERATURE,
                )

            elif "moonshot" in model_lower or "kimi" in model_lower or api_format == "openai_chat":
                # OpenAI-compatible chat format used by Moonshot/Kimi on Bedrock InvokeModel
                body = self._build_openai_chat_body(
                    messages=messages,
                    system=system,
                    max_tokens=effective_max_tokens,
                    temperature=TEMPERATURE,
                )
            
            elif "meta" in model_lower or "llama" in model_lower or api_format == "llama":
                # Llama format - convert messages to prompt
                prompt = self._convert_messages_to_llama_prompt(messages, system)
                body = {
                    "prompt": prompt,
                    "max_gen_len": effective_max_tokens,
                    "temperature": TEMPERATURE,
                }
            
            elif "amazon." in model_lower or "titan" in model_lower or api_format == "titan":
                # Titan format
                prompt = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": effective_max_tokens,
                        "temperature": TEMPERATURE,
                    }
                }
            
            else:
                # Fallback to Claude format for unknown models
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": effective_max_tokens,
                    "temperature": TEMPERATURE,
                    "system": system,
                    "messages": messages,
                }
            
            # ─── Invoke with inference profile support ──────────────────
            if stream_callback and STREAM_RESPONSES:
                if self._streaming_supported_for_model(EXPENSIVE_MODEL_ID, api_format):
                    try:
                        result = self._invoke_with_response_stream(EXPENSIVE_MODEL_ID, body, model_info, stream_callback)
                    except Exception as stream_error:
                        print(f"\n   Streaming unavailable ({stream_error}). Falling back to normal response...")
                        self._last_response_was_streamed = False
                        result = self._invoke_with_inference_profile(EXPENSIVE_MODEL_ID, body, model_info)
                else:
                    print("   Selected model does not support Bedrock response streaming. Using normal response...")
                    result = self._invoke_with_inference_profile(EXPENSIVE_MODEL_ID, body, model_info)
            else:
                result = self._invoke_with_inference_profile(EXPENSIVE_MODEL_ID, body, model_info)
            
            # Parse based on API format
            if api_format == "claude" or "anthropic" in model_lower or "claude" in model_lower:
                content = result.get("content", "No response")
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
            
            elif api_format == "llama" or "meta" in model_lower or "llama" in model_lower:
                content = result.get("content", "No response")
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
            
            elif api_format == "titan" or "titan" in model_lower:
                content = result.get("content", "No response")
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
            
            else:
                content = result.get("content", "No response")
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
            
            # Track the usage
            self.tracker.add_expensive_usage(input_tokens, output_tokens)

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Main model error: {error_msg}")
            
            # If it's an inference profile error, try fallback
            if "inference profile" in error_msg.lower():
                print(f"🔄 Trying Claude Haiku fallback...")
                try:
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": MAX_TOKENS_EXPENSIVE,
                        "temperature": TEMPERATURE,
                        "system": system,
                        "messages": messages,
                    }
                    result = self._invoke_claude_fallback(body)
                    self.tracker.add_expensive_usage(result.get("input_tokens", 0), result.get("output_tokens", 0))
                    return result
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
                    raise
            
            raise

    def _invoke_cheap_model(self, prompt: str) -> dict:
        """
        Send a request to the cheap model for summarization.
        Universal handler for ALL Bedrock models.
        """
        try:
            # Get model info to determine format
            model_info = None
            for key, info in MODEL_CATALOG.items():
                if info["model_id"] == CHEAP_MODEL_ID:
                    model_info = info
                    break
            
            if not model_info:
                # Default to Claude format if model not in catalog
                model_info = {"api_format": "claude", "provider": "Unknown"}
            
            api_format = model_info.get("api_format", "claude")
            provider = model_info.get("provider", "").lower()
            
            # Universal model detection
            model_lower = CHEAP_MODEL_ID.lower()
            
            # ─── Claude Models (Anthropic) ───────────────────────────────────
            if "deepseek" in model_lower or api_format == "deepseek":
                body = self._build_deepseek_body(
                    model_id=CHEAP_MODEL_ID,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS_CHEAP,
                    temperature=0.3,
                )
                result = self._invoke_with_inference_profile(CHEAP_MODEL_ID, body, model_info)
                content = result.get("content", "No response")
                input_tokens = result.get("input_tokens", len(prompt) // 4)
                output_tokens = result.get("output_tokens", len(content) // 4)
            
            elif "anthropic" in model_lower or "claude" in model_lower or api_format == "claude":
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS_CHEAP,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                response = self._invoke_model_with_retries(
                    modelId=CHEAP_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                
                result = json.loads(response["body"].read())
                content = result.get("content", [{}])[0].get("text", "No response")
                usage = result.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            
            # ─── Amazon Titan & Nova Models ──────────────────────────────────
            elif "amazon." in model_lower or "titan" in model_lower or "nova" in model_lower or api_format == "nova":
                # Try Titan Text format first
                try:
                    body = {
                        "inputText": prompt,
                        "textGenerationConfig": {
                            "maxTokenCount": MAX_TOKENS_CHEAP,
                            "temperature": 0.3,
                        }
                    }
                    
                    response = self._invoke_model_with_retries(
                        modelId=CHEAP_MODEL_ID,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body),
                    )
                    
                    result = json.loads(response["body"].read())
                    content = result.get("results", [{}])[0].get("outputText", "No response")
                    input_tokens = len(prompt) // 4  # Estimate
                    output_tokens = len(content) // 4
                    
                except:
                    # Try Nova Messages format
                    body = {
                        "messages": [{"role": "user", "content": [{"text": prompt}]}],
                        "inferenceConfig": {
                            "maxTokens": MAX_TOKENS_CHEAP,
                            "temperature": 0.3,
                        },
                    }
                    
                    response = self._invoke_model_with_retries(
                        modelId=CHEAP_MODEL_ID,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body),
                    )
                    
                    result = json.loads(response["body"].read())
                    content = result.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "No response")
                    usage = result.get("usage", {})
                    input_tokens = usage.get("inputTokens", 0)
                    output_tokens = usage.get("outputTokens", 0)
            
            # ─── Meta Llama Models ──────────────────────────────────────────
            elif "meta." in model_lower or "llama" in model_lower or api_format == "llama":
                # Try both Llama 2 and Llama 3 formats
                if "llama2" in model_lower:
                    body = {
                        "prompt": f"[INST] {prompt} [/INST]",
                        "max_gen_len": MAX_TOKENS_CHEAP,
                        "temperature": 0.3,
                    }
                else:  # Llama 3 format
                    body = {
                        "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                        "max_gen_len": MAX_TOKENS_CHEAP,
                        "temperature": 0.3,
                    }
                
                response = self._invoke_model_with_retries(
                    modelId=CHEAP_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                
                result = json.loads(response["body"].read())
                content = result.get("generation", "No response")
                input_tokens = result.get("prompt_token_count", 0)
                output_tokens = result.get("generation_token_count", 0)
            
            # ─── Mistral & Mixtral Models ───────────────────────────────────
            elif "mistral" in model_lower or "mixtral" in model_lower or api_format == "mistral":
                body = {
                    "prompt": f"[INST] {prompt} [/INST]",
                    "max_tokens": MAX_TOKENS_CHEAP,
                    "temperature": 0.3,
                }
                
                response = self._invoke_model_with_retries(
                    modelId=CHEAP_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                
                result = json.loads(response["body"].read())
                outputs = result.get("outputs", [{}])
                content = outputs[0].get("text", "No response") if outputs else "No response"
                input_tokens = len(prompt) // 4
                output_tokens = len(content) // 4
            
            # ─── Cohere Models ──────────────────────────────────────────────
            elif "cohere" in model_lower or "command" in model_lower or api_format == "cohere":
                body = {
                    "prompt": prompt,
                    "max_tokens": MAX_TOKENS_CHEAP,
                    "temperature": 0.3,
                }
                
                response = self._invoke_model_with_retries(
                    modelId=CHEAP_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                
                result = json.loads(response["body"].read())
                content = result.get("generations", [{}])[0].get("text", "No response")
                meta = result.get("meta", {}).get("tokens", {})
                input_tokens = meta.get("input_tokens", 0)
                output_tokens = meta.get("output_tokens", 0)
            
            # ─── AI21 Jamba Models ──────────────────────────────────────────
            elif "ai21" in model_lower or "jamba" in model_lower or api_format == "ai21":
                body = {
                    "prompt": prompt,
                    "maxTokens": MAX_TOKENS_CHEAP,
                    "temperature": 0.3,
                }
                
                response = self._invoke_model_with_retries(
                    modelId=CHEAP_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                
                result = json.loads(response["body"].read())
                content = result.get("completions", [{}])[0].get("data", {}).get("text", "No response")
                input_tokens = len(prompt) // 4
                output_tokens = len(content) // 4
            
            # ─── DEFAULT Fallback ──────────────────────────────────────────
            else:
                print(f"⚠️  Unknown model format for {CHEAP_MODEL_ID}, using Claude fallback")
                # Try Claude format as universal fallback
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS_CHEAP,
                    "temperature": 0.3,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                try:
                    response = self._invoke_model_with_retries(
                        modelId=CHEAP_MODEL_ID,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body),
                    )
                    
                    result = json.loads(response["body"].read())
                    content = result.get("content", [{}])[0].get("text", "No response")
                    usage = result.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                except:
                    # Ultimate fallback
                    content = f"Model {CHEAP_MODEL_ID} response failed. Please check model compatibility."
                    input_tokens = output_tokens = 0

            # Track the usage
            self.tracker.add_cheap_usage(input_tokens, output_tokens)

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as e:
            print(f"❌ Cheap model error ({CHEAP_MODEL_ID}): {e}")
            raise

    def _create_summary(self, messages: list[dict]) -> str:
        """
        Use the cheap model to create a concise summary of older messages.
        This is the key cost-optimization technique!

        Instead of sending ALL old messages to the expensive model,
        we summarize them cheaply and send only the summary.

        Args:
            messages: List of older messages to summarize

        Returns:
            A concise summary string
        """
        if not messages:
            return ""

        # Format the messages into a readable conversation
        conversation_text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if msg["role"] == "assistant":
                content = normalize_answer_text(content)
            # Truncate very long messages for the summary
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            conversation_text += f"{role}: {content}\n\n"

        # Create the summarization prompt
        summary_prompt = f"""Please create a concise technical summary of the following conversation. 
Focus on:
1. Key topics discussed
2. Important decisions or conclusions reached
3. Any specific technical details, code, or configurations mentioned
4. The user's main goals or questions
5. Any unresolved questions or ongoing tasks

Keep the summary under 300 words. Be precise and factual.

CONVERSATION TO SUMMARIZE:
{conversation_text}

CONCISE SUMMARY:"""

        print("   📝 Creating smart summary of older messages...")

        try:
            result = self._invoke_cheap_model(summary_prompt)
            summary = result["content"]

            print(
                f"   ✅ Summary created ({result['input_tokens']} in / {result['output_tokens']} out tokens)"
            )

            # Store the summary
            self.history.add_summary(summary, len(messages))

            return summary

        except Exception as e:
            print(f"   ⚠️  Summary creation failed: {e}")
            print("   Falling back to simple truncation...")

            # Fallback: just take the first and last few messages
            fallback_parts = []
            for msg in messages[:2] + messages[-2:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                content = msg["content"][:200]
                fallback_parts.append(f"{role}: {content}")

            return "Previous conversation (truncated): " + " | ".join(fallback_parts)

    def _build_context(self) -> list[dict]:
        """
        Build the optimal context to send to the expensive model.

        Strategy:
        - If conversation is short: send everything
        - If conversation is long: summarize old messages + send recent ones

        This saves money by reducing tokens sent to the expensive model!

        Returns:
            List of messages formatted for the Claude API
        """
        total_messages = len(self.history.messages)

        if total_messages == 0:
            return []

        if CONTEXT_MODE == "full":
            print(f"   Full conversation mode: sending all {total_messages} messages with no summary")
            context_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.history.messages
            ]
            if context_messages and context_messages[0]["role"] != "user":
                context_messages.insert(0, {"role": "user", "content": "Let's continue our conversation."})
            return self._fix_message_alternation(context_messages)

        # Short conversation - send everything as-is
        if total_messages <= MAX_MESSAGES_BEFORE_SUMMARY:
            print("   📨 Sending full conversation context (short conversation)")
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.history.messages
            ]

        # Long conversation - use smart summarization!
        print(f"   🧠 Smart context: {total_messages} messages total")

        # Get older messages (to be summarized) and recent messages (sent in full)
        older_messages = self.history.get_older_messages(RECENT_MESSAGES_COUNT)
        recent_messages = self.history.get_recent_messages(RECENT_MESSAGES_COUNT)

        # Check if we need a new summary or can reuse the existing one
        if (
            self._last_summary_msg_count < len(older_messages)
            or not self.history.get_latest_summary()
        ):
            summary = self._create_summary(older_messages)
            self._last_summary_msg_count = len(older_messages)
        else:
            summary = self.history.get_latest_summary()
            print("   ♻️  Reusing existing summary")

        # Build the context: summary as first user message, then recent messages
        context_messages = []

        if summary:
            # Add summary as a context-setting message
            context_messages.append(
                {
                    "role": "user",
                    "content": f"[CONVERSATION CONTEXT - Summary of our earlier discussion]\n{summary}\n\n[END OF SUMMARY - The recent messages follow below]",
                }
            )
            # Claude needs alternating user/assistant messages, so add acknowledgment
            context_messages.append(
                {
                    "role": "assistant",
                    "content": "I understand the context from our previous conversation. I'll keep that in mind as we continue. What would you like to discuss?",
                }
            )

        # Add recent messages in full
        for msg in recent_messages:
            context_messages.append({"role": msg["role"], "content": msg["content"]})

        # Ensure the conversation starts with a user message (Claude requirement)
        if context_messages and context_messages[0]["role"] != "user":
            context_messages.insert(
                0, {"role": "user", "content": "Let's continue our conversation."}
            )

        # Ensure alternating roles (Claude requirement)
        context_messages = self._fix_message_alternation(context_messages)

        print(
            f"   📊 Context size: {len(older_messages)} summarized + {len(recent_messages)} recent"
        )

        return context_messages

    def _fix_message_alternation(self, messages: list[dict]) -> list[dict]:
        """
        Ensure messages alternate between user and assistant roles.
        Claude requires strict alternation. This fixes any violations.
        """
        if not messages:
            return messages

        fixed = [messages[0]]

        for i in range(1, len(messages)):
            if messages[i]["role"] == fixed[-1]["role"]:
                # Same role twice in a row - merge them
                fixed[-1]["content"] += "\n\n" + messages[i]["content"]
            else:
                fixed.append(messages[i])

        # Ensure it ends with a user message (since we're about to get a response)
        if fixed and fixed[-1]["role"] != "user":
            # This shouldn't normally happen, but just in case
            pass

        return fixed

    def send_message(self, user_message: str, stream: bool = False) -> Optional[str]:
        """
        Send a message to Claude and get a response.
        This is the main function that handles the full flow:
        1. Add user message to history
        2. Build optimized context (with summarization if needed)
        3. Send to Claude
        4. Save response to history

        Args:
            user_message: The user's input text

        Returns:
            The assistant's response text, or None if there was an error
        """
        # Step 1: Save the user's message
        self.history.add_message("user", user_message)

        # Step 2: Build the context (with smart summarization)
        print("\n🔄 Preparing context...")
        context_messages = self._build_context()

        if not context_messages:
            # This shouldn't happen since we just added a message, but just in case
            context_messages = [{"role": "user", "content": user_message}]

        # Step 3: Send to the selected main model
        print(f"🤖 Sending to {EXPENSIVE_MODEL_ID.split('.')[-1]}...")
        start_time = time.time()
        self._last_response_was_streamed = False

        partial_chunks = []
        partial_chars_since_save = 0
        stream_filter = ReasoningStreamFilter()
        stream_line_start = True
        # When SHOW_THINKING is on we track whether we are inside a reasoning block
        # so we can render it in a contrasting colour.
        _in_think_block = [False]   # mutable container so inner func can write it

        def save_partial(force: bool = False):
            nonlocal partial_chars_since_save
            if not force and partial_chars_since_save < PARTIAL_SAVE_EVERY_CHARS:
                return
            partial_text = "".join(partial_chunks)
            self._save_request_state({
                "status": "streaming",
                "user_message": user_message,
                "model_id": EXPENSIVE_MODEL_ID,
                "context_mode": CONTEXT_MODE,
                "partial_response": partial_text,
                "input_tokens": 0,
                "output_tokens": max(0, len(partial_text) // 4),
                "started_at": request_started_at,
            })
            partial_chars_since_save = 0

        def render_visible_stream(text: str):
            nonlocal partial_chars_since_save, stream_line_start
            if not text:
                return
            if not self._last_response_was_streamed:
                width = terminal_width()
                print(f"\n{Colors.NEON_AZURE}{Colors.BOLD}{'=' * width}{Colors.RESET}")
                model_info = MODEL_CATALOG_INSTANCE.get_model_by_id(EXPENSIVE_MODEL_ID)
                model_name = model_info.get("display_name", "Assistant") if model_info else "Assistant"
                print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}Assistant  {Colors.NEON_LIME}{model_name}{Colors.RESET}")
                if SHOW_THINKING:
                    print(f"{Colors.NEON_ORANGE}{Colors.DIM}Thinking blocks shown in amber — final answer in white.{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}Streaming answer (reasoning hidden — use /think on to show).{Colors.RESET}")
                print(f"{Colors.NEON_AZURE}{Colors.BOLD}{'=' * width}{Colors.RESET}\n")
                self._last_response_was_streamed = True

            for part in text.splitlines(keepends=True):
                # Detect entry/exit of reasoning blocks for coloured output
                if SHOW_THINKING:
                    lower = part.lower()
                    if any(tag in lower for tag in ("<thinking", "<reasoning", "<think")):
                        _in_think_block[0] = True
                    if any(tag in lower for tag in ("</thinking>", "</reasoning>", "</think>")):
                        _in_think_block[0] = False
                        # Print closing tag line in think colour, then reset
                        if stream_line_start:
                            print(f"{Colors.DIM}{Colors.NEON_AZURE}| {Colors.RESET}", end="")
                        print(f"{Colors.NEON_ORANGE}{Colors.DIM}{part}{Colors.RESET}", end="", flush=True)
                        stream_line_start = part.endswith("\n")
                        continue

                if stream_line_start:
                    print(f"{Colors.DIM}{Colors.NEON_AZURE}| {Colors.RESET}", end="")

                if SHOW_THINKING and _in_think_block[0]:
                    # Render reasoning in dim amber so it's visually distinct
                    print(f"{Colors.NEON_ORANGE}{Colors.DIM}{part}{Colors.RESET}", end="", flush=True)
                else:
                    print(f"{Colors.BRIGHT_WHITE}{part}{Colors.RESET}", end="", flush=True)
                stream_line_start = part.endswith("\n")

            partial_chunks.append(text)
            partial_chars_since_save += len(text)
            save_partial()

        def stream_to_console(text: str):
            if SHOW_THINKING:
                # Pass text through unfiltered — let the coloured renderer handle it
                render_visible_stream(text)
            else:
                render_visible_stream(stream_filter.feed(text))

        request_started_at = datetime.datetime.now().isoformat()
        self._save_request_state({
            "status": "started",
            "user_message": user_message,
            "model_id": EXPENSIVE_MODEL_ID,
            "context_mode": CONTEXT_MODE,
            "partial_response": "",
            "started_at": request_started_at,
        })

        try:
            result = self._invoke_claude(
                context_messages,
                stream_callback=stream_to_console if stream and STREAM_RESPONSES else None,
            )
            render_visible_stream(stream_filter.flush())
            elapsed = time.time() - start_time

            assistant_response = normalize_answer_text(result["content"])
            if not str(assistant_response).strip():
                raise RuntimeError("Model returned an empty response. Check model format/access or switch models.")
            if self._last_response_was_streamed:
                save_partial(force=True)
                # Safety check: if no content was actually printed via streaming, fall back to printing it
                streamed_content = "".join(partial_chunks).strip()
                if not streamed_content:
                    # Streaming callback didn't output properly, print the response now
                    print(f"{Colors.BRIGHT_WHITE}{assistant_response}{Colors.RESET}\n")
                
                # Show footer after streaming completes
                print(f"\n{Colors.NEON_AZURE}{Colors.BOLD}{'=' * terminal_width()}{Colors.RESET}")
                # Show metadata after streamed response
                metadata = []
                if result["output_tokens"] > 0:
                    metadata.append(f"{Colors.NEON_PINK}⚡ Tokens: {result['output_tokens']:,}{Colors.RESET}")
                # Calculate cost for this response
                expensive_rates = COST_PER_1K.get(
                    EXPENSIVE_MODEL_ID, {"input": 0.003, "output": 0.015}
                )
                response_cost = (
                    result["input_tokens"] / 1000 * expensive_rates["input"] +
                    result["output_tokens"] / 1000 * expensive_rates["output"]
                )
                if response_cost > 0:
                    metadata.append(f"{Colors.NEON_ORANGE}💵 Cost: ${response_cost:.6f}{Colors.RESET}")
                if metadata:
                    print(f"{Colors.DIM}  {' • '.join(metadata)}{Colors.RESET}")
                print(f"{Colors.NEON_AZURE}{Colors.BOLD}{'=' * terminal_width()}{Colors.RESET}\n")

            # Track response metadata for display
            self._last_response_tokens = result["output_tokens"]
            
            # Calculate cost for this response
            expensive_rates = COST_PER_1K.get(
                EXPENSIVE_MODEL_ID, {"input": 0.003, "output": 0.015}
            )
            self._last_response_cost = (
                result["input_tokens"] / 1000 * expensive_rates["input"] +
                result["output_tokens"] / 1000 * expensive_rates["output"]
            )
            self._last_response_model = EXPENSIVE_MODEL_ID

            # Step 4: Save the assistant's response
            self.history.add_message("assistant", assistant_response)
            self._clear_request_state()
            self._auto_checkpoint_if_needed()

            # Print usage info
            print(
                f"   ⏱️  Response time: {elapsed:.1f}s | "
                f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out"
            )

            return assistant_response

        except Exception as e:
            partial_text = "".join(partial_chunks)
            self._save_request_state({
                "status": "failed",
                "user_message": user_message,
                "model_id": EXPENSIVE_MODEL_ID,
                "context_mode": CONTEXT_MODE,
                "partial_response": partial_text,
                "error": str(e),
                "started_at": request_started_at,
            })
            self.history.create_checkpoint("request_failed")

            print(f"\n❌ Error: {e}")
            print("Your message and any partial answer were saved. Restart or use recovery to continue safely.")
            return None

    def _auto_checkpoint_if_needed(self):
        """Create periodic safety checkpoints during long project chats."""
        message_count = len(self.history.messages)
        if message_count - self._last_checkpoint_msg_count < AUTOSAVE_CHECKPOINT_EVERY:
            return

        checkpoint = self.history.create_checkpoint("auto")
        if checkpoint:
            self._last_checkpoint_msg_count = message_count
            print(f"   💾 Auto-checkpoint saved: {checkpoint}")

    def upload_file(self, file_path: str, description: str = ""):
        """Upload a file to use in chat."""
        file_id = self.file_manager.upload_file(file_path, description)
        if file_id:
            # Add file to conversation context
            file_info = self.file_manager.get_file_info(file_id)
            file_message = f"[FILE UPLOADED: {file_info['original_name']}]\n"
            file_message += f"Description: {description}\n"
            file_message += f"Preview: {file_info['content_preview']}\n"
            
            self.history.add_message("user", file_message)
            print(f"📎 File added to conversation. Use /file show {file_id} to reference it.")
        
        return file_id

    def view_recent(self, count: int = 10):
        """Display the most recent messages in a formatted way with vibrant colors."""
        messages = self.history.messages[-count:] if self.history.messages else []

        if not messages:
            print(f"\n{Colors.warning('No messages yet. Start chatting!')}")
            return

        print(f"\n{Colors.NEON_PURPLE}{'='*60}")
        print(f"📋 Last {min(count, len(messages))} Messages")
        print(f"{'='*60}{Colors.RESET}")

        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                role_icon = f"{Colors.NEON_LIME}👤 You{Colors.RESET}"
            else:
                role_icon = f"{Colors.NEON_AZURE}🤖 Bot{Colors.RESET}"
            
            timestamp = msg.get("timestamp", "")

            # Format timestamp nicely
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%m/%d %H:%M")
                except (ValueError, TypeError):
                    pass

            print(f"\n{Colors.DIM}{'─'*60}{Colors.RESET}")
            print(f"{role_icon}  {Colors.DIM}[{timestamp}]{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

            # Truncate very long messages for display
            content = msg["content"]
            if len(content) > 500:
                # Colorize content based on role
                if msg["role"] == "user":
                    print(f"{Colors.BRIGHT_WHITE}{content[:500]}{Colors.RESET}")
                else:
                    print(f"{Colors.NEON_AZURE}{content[:500]}{Colors.RESET}")
                print(f"{Colors.DIM}... [message truncated, {len(content)} chars total]{Colors.RESET}")
            else:
                if msg["role"] == "user":
                    print(f"{Colors.BRIGHT_WHITE}{content}{Colors.RESET}")
                else:
                    print(f"{Colors.NEON_AZURE}{content}{Colors.RESET}")

        print(f"\n{Colors.NEON_PURPLE}{'='*60}")
        print(f"Total messages in history: {Colors.NEON_ORANGE}{len(self.history.messages)}{Colors.RESET}")
        print(f"{Colors.NEON_PURPLE}{'='*60}{Colors.RESET}")

    def view_full_summary(self):
        """Generate and display a summary of the entire conversation with vibrant colors."""
        if not self.history.messages:
            print(f"\n{Colors.warning('No messages to summarize.')}")
            return

        print(f"\n{Colors.NEON_PURPLE}{'='*60}")
        print("📊 Full Conversation Summary")
        print(f"{'='*60}{Colors.RESET}")

        # Show metadata
        print(f"{Colors.NEON_AZURE}Total messages:{Colors.RESET}    {Colors.NEON_ORANGE}{len(self.history.messages)}{Colors.RESET}")
        print(f"{Colors.NEON_AZURE}Created:{Colors.RESET}           {self.history.metadata.get('created', 'N/A')}")
        print(f"{Colors.NEON_AZURE}Last updated:{Colors.RESET}      {self.history.metadata.get('last_updated', 'N/A')}")

        # Count messages by role
        user_msgs = sum(1 for m in self.history.messages if m["role"] == "user")
        asst_msgs = sum(1 for m in self.history.messages if m["role"] == "assistant")
        print(f"{Colors.NEON_AZURE}Your messages:{Colors.RESET}     {Colors.NEON_LIME}{user_msgs}{Colors.RESET}")
        print(f"{Colors.NEON_AZURE}AI responses:{Colors.RESET}      {Colors.NEON_LIME}{asst_msgs}{Colors.RESET}")

        # Show existing summaries
        if self.history.summaries:
            print(f"\n{Colors.DIM}{'─'*60}{Colors.RESET}")
            print(f"{Colors.NEON_PINK}📝 Existing Summaries:{Colors.RESET}")
            for i, s in enumerate(self.history.summaries):
                print(f"\n  {Colors.NEON_ORANGE}Summary #{i+1}{Colors.RESET} {Colors.DIM}({s['messages_summarized']} messages summarized):{Colors.RESET}")
                print(f"  {Colors.WHITE}{s['summary'][:300]}...{Colors.RESET}")
        else:
            print(f"{Colors.DIM}No summaries generated yet.{Colors.RESET}")

        # Generate a fresh summary if we have enough messages
        if len(self.history.messages) > 3:
            print(f"\n{Colors.DIM}{'─'*60}{Colors.RESET}")
            print(f"{Colors.NEON_AZURE}Generating fresh summary of entire conversation...{Colors.RESET}")

            try:
                summary = self._create_summary(self.history.messages)
                print(f"\n{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"{Colors.BRIGHT_GREEN}📄 Full Summary:{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"{Colors.BRIGHT_WHITE}{summary}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.error(f'Could not generate summary: {e}')}")

        print(f"\n{Colors.NEON_PURPLE}{'='*60}{Colors.RESET}")

    def search_history(self, query: str):
        """Search through conversation history for a keyword or phrase."""
        if not self.history.messages:
            print("\n📭 No messages to search.")
            return

        query_lower = query.lower()
        matches = []

        for i, msg in enumerate(self.history.messages):
            if query_lower in msg["content"].lower():
                matches.append((i, msg))

        if not matches:
            print(f"\n🔍 No messages found containing '{query}'")
            return

        print(f"\n🔍 Found {len(matches)} messages containing '{query}':")
        print(f"{'='*60}")

        for idx, msg in matches:
            role_icon = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
            content = msg["content"]

            # Show a snippet around the match
            pos = content.lower().find(query_lower)
            start = max(0, pos - 50)
            end = min(len(content), pos + len(query) + 50)
            snippet = content[start:end]

            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."

            print(f"\n  [{idx+1}] {role_icon}: {snippet}")

        print(f"\n{'='*60}")


# ============================================================================
# MAIN MENU AND APPLICATION LOOP
# ============================================================================


def print_banner():
    """Print a nice welcome banner."""
    print(
        """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🤖 Bedrock Smart Chat v2.1 - Cost-Optimized AI Assistant 🤖   ║
║                                                                  ║
║   Features:                                                      ║
║   • Persistent conversation history                              ║
║   • Smart context summarization (saves $$$)                      ║
║   • Full model catalog with pricing                              ║
║   • Runtime-configurable settings                                ║
║   • Token usage tracking & cost estimation                       ║
║   • Search, export, & more                                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    )


def print_menu(chat=None):
    """Print the enhanced main menu with vibrant colors and varied themes."""
    # Get current models with availability check
    exp_info = MODEL_CATALOG_INSTANCE.get_model_by_id(EXPENSIVE_MODEL_ID)
    cheap_info = MODEL_CATALOG_INSTANCE.get_model_by_id(CHEAP_MODEL_ID)
    
    exp_name = exp_info["display_name"] if exp_info else MODEL_CATALOG.get("claude-3.5-sonnet-v2", {}).get("display_name", "Unknown")
    cheap_name = cheap_info["display_name"] if cheap_info else MODEL_CATALOG.get("nova-micro", {}).get("display_name", "Unknown")
    
    # Count files - safely handle if chat is not available
    file_count = 0
    if chat and hasattr(chat, 'file_manager'):
        try:
            file_count = len(chat.file_manager.list_files())
        except:
            file_count = 0
    
    menu = f"""{Colors.NEON_PURPLE}
┌─────────────────────────────────────────────────────────┐
│           🤖 BEDROCK RESEARCH CHAT 2.1                  │
├─────────────────────────────────────────────────────────┤{Colors.RESET}
{Colors.NEON_AZURE}│ Region: {Colors.NEON_LIME}{AWS_REGION:<12}{Colors.NEON_AZURE} Files: {Colors.NEON_ORANGE}{file_count:<3}{Colors.NEON_AZURE} │
│ Main:    {Colors.BRIGHT_GREEN}{exp_name[:40]:<40}{Colors.NEON_AZURE} │
│ Summary: {Colors.BRIGHT_GREEN}{cheap_name[:40]:<40}{Colors.NEON_AZURE} │{Colors.RESET}
{Colors.NEON_PURPLE}├─────────────────────────────────────────────────────────┤
│  {Colors.NEON_PINK}1.  💬 Send Message{Colors.NEON_PURPLE}          {Colors.NEON_PINK}6.  💰 Cost Statistics{Colors.NEON_PURPLE}   │
│  {Colors.NEON_PINK}2.  📋 View Recent{Colors.NEON_PURPLE}           {Colors.NEON_PINK}7.  🔄 Model Switch{Colors.NEON_PURPLE}      │
│  {Colors.NEON_PINK}3.  📊 Full Summary{Colors.NEON_PURPLE}          {Colors.NEON_PINK}8.  ⚙️  Settings{Colors.NEON_PURPLE}         │
│  {Colors.NEON_PINK}4.  🔍 Search History{Colors.NEON_PURPLE}        {Colors.NEON_PINK}9.  📚 Model Catalog{Colors.NEON_PURPLE}     │
│  {Colors.NEON_PINK}5.  📁 File Management{Colors.NEON_PURPLE}       {Colors.NEON_PINK}10. 🗑️  Clear History{Colors.NEON_PURPLE}    │
│{Colors.NEON_PINK}                              11. 🚪 Exit{Colors.NEON_PURPLE}               │
├─────────────────────────────────────────────────────────┤
│ Type {Colors.NEON_ORANGE}/help{Colors.NEON_PURPLE} for commands • Just type to chat            │
│ Upload files with {Colors.NEON_ORANGE}/file upload <path>{Colors.NEON_PURPLE}                 │
└─────────────────────────────────────────────────────────┘{Colors.RESET}
"""
    print(menu)


def safe_input(prompt: str, default: str = "") -> str:
    """
    Get input from user with error handling.
    Never crashes, always returns a string.
    """
    try:
        result = input(prompt)
        return result if result is not None else default
    except (KeyboardInterrupt, EOFError):
        print()
        return default
    except Exception:
        return default


def get_user_input(prompt: str = "You: ") -> str:
    """Get multi-line input from the user with intelligent paste detection."""
    print(f"\n{Colors.BRIGHT_CYAN}{prompt}{Colors.RESET}", end="")
    print(f"{Colors.DIM}(Type your message. Press Enter twice to send, or 'q' to cancel){Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

    lines = []
    empty_line_count = 0
    consecutive_empty_for_exit = 0

    while True:
        try:
            line = input()

            if line.strip().lower() == "q" and not lines:
                return ""

            if line == "":
                empty_line_count += 1
                consecutive_empty_for_exit += 1
                # Two consecutive empty lines triggers send
                if consecutive_empty_for_exit >= 2 and lines:
                    # Remove the last empty line that triggered the exit
                    if lines and lines[-1] == "":
                        lines.pop()
                    break
                lines.append("")
            else:
                empty_line_count = 0
                consecutive_empty_for_exit = 0
                lines.append(line)

        except (EOFError, KeyboardInterrupt):
            break

    # Remove trailing empty lines
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def normalize_choice(user_input: str) -> str:
    """
    Normalize user menu input to handle typos and variations.
    Makes the app bulletproof against user mistakes.
    """
    if not user_input:
        return ""

    cleaned = user_input.strip().lower()

    # Direct number matches
    if cleaned.isdigit():
        return cleaned

    # Common word mappings to menu numbers
    word_map = {
        # Option 1 - Send message
        "send": "1", "chat": "1", "message": "1", "msg": "1", "talk": "1",
        "write": "1", "type": "1", "new": "1",
        # Option 2 - View recent
        "recent": "2", "view": "2", "last": "2", "show": "2", "history": "2",
        "messages": "2", "recnt": "2", "lst": "2",
        # Option 3 - Summary
        "summary": "3", "sum": "3", "summarize": "3", "full": "3",
        "sumary": "3", "summry": "3",
        # Option 4 - Search
        "search": "4", "find": "4", "look": "4", "grep": "4",
        "serch": "4", "serach": "4", "sarch": "4",
        # Option 5 - File Management
        "file": "5", "files": "5", "upload": "5", "download": "5", "storage": "5",
        "attach": "5", "document": "5", "documents": "5",
        # Option 6 - Costs
        "cost": "6", "costs": "6", "price": "6", "money": "6", "tokens": "6",
        "usage": "6", "stats": "6", "cots": "6", "coost": "6",
        # Option 7 - Change models
        "model": "7", "models": "7", "switch": "7", "change": "7",
        "swap": "7", "modl": "7", "modle": "7", "mdel": "7",
        # Option 8 - Settings
        "settings": "8", "config": "8", "configure": "8", "options": "8",
        "prefs": "8", "preferences": "8", "setup": "8", "setting": "8",
        "setings": "8", "settigns": "8", "confg": "8",
        # Option 9 - Catalog
        "catalog": "9", "catalogue": "9", "list": "9", "available": "9",
        "pricing": "9", "prices": "9", "catlog": "9", "catlogue": "9",
        # Option 10 - Clear
        "clear": "10", "delete": "10", "erase": "10", "reset": "10",
        "wipe": "10", "clr": "10", "cler": "10", "claer": "10",
        # Option 11 - Exit
        "exit": "11", "quit": "11", "bye": "11", "close": "11",
        "leave": "11", "q": "11", "end": "11", "exti": "11",
        "eixt": "11", "ext": "11", "quiit": "11",
    }

    # Check if the cleaned input matches any word
    if cleaned in word_map:
        return word_map[cleaned]

    # Check if any word key is contained in the input
    for word, num in word_map.items():
        if word in cleaned:
            return num

    # If input is longer than 10 chars, it's probably a direct message
    if len(cleaned) > 10:
        return "direct_message"

    return cleaned


def print_chat_header(chat):
    """Show a compact chat-first workspace header with vibrant colors."""
    exp_info = MODEL_CATALOG_INSTANCE.get_model_by_id(EXPENSIVE_MODEL_ID)
    cheap_info = MODEL_CATALOG_INSTANCE.get_model_by_id(CHEAP_MODEL_ID)
    exp_name = exp_info["display_name"] if exp_info else EXPENSIVE_MODEL_ID
    cheap_name = cheap_info["display_name"] if cheap_info else CHEAP_MODEL_ID
    
    header = f"""
{Colors.NEON_PURPLE}{'=' * 72}
🤖 Bedrock Project Chat
{'=' * 72}{Colors.RESET}
{Colors.NEON_AZURE}Region:{Colors.RESET} {Colors.NEON_LIME}{AWS_REGION}{Colors.RESET}
{Colors.NEON_AZURE}Main Model:{Colors.RESET}    {Colors.BRIGHT_GREEN}{exp_name}{Colors.RESET}
{Colors.NEON_AZURE}Memory:{Colors.RESET}       {Colors.NEON_ORANGE}{len(chat.history.messages)} messages{Colors.RESET} | {Colors.NEON_ORANGE}Files: {len(chat.file_manager.list_files())}{Colors.RESET}
{Colors.NEON_AZURE}Summary Model:{Colors.RESET} {Colors.BRIGHT_GREEN}{cheap_name}{Colors.RESET}
{Colors.NEON_AZURE}Context:{Colors.RESET}      {CONTEXT_MODE} | {Colors.NEON_AZURE}Streaming:{Colors.RESET} {Colors.BRIGHT_GREEN if STREAM_RESPONSES else Colors.BRIGHT_RED}{'✓ on' if STREAM_RESPONSES else '✗ off'}{Colors.RESET}
{Colors.DIM}Just type to chat. Paste freely — long pastes auto-detected. /p for manual paste mode. /think to toggle reasoning. /help for all commands. /quit to exit.{Colors.RESET}
{Colors.NEON_PURPLE}{'=' * 72}{Colors.RESET}
"""
    print(header)


def read_chat_input() -> str:
    """
    Read user input with reliable multi-line paste support.

    • Normal typing  – type one line, press Enter → sent immediately.
    • Paste (any OS) – paste any amount of text; all lines are collected
                       automatically because stdin is checked for buffered
                       data after every line (50 ms look-ahead window).
                       No special command or terminator needed.
    • /multi or /p   – explicit multi-line mode: type freely, enter END on
                       its own line to send.  Use this on Windows or when
                       the automatic paste detection misses something.
    • Backslash cont – end a line with \\ to continue to the next line.
    """
    first = safe_input(f"\n{Colors.BRIGHT_CYAN}You:{Colors.RESET} ")

    # ── Explicit multi-line mode ─────────────────────────────────────────────
    if first.strip().lower() in ("/multi", "/paste", "/p", "/m"):
        return _collect_multiline_block()

    # ── Backslash line continuation ──────────────────────────────────────────
    if first.rstrip().endswith("\\"):
        lines = [first.rstrip()[:-1]]
        print(f"{Colors.DIM}(Continue. Blank line OR type END to send.){Colors.RESET}")
        while True:
            line = safe_input(f"{Colors.DIM}... {Colors.RESET}")
            if line.strip().upper() == "END" or line == "":
                break
            lines.append(line.rstrip("\\"))
            if not line.rstrip().endswith("\\"):
                break
        return "\n".join(lines)

    # ── Automatic paste detection (Unix/Linux/macOS) ─────────────────────────
    # After reading the first line we wait up to 80 ms for more data in stdin.
    # A human typing cannot produce a second line that fast; a paste always can.
    if sys.platform != "win32":
        try:
            all_lines = [first]
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.08)
                if not ready:
                    break                           # stdin quiet → no more paste data
                chunk = sys.stdin.readline()
                if not chunk:                       # EOF
                    break
                all_lines.append(chunk.rstrip("\n"))
            if len(all_lines) > 1:
                combined = "\n".join(all_lines)
                char_count = len(combined)
                print(
                    f"{Colors.DIM}[Paste detected: {char_count:,} chars / "
                    f"{len(all_lines)} lines]{Colors.RESET}"
                )
                return combined
        except Exception:
            pass  # select not available on this platform — fall through

    return first


def _collect_multiline_block() -> str:
    """Explicit multi-line input mode. Type freely; send by entering END alone."""
    print(f"\n{Colors.NEON_ORANGE}── Multi-line / paste mode ──────────────────────────────────{Colors.RESET}")
    print(f"{Colors.DIM}Paste or type your message. Type  END  on its own line to send,")
    print(f"or CANCEL to abort.{Colors.RESET}")
    print(f"{Colors.NEON_ORANGE}─────────────────────────────────────────────────────────────{Colors.RESET}")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        stripped = line.strip().upper()
        if stripped == "END":
            break
        if stripped == "CANCEL":
            return ""
        lines.append(line)
    # Remove trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    result = "\n".join(lines)
    if result:
        print(f"{Colors.DIM}[{len(result):,} chars collected]{Colors.RESET}")
    return result


def print_assistant_response(response: str, model_name: str = "Assistant", token_count: int = 0, cost_estimate: float = 0.0):
    """Print assistant output in a readable, markdown-aware terminal layout."""
    response = normalize_answer_text(response)
    width = terminal_width()
    content_width = max(48, width - 4)

    print(f"\n{Colors.NEON_AZURE}{Colors.BOLD}{'=' * width}{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}Assistant  {Colors.NEON_LIME}{model_name}{Colors.RESET}")
    if token_count > 0 or cost_estimate > 0:
        metadata = []
        if token_count > 0:
            metadata.append(f"Tokens: {token_count:,}")
        if cost_estimate > 0:
            metadata.append(f"Cost: ${cost_estimate:.6f}")
        print(f"{Colors.DIM}{' | '.join(metadata)}{Colors.RESET}")
    print(f"{Colors.NEON_AZURE}{Colors.BOLD}{'=' * width}{Colors.RESET}")

    in_code = False
    previous_blank = False

    for raw_line in response.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if not previous_blank:
                print()
            previous_blank = True
            continue
        previous_blank = False

        if stripped.startswith("```"):
            in_code = not in_code
            label = stripped.strip("`").strip()
            title = f" code: {label} " if label else " code "
            print(f"{Colors.DIM}{Colors.NEON_VIOLET}+{title:-^{content_width}}+{Colors.RESET}")
            continue

        if in_code:
            wrapped = textwrap.wrap(line, width=content_width - 2, replace_whitespace=False, drop_whitespace=False) or [""]
            for item in wrapped:
                print(f"{Colors.DIM}{Colors.NEON_VIOLET}| {item}{Colors.RESET}")
            continue

        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            print(f"\n{Colors.NEON_ORANGE}{Colors.BOLD}{title}{Colors.RESET}")
            print(f"{Colors.DIM}{'-' * min(len(title), content_width)}{Colors.RESET}")
            continue

        bullet_prefix = ""
        bullet_text = stripped
        if re.match(r"^[-*]\s+", stripped):
            bullet_prefix = "- "
            bullet_text = re.sub(r"^[-*]\s+", "", stripped)
        elif re.match(r"^\d+[.)]\s+", stripped):
            match = re.match(r"^(\d+[.)]\s+)(.*)", stripped)
            bullet_prefix = match.group(1)
            bullet_text = match.group(2)

        if bullet_prefix:
            wrap_width = max(32, content_width - len(bullet_prefix) - 2)
            wrapped = textwrap.wrap(bullet_text, width=wrap_width) or [""]
            print(f"{Colors.NEON_AZURE}| {Colors.NEON_PINK}{bullet_prefix}{Colors.BRIGHT_WHITE}{wrapped[0]}{Colors.RESET}")
            for item in wrapped[1:]:
                print(f"{Colors.NEON_AZURE}| {' ' * len(bullet_prefix)}{Colors.BRIGHT_WHITE}{item}{Colors.RESET}")
            continue

        if stripped.startswith(">"):
            quote = stripped.lstrip("> ").strip()
            for item in textwrap.wrap(quote, width=content_width - 3):
                print(f"{Colors.NEON_VIOLET}| {Colors.ITALIC}{item}{Colors.RESET}")
            continue

        for item in textwrap.wrap(stripped, width=content_width):
            print(f"{Colors.NEON_AZURE}| {Colors.BRIGHT_WHITE}{item}{Colors.RESET}")

    print(f"{Colors.NEON_AZURE}{Colors.BOLD}{'=' * width}{Colors.RESET}\n")


def print_user_message(message: str):
    """Print user message with vibrant colors and formatting."""
    print(f"{Colors.NEON_LIME}{Colors.BOLD}👤  You{Colors.RESET}")
    print(f"{Colors.NEON_LIME}{'─' * 80}{Colors.RESET}")
    
    # Split and format message for readability
    paragraphs = message.split('\n\n')
    for i, para in enumerate(paragraphs):
        lines = para.split('\n')
        for line in lines:
            print(f"{Colors.BRIGHT_WHITE}{line}{Colors.RESET}")
        if i < len(paragraphs) - 1:
            print()
    
    print(f"{Colors.NEON_LIME}{'─' * 80}\n{Colors.RESET}")


def run_chat_console(chat):
    """Run the default conversation-first interface."""
    print_chat_header(chat)
    if chat.history.messages:
        print("Continuing previous conversation. Use /recent 6 to review the latest turns.")

    while True:
        try:
            raw_input = read_chat_input()
            if not raw_input.strip():
                continue

            command_result = chat.commands.process(raw_input.strip())
            if command_result is not None:
                if command_result.startswith("__EXIT__"):
                    parts = command_result.split("::", 1)
                    if len(parts) == 2:
                        print(f"{Colors.success(f'Saved exit checkpoint: {parts[1]}')}")
                    print(f"{Colors.success('Goodbye. Conversation saved.')}")
                    break
                if command_result.strip():
                    print(command_result)
                continue

            # Print user message with colors
            print_user_message(raw_input)
            
            response = chat.send_message(raw_input, stream=STREAM_RESPONSES)
            if response:
                # Only show formatted response if it wasn't already streamed with formatting
                if not chat._last_response_was_streamed:
                    # Get model display name
                    model_info = MODEL_CATALOG_INSTANCE.get_model_by_id(chat._last_response_model)
                    model_name = model_info.get("display_name", "Assistant") if model_info else "Assistant"
                    print_assistant_response(
                        response, 
                        model_name=model_name,
                        token_count=chat._last_response_tokens,
                        cost_estimate=chat._last_response_cost
                    )

        except KeyboardInterrupt:
            checkpoint = chat.history.create_checkpoint("interrupt")
            if checkpoint:
                print(f"\n{Colors.warning(f'Interrupted. Checkpoint saved: {checkpoint}')}")
            print(f"{Colors.success('Goodbye. Conversation saved.')}")
            break
        except Exception as e:
            checkpoint = chat.history.create_checkpoint("error")
            print(f"\n{Colors.error(f'Error handled without closing history: {e}')}")
            if checkpoint:
                print(f"{Colors.success(f'Safety checkpoint saved: {checkpoint}')}")
            print("You can continue, or use /recover list if something looks wrong.")


def print_startup_options(chat):
    """Show setup choices before entering the natural chat console."""
    exp_info = MODEL_CATALOG_INSTANCE.get_model_by_id(EXPENSIVE_MODEL_ID)
    cheap_info = MODEL_CATALOG_INSTANCE.get_model_by_id(CHEAP_MODEL_ID)
    exp_name = exp_info["display_name"] if exp_info else EXPENSIVE_MODEL_ID
    cheap_name = cheap_info["display_name"] if cheap_info else CHEAP_MODEL_ID
    past_chat_count = len(chat.history.list_checkpoints())

    print("\n" + "=" * 72)
    print("Startup Setup")
    print("=" * 72)
    print(f"1. Start natural chat")
    print(f"2. Change main chat model        Current: {exp_name[:38]}")
    print(f"3. Change summary/memory model   Current: {cheap_name[:38]}")
    print(f"4. Full settings")
    print(f"5. Model catalog")
    print(f"6. View recent conversation")
    print(f"7. Continue past chat            Available: {past_chat_count}")
    print(f"8. Export conversation")
    print(f"9. Context mode                 Current: {CONTEXT_MODE}")
    print(f"10. Streaming                   Current: {'on' if STREAM_RESPONSES else 'off'}")
    print(f"11. Old full menu")
    print(f"0. Exit")
    print("=" * 72)


def choose_startup_model(chat, target: str):
    """Change a startup model and persist settings."""
    result = chat.settings_manager.choose_model(target)
    if not result:
        print("Model change cancelled.")
        return

    key, model_id = result
    if target == "main":
        chat.settings_manager.settings["expensive_model_key"] = key
        chat.settings_manager.settings["expensive_model_id"] = model_id
    else:
        chat.settings_manager.settings["cheap_model_key"] = key
        chat.settings_manager.settings["cheap_model_id"] = model_id

    chat.settings_manager._apply_globals()
    chat.settings_manager.save()
    chat.reinitialize_client()
    print(f"Changed {target} model to {MODEL_CATALOG[key]['display_name']}")


def print_past_chats(chat) -> list[dict]:
    """Print checkpoint-backed past chats and return the displayed list."""
    chats = chat.history.list_past_chats()
    print("\nPast Chats")
    print("-" * 72)
    if not chats:
        print("No past chats found yet. They are saved automatically as checkpoints.")
        return []

    for i, item in enumerate(chats, 1):
        updated = item["last_updated"][:19].replace("T", " ") if item["last_updated"] else "unknown time"
        print(f"{i}. {updated} | {item['messages']} messages | {item['name']}")
        print(f"   {item['snippet']}")
    return chats


def startup_past_chats_menu(chat) -> bool:
    """Let the user continue a previous checkpoint-backed chat."""
    while True:
        chats = print_past_chats(chat)
        print("\nType a number to continue that chat, S to save current chat, or Enter to go back.")
        choice = safe_input("Past chat choice: ").strip()

        if not choice:
            return False
        if choice.lower() == "s":
            name = safe_input("Save current chat as: ").strip() or "saved_chat"
            checkpoint = chat.history.create_checkpoint(name)
            print(f"Saved: {checkpoint}" if checkpoint else "No messages to save yet.")
            continue
        if not choice.isdigit():
            print("Please enter a chat number.")
            continue

        index = int(choice) - 1
        if not 0 <= index < len(chats):
            print("Invalid chat number.")
            continue

        if chat.history.messages:
            chat.history.create_checkpoint("before_switching_chat")

        if chat.history.restore_checkpoint(chats[index]["path"]):
            chat._last_checkpoint_msg_count = len(chat.history.messages)
            print(f"Loaded past chat: {chats[index]['name']}")
            return True
        print("Could not load that chat.")


def startup_recovery_menu(chat):
    """List and restore checkpoints before starting chat."""
    while True:
        checkpoints = chat.history.list_checkpoints()
        print("\nRecovery Checkpoints")
        print("-" * 72)
        if not checkpoints:
            print("No checkpoints found.")
        else:
            for i, checkpoint in enumerate(checkpoints, 1):
                try:
                    stamp = datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except OSError:
                    stamp = "unknown time"
                print(f"{i}. {checkpoint.name} ({stamp})")
        print("\nType a number to restore, or press Enter to go back.")

        choice = safe_input("Recover choice: ").strip()
        if not choice:
            return
        if not choice.isdigit():
            print("Please enter a checkpoint number.")
            continue
        index = int(choice) - 1
        if not 0 <= index < len(checkpoints):
            print("Invalid checkpoint number.")
            continue

        confirm = safe_input(f"Restore {checkpoints[index].name}? Type RESTORE to confirm: ").strip()
        if confirm == "RESTORE":
            if chat.history.restore_checkpoint(checkpoints[index]):
                chat._last_checkpoint_msg_count = len(chat.history.messages)
                print("Checkpoint restored.")
                return
            print("Restore failed.")
        else:
            print("Restore cancelled.")


def run_startup_setup(chat) -> bool:
    """Run startup options. Return True when chat should start."""
    while True:
        print_startup_options(chat)
        choice = safe_input("Choose setup option: ").strip().lower()

        if choice in ("", "1", "start", "chat"):
            return True
        if choice in ("2", "main"):
            choose_startup_model(chat, "main")
        elif choice in ("3", "summary", "memory"):
            choose_startup_model(chat, "summary")
        elif choice in ("4", "settings"):
            chat.settings_manager.interactive_settings_menu()
            chat.reinitialize_client()
        elif choice in ("5", "catalog", "models"):
            print("\nFilter: 1 all, 2 budget, 3 mid, 4 premium")
            filter_choice = safe_input("Filter choice: ").strip()
            filter_map = {"2": "budget", "3": "mid", "4": "premium"}
            chat.settings_manager.display_model_catalog(filter_map.get(filter_choice))
        elif choice in ("6", "recent"):
            count = safe_input("How many messages? (default 8): ").strip()
            try:
                count_value = int(count) if count else 8
            except ValueError:
                count_value = 8
            chat.view_recent(max(1, min(count_value, 50)))
        elif choice in ("7", "past", "chats", "continue", "recover", "recovery"):
            if startup_past_chats_menu(chat):
                return True
        elif choice in ("8", "export"):
            fmt = safe_input("Export format: 1 markdown, 2 text: ").strip()
            filepath = chat.history.export("text" if fmt == "2" else "markdown")
            print(f"Exported to: {filepath}" if filepath else "Nothing exported.")
        elif choice in ("9", "context"):
            print("\nContext modes:")
            print("1. smart - summarize older messages with the cheap model")
            print("2. full  - continue with full conversation, no summaries")
            mode_choice = safe_input("Choose context mode: ").strip().lower()
            if mode_choice in ("1", "smart", "summary"):
                chat.settings_manager.settings["context_mode"] = "smart"
                chat.settings_manager._apply_globals()
                chat.settings_manager.save()
                print("Context mode set to smart summaries.")
            elif mode_choice in ("2", "full", "pure"):
                chat.settings_manager.settings["context_mode"] = "full"
                chat.settings_manager._apply_globals()
                chat.settings_manager.save()
                print("Context mode set to full pure conversation.")
            else:
                print("Context mode unchanged.")
        elif choice in ("10", "stream", "streaming"):
            chat.settings_manager.settings["stream_responses"] = not STREAM_RESPONSES
            chat.settings_manager._apply_globals()
            chat.settings_manager.save()
            print(f"Streaming is now {'on' if STREAM_RESPONSES else 'off'}.")
        elif choice in ("11", "menu", "old"):
            legacy_menu_main(chat)
            return False
        elif choice in ("0", "exit", "quit", "q"):
            checkpoint = chat.history.create_checkpoint("startup_exit")
            if checkpoint:
                print(f"Exit checkpoint saved: {checkpoint}")
            print("Goodbye. Conversation saved.")
            return False
        else:
            print("Unknown option. Choose a number from the startup setup.")



def main():
    """Main application entry point with bulletproof error handling."""
    print_banner()

    # Create the chat instance
    chat = BedrockChat()

    # Initialize (connect to AWS, load history)
    if not chat.initialize():
        print("\n❌ Failed to initialize. Please check your AWS configuration.")
        print("   Run 'aws configure' to set up your credentials.")
        print("\n   Press Enter to exit...")
        safe_input("")
        sys.exit(1)

    if run_startup_setup(chat):
        run_chat_console(chat)


def legacy_menu_main(chat):
    """Old menu loop retained for reference and future fallback."""
    while True:
        try:
            print_menu(chat)
            raw_input = safe_input("Enter choice (1-11) or type message: ")

            if not raw_input.strip():
                continue

            choice = normalize_choice(raw_input)

            # ─── Option 1: Send a message ────────────────────────────
            if choice == "1":
                user_message = get_user_input()

                if not user_message.strip():
                    print("❌ Empty message. Cancelled.")
                    continue

                print(f"\n{'─'*60}")
                response = chat.send_message(user_message)

                if response:
                    print(f"\n{'─'*60}")
                    print("🤖 Assistant:")
                    print(f"{'─'*60}")
                    print(response)
                    print(f"{'─'*60}")

            # ─── Option 2: View recent messages ──────────────────────
            elif choice == "2":
                count_str = safe_input("How many recent messages? (default 10): ")
                try:
                    count = int(count_str) if count_str.strip() else 10
                    count = max(1, min(count, 100))  # Clamp between 1 and 100
                except ValueError:
                    count = 10
                    print(f"   (Using default: {count})")

                chat.view_recent(count)

            # ─── Option 3: View full summary ─────────────────────────
            elif choice == "3":
                chat.view_full_summary()

            # ─── Option 4: Search conversation ───────────────────────
            elif choice == "4":
                query = safe_input("🔍 Search for: ")
                if query.strip():
                    chat.search_history(query.strip())
                else:
                    print("❌ Empty search query.")

            # ─── Option 5: File Management ───────────────────────────
            elif choice == "5":
                print(f"\n{'═'*60}")
                print("📁 FILE MANAGEMENT")
                print(f"{'═'*60}")
                print("\nWhat do you want to do?")
                print("  1. Upload a file")
                print("  2. List uploaded files")
                print("  3. Show file content")
                print("  4. Delete a file")
                print("  5. Back to main menu")

                sub_choice = safe_input("\nChoice: ").strip()

                if sub_choice == "1":
                    file_path = safe_input("Enter file path: ").strip()
                    description = safe_input("Enter description (optional): ").strip()
                    if file_path:
                        chat.upload_file(file_path, description)

                elif sub_choice == "2":
                    files = chat.file_manager.list_files()
                    if not files:
                        print("No files uploaded yet.")
                    else:
                        print("\n📁 Uploaded Files:")
                        for f in files:
                            print(f"  • {f['id']}: {f['original_name']} ({f['size']} bytes)")

                elif sub_choice == "3":
                    file_id = safe_input("Enter file ID: ").strip()
                    content = chat.file_manager.get_file_content(file_id, max_chars=1000)
                    if content:
                        print(f"\n📄 File Content (first 1000 chars):")
                        print(content)
                    else:
                        print("❌ File not found.")

                elif sub_choice == "4":
                    file_id = safe_input("Enter file ID to delete: ").strip()
                    chat.file_manager.delete_file(file_id)

            # ─── Option 6: View session costs ────────────────────────
            elif choice == "6":
                print(chat.tracker.get_summary())

            # ─── Option 7: Quick model switch ────────────────────────
            elif choice == "7":
                print(f"\n{'═'*60}")
                print("🔄 QUICK MODEL SWITCH")
                print(f"{'═'*60}")
                print("\nWhat do you want to change?")
                print("  1. Main model (for responses)")
                print("  2. Summary model (for cost savings)")
                print("  3. Both")
                print("  4. Cancel")

                sub_choice = safe_input("\nChoice: ").strip()

                if sub_choice in ("1", "3"):
                    result = chat.settings_manager.choose_model("main")
                    if result:
                        key, model_id = result
                        chat.settings_manager.settings["expensive_model_key"] = key
                        chat.settings_manager.settings["expensive_model_id"] = model_id
                        chat.settings_manager._apply_globals()
                        chat.settings_manager.save()
                        print(f"\n✅ Main model → {MODEL_CATALOG[key]['display_name']}")

                if sub_choice in ("2", "3"):
                    result = chat.settings_manager.choose_model("summary")
                    if result:
                        key, model_id = result
                        chat.settings_manager.settings["cheap_model_key"] = key
                        chat.settings_manager.settings["cheap_model_id"] = model_id
                        chat.settings_manager._apply_globals()
                        chat.settings_manager.save()
                        print(f"\n✅ Summary model → {MODEL_CATALOG[key]['display_name']}")

                if sub_choice == "4":
                    print("Cancelled.")

            # ─── Option 8: Full settings configuration ───────────────
            elif choice == "8":
                chat.settings_manager.interactive_settings_menu()
                # Reinitialize client if region changed
                chat.reinitialize_client()

            # ─── Option 9: View model catalog ────────────────────────
            elif choice == "9":
                print("\n   Filter by category?")
                print("   1. All models")
                print("   2. Budget only (💚 cheapest)")
                print("   3. Mid-tier only (💛)")
                print("   4. Premium only (💎 best quality)")

                filter_choice = safe_input("   Choice (default: all): ").strip()
                filter_map = {"2": "budget", "3": "mid", "4": "premium"}
                chat.settings_manager.display_model_catalog(filter_map.get(filter_choice))

            # ─── Option 10: Clear history ─────────────────────────────
            elif choice == "10":
                confirm = safe_input("⚠️  Delete ALL chat history? This cannot be undone! (yes/no): ")
                if confirm.strip().lower() in ("yes", "y"):
                    # Double confirm for safety
                    confirm2 = safe_input("   Type 'DELETE' to confirm: ")
                    if confirm2.strip() == "DELETE":
                        chat.history.clear()
                    else:
                        print("   Cancelled (didn't type DELETE).")
                else:
                    print("   Cancelled.")

            # ─── Option 11: Exit ──────────────────────────────────────
            elif choice == "11":
                print(chat.tracker.get_summary())
                chat.settings_manager.save()
                print("\n👋 Goodbye! Your conversation and settings have been saved.")
                print(f"   History: {HISTORY_FILE}")
                print(f"   Settings: {SETTINGS_FILE}")
                break

            # ─── Direct message (user typed a message instead of menu choice) ───
            elif choice == "direct_message":
                print(f"\n💡 Detected direct message. Sending to AI...")
                print(f"{'─'*60}")
                response = chat.send_message(raw_input.strip())

                if response:
                    print(f"\n{'─'*60}")
                    print("🤖 Assistant:")
                    print(f"{'─'*60}")
                    print(response)
                    print(f"{'─'*60}")

            # ─── Unknown input handling ───────────────────────────────
            else:
                # If it's somewhat long, ask if they want to send it as a message
                if len(raw_input.strip()) > 3:
                    print(f"\n💡 '{raw_input.strip()[:30]}...' doesn't match a menu option.")
                    send_it = safe_input("   Send it as a chat message? (y/n): ").strip().lower()

                    if send_it in ("y", "yes", ""):
                        response = chat.send_message(raw_input.strip())
                        if response:
                            print(f"\n{'─'*60}")
                            print("🤖 Assistant:")
                            print(f"{'─'*60}")
                            print(response)
                            print(f"{'─'*60}")
                else:
                    print("❌ Invalid choice. Enter 1-11 or type a message directly.")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! (Ctrl+C)")
            print("   Type '11' or 'exit' to quit properly and save everything.")
            continue

        except Exception as e:
            # BULLETPROOF: catch ANY error and keep running
            print(f"\n❌ An error occurred: {e}")
            print("   Don't worry - your history is saved. You can continue chatting.")
            print("   If this keeps happening, try option 8 (settings) to check your config.")

            # Log the error for debugging
            try:
                with open("bedrock_chat_errors.log", "a") as f:
                    f.write(f"\n[{datetime.datetime.now().isoformat()}] Error: {e}\n")
            except IOError:
                pass

            continue


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Chat interrupted. History has been saved. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("   Your conversation history should be saved in the JSON file.")
        print("   Please check your AWS configuration and try again.")

        # Log fatal error
        try:
            with open("bedrock_chat_errors.log", "a") as f:
                import traceback
                f.write(f"\n[{datetime.datetime.now().isoformat()}] FATAL: {e}\n")
                f.write(traceback.format_exc())
                f.write("\n")
        except IOError:
            pass
