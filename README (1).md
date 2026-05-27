<div align="center">

# 🤖 Bedrock Chat

**Private, offline-first AI workspace for Amazon Bedrock**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-FF9900?logo=amazon-aws)](https://aws.amazon.com/bedrock/)

*All data stays on your machine. No external APIs, no telemetry.*

</div>

![Bedrock Chat UI](https://via.placeholder.com/800x450?text=Bedrock+Chat+UI)

---

## Table of Contents

- [What is Bedrock Chat?](#what-is-bedrock-chat)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Web UI Guide](#web-ui-guide)
- [Terminal Chat Guide](#terminal-chat-guide)
- [Configuration](#configuration)
- [Projects & Documents](#projects--documents)
- [Data Storage](#data-storage)
- [Troubleshooting](#troubleshooting)
- [Security Notes](#security-notes)
- [Contributing](#contributing)

---

## What is Bedrock Chat?

Bedrock Chat is a **local-first, private AI workspace** that connects to [Amazon Bedrock](https://aws.amazon.com/bedrock/) — AWS's fully managed service for foundation models. It gives you a clean web interface *and* a powerful terminal client to chat with any Bedrock-supported model (Claude, Nova, Llama, Mistral, DeepSeek, Kimi, Cohere, and more) without sending your data anywhere except AWS.

**Key philosophy:**
- 🔒 **Privacy-first** — All conversations, files, credentials, and settings live in a local `chat_history/` folder
- 💰 **Cost-conscious** — Smart context summarization, real-time cost tracking in INR/USD, and configurable cost guardrails
- 🏗️ **Project-oriented** — Organize chats into projects with shared documents and system instructions
- 🌐 **Offline-resilient** — Messages queue when connectivity drops and auto-send on reconnection

---

## Features

### Universal Model Support
Chat with **any model available in your AWS Bedrock console**:
- **Anthropic** — Claude 3.5 Sonnet, Claude 3.7 Sonnet, Claude 3 Opus, Claude 3.5 Haiku
- **Amazon** — Nova Pro, Nova Lite, Nova Micro, Titan Text
- **Meta** — Llama 3.1 (8B/70B), Llama 3.2 (90B), Llama 3.3 (70B)
- **Mistral AI** — Mistral Large, Mistral Small
- **Moonshot AI** — Kimi K2 Instruct, Kimi K2 Thinking
- **DeepSeek** — DeepSeek V3, DeepSeek R1
- **Cohere** — Command R+, Command R
- **AI21** — Jamba Large, Jamba Mini

The app dynamically fetches your account's available models via the Bedrock API and displays them with live pricing.

### Smart Context Management
Long conversations get expensive fast. Bedrock Chat automatically:
- **Summarizes older messages** using a cheap model (e.g., Nova Micro) to preserve context without inflating token usage
- Keeps the **most recent N messages** in full (configurable, default 30)
- Falls back gracefully if summarization fails
- Toggle between **Smart** (summarize) and **Full** (send everything) modes per conversation

### Real-Time Cost Tracking
Every message shows:
- Token count (↑ input / ↓ output)
- Estimated cost in **INR** or **USD**
- Per-session running total
- All-time usage statistics

A **cost guardrail** warns or blocks you if an estimated prompt would exceed your configured spending limit.

### File & Image Uploads
Attach files directly to messages:
- **Images** — Sent to vision-capable models (Claude) as base64
- **Text files** — Code, markdown, CSV, JSON, etc. are injected into the prompt
- **PDFs** — Text extracted and embedded (requires `PyPDF2`)

### Project-Based Organization
Group related chats into **projects**:
- Each project has a **system prompt** injected into every conversation
- **Documents** — Upload reusable text files that get injected as context
- **Selection Tray** — Highlight text in any message, click **+ Tray**, collect snippets across messages, then generate a structured summary and save it back as a project document
- Conversations inside a project share context automatically

### Terminal Chat Client (`AWS_Chat.py`)
A full-featured CLI companion with:
- Same smart context management and cost tracking as the web UI
- **Streaming responses** with real-time token display
- **Multi-line paste detection** — Paste any amount of text; it's auto-detected
- **Checkpoint system** — Auto-saves every N messages; recover from crashes or interruptions
- **Conversation branching** — Fork a chat from any message to explore alternate paths
- **Model catalog browser** with fuzzy search and typo-tolerant selection
- **Command system** — `/help`, `/file`, `/summary`, `/search`, `/models`, `/settings`, `/checkpoint`, `/recover`, and more

### Offline-First Design
- Messages **queue locally** when you lose connectivity
- Auto-retry with exponential backoff on transient errors
- Resume and send queued messages when the connection returns
- Drafts auto-save locally (browser `localStorage`) and server-side every 10 seconds

### Multi-Model Comparison
Send the same prompt to **2–3 models simultaneously** and compare responses side-by-side. Each response shows its own token count and cost.

### Prompt Templates
Save reusable prompt templates with `{{variable}}` placeholders. Apply a template and you'll be prompted to fill in the blanks before sending.

### Model Presets
Save and switch between complete model configurations (model + temperature + max tokens + system prompt) with one click.

### Backup & Restore
- **Download Backup** — Export the entire `chat_history/` folder as a ZIP
- **Restore Backup** — Import a ZIP to recover on a new machine or after reinstall

---

## Architecture

```
bedrock-chat/
├── server.py              # FastAPI backend (runs on localhost:8000)
├── index.html             # Single-page web UI (served by server.py)
├── AWS_Chat.py            # Standalone terminal chat client
└── chat_history/          # All local data (auto-created)
    ├── conversations/     # One JSON file per chat
    ├── projects/          # Project definitions + documents
    ├── uploads/           # Attached files
    ├── settings.json      # App configuration
    ├── credentials.json   # AWS keys (plain JSON — keep safe!)
    ├── stats.json         # Lifetime token usage & cost
    ├── templates.json     # Saved prompt templates
    ├── presets.json       # Saved model presets
    └── backups/           # Exported ZIP backups
```

**Runtime flow:**
1. `server.py` starts a FastAPI server on `localhost:8000`
2. Browser loads `index.html` → connects to the API
3. User enters AWS credentials → server tests Bedrock connectivity
4. Server fetches available models from the Bedrock API
5. All chat data is read/written as JSON files in `chat_history/`
6. `AWS_Chat.py` can run independently and uses the same data folder

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- An **AWS account** with [Bedrock access enabled](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)
- Valid **AWS credentials** (Access Key + Secret Key) or a pre-configured `~/.aws/credentials` file

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/bedrock-chat.git
cd bedrock-chat
pip install boto3 fastapi uvicorn

# Optional: PDF text extraction
pip install PyPDF2
```

### 2. Run the Web Server

```bash
python server.py
```

You'll see:
```
============================================================
  Bedrock Chat - Local Web UI
============================================================
  Local :  http://localhost:8000
  LAN   :  http://192.168.x.x:8000
  Data  :  /path/to/bedrock-chat/chat_history
  All chats, credentials, projects, and backups save here
  Press Ctrl+C to stop
============================================================
```

### 3. Open the App

Navigate to **http://localhost:8000** in your browser.

> ⚠️ **Important:** Always access via `http://localhost:8000`. Do **not** open `index.html` directly from the file system — the API will not work.

### 4. First-Time Setup

1. Click **⚙️ Settings** (top-right gear icon)
2. Go to the **AWS Credentials** tab
3. Enter your **Access Key ID** and **Secret Access Key**
4. (Optional) Enter a **Session Token** if using temporary STS/SSO credentials
5. Select your **AWS Region** (default: `us-east-1`)
6. Click **⚡ Test Connection** to verify
7. Click **💾 Save Credentials**
8. Start chatting!

---

## Web UI Guide

### Sidebar
- **🗂 Projects** — Expand to see project chats, create new projects, manage documents
- **💬 Conversations** — List of all standalone chats, pinned chats appear at the top
- **🔍 Search** — Search across all conversation titles and message content

### Chat Area
- **Top bar** — Shows chat title (click to rename), current model pill, session cost, and action buttons
- **Model dropdown** — Click the model pill to switch models mid-conversation; filter by provider or name
- **Message cards** — AI responses show model name, timestamp, token count, cost, and action buttons (Copy, Regenerate, Continue, Branch, Pin, Add Note, Delete)
- **Thinking blocks** — Models that output reasoning (Claude, DeepSeek, Kimi) show collapsible amber thinking sections

### Input Area
- **Textarea** — Auto-expands; `Enter` to send, `Shift+Enter` for a new line
- **📎 Attach** — Upload files or drag-and-drop onto the input box
- **📋 Templates** — Insert a saved prompt template
- **Cost preview** — Shows estimated tokens and cost before you send

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl + N` | New chat |
| `Ctrl + ,` | Open Settings |
| `Enter` | Send message |
| `Shift + Enter` | New line in message |

### Action Buttons (per message)
| Button | Action |
|--------|--------|
| **Copy** | Copy full response to clipboard |
| **Regenerate** | Retry the response (removes later messages) |
| **Continue** | Ask the model to continue from where it stopped |
| **🌿 Branch** | Fork the conversation from this message into a new chat |
| **Pin** | Save this response to the conversation's pinned notes |
| **+ Note** | Add a private annotation to this message |
| **🗑** | Delete this message |

### Selection Tray
1. Highlight any text in a message
2. A floating **+ Tray** button appears
3. Click it to add the snippet to your tray
4. Add manual notes in the tray panel
5. Choose a summary model and click **✨ Generate Summary**
6. Review the output, pick a project, name the document, and **💾 Save**

### Compare Mode
1. Click the **⚡ Compare** button in the top bar
2. Select 2–3 models from the chips
3. Type your prompt and click **Run Compare**
4. Responses appear side-by-side; click **Save this response to chat** to keep any of them

---

## Terminal Chat Guide

Run the standalone terminal client:

```bash
python AWS_Chat.py
```

### Startup Flow
1. **AWS Credentials** — Enter keys interactively (saved to `aws_credentials.json`)
2. **Test Connection** — Verifies Bedrock access and lists available models
3. **Setup Menu** — Choose models, region, context mode, streaming, or jump straight to chat

### Chatting
- **Normal mode** — Type a line, press `Enter` → sent immediately
- **Paste mode** — Paste any amount of text; auto-detected on Unix/macOS
- **Multi-line mode** — Type `/multi` or `/p`, enter your text, then type `END` on its own line
- **Line continuation** — End a line with `\` to continue on the next line

### Commands
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/p`, `/paste`, `/multi` | Enter multi-line input mode |
| `/file upload <path>` | Upload a file to the conversation |
| `/file list` | List uploaded files |
| `/file show <id>` | View file content |
| `/file delete <id>` | Remove a file |
| `/recent [n]` | Show last N messages |
| `/summary` | Generate and display conversation summary |
| `/search <text>` | Search message history |
| `/models [budget|mid|premium]` | Browse model catalog |
| `/model main` | Change the main response model |
| `/model summary` | Change the summarization model |
| `/settings` | Open full settings menu |
| `/context smart` or `/context full` | Toggle context mode |
| `/stream on` or `/stream off` | Toggle streaming |
| `/think on` or `/think off` | Toggle reasoning block visibility |
| `/checkpoint [note]` | Save a manual checkpoint |
| `/recover list` | List recoverable checkpoints |
| `/recover restore N` | Restore checkpoint number N |
| `/chats` | List past chats |
| `/chats continue N` | Restore and continue past chat N |
| `/chats save <name>` | Save current chat as named checkpoint |
| `/export [md|txt]` | Export conversation to file |
| `/cost`, `/stats` | Show session cost statistics |
| `/clear` | Clear conversation history |
| `/quit`, `/exit` | Exit and save checkpoint |

### Crash Recovery
If the terminal client crashes or your machine shuts down mid-response:
1. Restart `AWS_Chat.py`
2. It will detect the interrupted request and offer to:
   - **Continue/retry** the request now
   - **Save the partial answer** and continue later
   - **Discard** the recovery state

---

## Configuration

All settings are stored in `chat_history/settings.json`. You can edit them via the web UI Settings panel or by editing the JSON directly (while the server is stopped).

### Settings Reference

| Setting | Description | Default |
|---------|-------------|---------|
| `aws_region` | AWS region for Bedrock API calls | `us-east-1` |
| `main_model` | Default model for chat responses | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `summary_model` | Cheap model for context summarization | `amazon.nova-micro-v1:0` |
| `temperature` | Sampling temperature (0.0 deterministic → 1.0 creative) | `0.7` |
| `max_tokens` | Maximum output tokens per response | `8192` |
| `max_ctx_messages` | Number of recent messages kept in full context | `30` |
| `context_mode` | `smart` (summarize old) or `full` (keep everything) | `smart` |
| `cost_guard_enabled` | Enable spending warnings | `true` |
| `cost_guard_inr` | Warn/block if estimated cost exceeds this (INR) | `5.0` |
| `show_thinking` | Display model reasoning blocks | `true` |
| `system_prompt` | Base system instructions for all chats | *(see code)* |

### Credential Modes
The app supports three authentication methods:

1. **Access Keys** — Direct AWS Access Key ID + Secret Key (+ optional Session Token)
2. **AWS Profile** — Use a named profile from `~/.aws/credentials` or `~/.aws/config`
3. **Default Chain** — Fall through to environment variables → `~/.aws/credentials` → IAM/ECS role (nothing stored by the app)

---

## Projects & Documents

### Creating a Project
1. In the sidebar, click **+** next to **🗂 Projects**
2. Enter a **Project Name**
3. (Optional) Add **Instructions** — these are injected as the system prompt for every chat in this project
4. Click **Save Project**

### Starting a Project Chat
1. Expand the project in the sidebar
2. Click **＋ New chat in project**
3. Select which project documents to inject as context
4. Start chatting — the project instructions and selected documents are automatically included

### Managing Documents
- **Via Selection Tray** — Collect snippets from messages, generate a summary, and save it to any project
- **Via Library Panel** — Open the **Library** button in the top bar, select a project, and upload text files or create manual documents
- **In Project Settings** — Edit a project to see all documents, view token counts, and delete unused ones

### Updating Context Mid-Chat
Click the **⚙ Docs** button in the project banner at the top of a chat to add or remove injected documents without starting a new conversation.

---

## Data Storage

Everything lives in the `chat_history/` directory next to `server.py`:

| Path | Contents |
|------|----------|
| `conversations/*.json` | Individual chat histories |
| `projects/*.json` | Project definitions, instructions, and documents |
| `uploads/` | Files attached to messages |
| `settings.json` | App configuration |
| `credentials.json` | AWS credentials (plain JSON) |
| `stats.json` | Lifetime token usage and cost |
| `templates.json` | Saved prompt templates |
| `presets.json` | Saved model presets |
| `backups/` | Exported ZIP files |

> 🔐 **Security note:** `credentials.json` stores keys as plain text. Protect this file — do not commit it to version control. Add `chat_history/credentials.json` to your `.gitignore`.

### Backup
Click **Download Backup** in the Settings panel to export the entire `chat_history/` folder as a ZIP. To restore, click **Restore Backup** and select a previously exported ZIP.

---

## Troubleshooting

### "Cannot connect to server"
- Ensure `server.py` is running
- Use `http://localhost:8000`, not `file:///path/to/index.html`
- Check that port 8000 is free: `lsof -i :8000` (macOS/Linux) or `netstat -ano | findstr :8000` (Windows)

### "No credentials found" / Bedrock errors
- Verify your AWS credentials are valid and not expired
- Ensure your IAM user/role has `bedrock:InvokeModel` and `bedrock:ListFoundationModels` permissions
- Check that the selected region has Bedrock enabled for your account
- Use the **⚡ Test Connection** button in Settings for detailed diagnostics

### Model not appearing in dropdown
- The app fetches models dynamically from your AWS account. If a model is missing:
  - Verify it's enabled in the [Bedrock console](https://console.aws.amazon.com/bedrock)
  - Check that it's available in your selected region
  - Some models require [cross-region inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)

### Streaming doesn't work for some models
- Some Bedrock models (e.g., DeepSeek) don't support response streaming
- The UI automatically falls back to non-streaming mode
- You can disable streaming entirely in Settings if preferred

### Cost guard blocking legitimate prompts
- Increase `cost_guard_inr` in Settings, or
- Disable the guard entirely (not recommended for heavy use)

### Terminal client crashes on startup
- Ensure `boto3` is installed: `pip install boto3`
- Check that your AWS credentials are valid
- Look for `bedrock_chat_errors.log` in the project root for detailed error traces

---

## Security Notes

- **Local-only by design** — No data is sent to any server except AWS Bedrock
- **Credentials** — Stored as plain JSON in `chat_history/credentials.json`. Keep this directory secure
- **No telemetry** — No usage analytics, crash reports, or network calls to third parties
- **Backup before sharing** — If you share your project folder, remove `credentials.json` first
- **Session tokens** — If using temporary STS credentials, the session token is also stored locally

---

## Contributing

This is my first open-source project — all feedback, bug reports, and pull requests are welcome!

### Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/bedrock-chat.git
cd bedrock-chat
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

### Project Structure
- `server.py` — FastAPI backend, handles all API routes, AWS Bedrock streaming, file I/O
- `index.html` — Complete single-page web UI (HTML + CSS + vanilla JS, no build step)
- `AWS_Chat.py` — Standalone terminal client with its own model catalog, retry logic, and checkpoint system

### Ideas for Contributions
- [ ] Dark/light theme toggle
- [ ] Mobile-responsive layout improvements
- [ ] Additional export formats (PDF, DOCX)
- [ ] Voice input/output integration
- [ ] Plugin system for custom model providers
- [ ] Docker container for easy deployment

---

<div align="center">

**[⬆ Back to Top](#-bedrock-chat)**

Built with Python, FastAPI, and vanilla JavaScript.

</div>
