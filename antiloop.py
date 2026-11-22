#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anti-loop -- defense-in-depth toolkit against LLM agent repetition loops.

The seven layers where repetition loops are born (full theory in README.md):

    L1 weights    the model never learned when to stop (e.g. pre-MOPD MiMo)
    L2 sampling   penalties disabled by default (DRY=0.0, repeat-penalty=1.0)
    L3 harness    tool results never enter the context -> invariant context
    L4 budget     reasoning eats max_tokens -> empty output -> harness retries
    L5 template   broken chat template -> thinking->tool_use transition fails
    L6 runtime    eval bugs, KV-cache degradation at long context
    L7 quant      aggressive quantization degrades logits

Core of the failure: a loop is a FIXED POINT of the generation dynamical
system.  When the context does not change (L3) and the token distribution is
peaked (L2), state[n+1] = f(state[n]) converges to re-emitting the exact same
tokens forever.

Commands:
  lint    audit configs (llama-swap.yaml / opencode.json) across the layers
  detect  find loop signatures in an OpenCode session database
  patch   add missing anti-repetition flags to a llama-swap.yaml (with backup)
  verify  run the tool-call loop test against a live OpenAI-compatible endpoint

Stdlib only.  Python >= 3.8.  MIT license.
"""

import argparse
import collections
import datetime
import difflib
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request

# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else str(text)

def ok(t):    return _c("32", "OK  ") + " " + t
def warn(t):  return _c("33", "WARN") + " " + t
def bad(t):   return _c("31", "FAIL") + " " + t
def info(t):  return _c("36", "info") + " " + t
def head(t):  return _c("1", t)

# --------------------------------------------------------------------------- #
# L2 / L5 / L6 / L7 checks over a llama-swap.yaml
# --------------------------------------------------------------------------- #

MODEL_RE = re.compile(r"^  ([A-Za-z0-9._-]+):\s*$")

def _flag(cmd, name):
    """Return float value of `--name <val>` in cmd, or None."""
    m = re.search(re.escape(name) + r"\s+(-?[0-9.]+)", cmd)
    return float(m.group(1)) if m else None

def _flag_raw(cmd, name):
    m = re.search(re.escape(name) + r"\s+(\S+)", cmd)
    return m.group(1) if m else None

def _path_flag(cmd, name):
    m = re.search(re.escape(name) + r"\s+(\S+)", cmd)
    return m.group(1) if m else None

def parse_llamaswap(path):
    """Return {model: cmd} for every model block with a cmd line."""
    models, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = MODEL_RE.match(line)
            if m:
                cur = m.group(1)
                continue
            if cur and line.startswith("    cmd: "):
                models[cur] = line[len("    cmd: "):].strip()
    return models

def lint_llamaswap(path, only=None, check_fs=True):
    models = parse_llamaswap(path)
    findings = []  # (model, layer, level, msg, fix)

    for name, cmd in sorted(models.items()):
        if only and name not in only:
            continue
        if "--embedding" in cmd:
            continue  # embeddings do not generate

        # --- L2: DRY -------------------------------------------------------
        dry = _flag(cmd, "--dry-multiplier")
        lastn = _flag(cmd, "--dry-penalty-last-n")
        if dry is None or dry == 0:
            findings.append((name, "L2", "FAIL",
                "DRY anti-repetition disabled (dry-multiplier missing/0)",
                "--dry-multiplier 0.8 --dry-penalty-last-n 128"))
        elif dry < 0.8:
            findings.append((name, "L2", "WARN",
                f"DRY weak (dry-multiplier={dry} < 0.8)",
                "raise to --dry-multiplier 0.8"))
        else:
            findings.append((name, "L2", "OK",
                f"DRY active (multiplier={dry}, last-n={lastn or 'default'})", None))
        if dry and lastn is None:
            findings.append((name, "L2", "WARN",
                "dry-penalty-last-n not set (default 64, recommended 128)",
                "--dry-penalty-last-n 128"))

        # --- L2: repeat penalty -------------------------------------------
        # Family guidance (documented): MiMo benefits from 1.05 (community
        # verified); for Qwen, repeat-penalty >1.0 interferes with llama.cpp
        # sampling -- DRY is the recommended defense there.
        is_mimo = name.lower().startswith("mimo")
        rp = _flag(cmd, "--repeat-penalty")
        if rp is None or rp <= 1.0:
            if is_mimo:
                findings.append((name, "L2", "FAIL",
                    "repeat-penalty missing or 1.0 (= disabled, llama.cpp "
                    "default) -- MiMo validated with 1.05",
                    "--repeat-penalty 1.05"))
            else:
                findings.append((name, "L2", "WARN",
                    "repeat-penalty disabled; fine for Qwen if DRY is active "
                    "(official Qwen guidance: repeat-penalty interferes with "
                    "llama.cpp sampling)",
                    "--repeat-penalty 1.05 (validated for MiMo only)"))
        elif rp > 1.1:
            findings.append((name, "L2", "WARN",
                f"repeat-penalty={rp} > 1.1 can break syntax/JSON tool calls",
                "lower to 1.05 and rely on DRY"))
        else:
            findings.append((name, "L2", "OK",
                f"repeat-penalty active ({rp})", None))

        # --- L2: temperature ----------------------------------------------
        temp = _flag(cmd, "--temp")
        if temp is None:
            findings.append((name, "L2", "WARN",
                "--temp not set (near-determinism depends on server default)",
                "--temp 0.5"))
        elif temp < 0.4:
            findings.append((name, "L2", "WARN",
                f"temp={temp} is near-deterministic: identical contexts "
                "reproduce identical outputs (fixed-point risk)",
                "--temp 0.5"))
        else:
            findings.append((name, "L2", "OK", f"temp={temp}", None))

        # --- L5: chat template --------------------------------------------
        tpl = _path_flag(cmd, "--chat-template-file")
        if tpl is None:
            findings.append((name, "L5", "WARN",
                "no --chat-template-file: embedded template may be the "
                "broken upstream one (thinking->tool_use transition bug)",
                "point --chat-template-file at a corrected template"))
        elif not check_fs:
            findings.append((name, "L5", "OK",
                f"template: {tpl} (existence not checked)", None))
        elif not os.path.exists(tpl) and not tpl.startswith("${"):
            findings.append((name, "L5", "WARN",
                f"template path not found from this machine: {tpl} -- ok if "
                "this config belongs to a remote host (re-run lint there, "
                "or pass --no-fs)", "verify the path on the serving host"))
        else:
            findings.append((name, "L5", "OK", f"template: {tpl}", None))

        # --- L6: KV cache quantization -------------------------------------
        ck = _flag_raw(cmd, "--cache-type-k")
        if ck and ck.startswith("q4"):
            findings.append((name, "L6", "WARN",
                f"KV cache k={ck}: degrades quality at long context "
                "(contributing factor to loops)", "--cache-type-k q8_0"))
        else:
            findings.append((name, "L6", "OK",
                f"KV cache k={ck or 'f16(default)'}", None))

        # --- L7: weight quantization ---------------------------------------
        mm = re.search(r"-m\s+(\S+)", cmd)
        if mm:
            fn = os.path.basename(mm.group(1))
            q = re.search(r"(Q[0-9]_[A-Z0-9]+|IQ[0-9]_[A-Z0-9]+|F16|BF16|f16)", fn, re.I)
            if q:
                qq = q.group(1).upper()
                if qq.startswith(("Q3", "IQ2", "IQ3", "Q2")):
                    findings.append((name, "L7", "WARN",
                        f"aggressive quant {qq}: community reports looping "
                        "symptoms from degraded logits", "use Q6_K/Q8_0"))
                elif qq.startswith(("Q4", "Q5")):
                    findings.append((name, "L7", "OK",
                        f"quant {qq} (acceptable; Q6_K+ preferred)", None))
                else:
                    findings.append((name, "L7", "OK", f"quant {qq}", None))

    return findings

# --------------------------------------------------------------------------- #
# L4 checks over an opencode.json
# --------------------------------------------------------------------------- #

def lint_opencode(path, only=None):
    with open(path, encoding="utf-8", errors="replace") as fh:
        cfg = json.load(fh)
    findings = []
    providers = cfg.get("providers") or cfg.get("provider") or {}

    all_names = []
    for pid, pv in providers.items():
        for mid in (pv.get("models") or {}):
            all_names.append((pid, mid))

    for pid, pv in providers.items():
        for mid, m in (pv.get("models") or {}).items():
            if only and mid not in only:
                continue
            if not isinstance(m, dict):
                continue
            limit = m.get("limit") or {}
            out = limit.get("output")
            ctx = limit.get("context")

            # --- L4: output budget (reasoning must not starve the answer) ---
            if out is None:
                findings.append((mid, "L4", "WARN",
                    f"[{pid}] limit.output not set: reasoning models may "
                    "return empty visible answers (thinking eats budget)",
                    '"limit": {"output": 16384} or higher'))
            elif out < 8192:
                findings.append((mid, "L4", "FAIL",
                    f"[{pid}] limit.output={out} << 8192: thinking-heavy "
                    "replies get cut off / retried", "raise to >=16384"))
            elif out < 16384:
                findings.append((mid, "L4", "WARN",
                    f"[{pid}] limit.output={out}: tight for reasoning models",
                    "raise to >=16384"))
            else:
                findings.append((mid, "L4", "OK",
                    f"[{pid}] limit.output={out}", None))

            if ctx is None:
                findings.append((mid, "L4", "WARN",
                    f"[{pid}] limit.context not set", '"context": 65536'))

    # --- L3 hygiene: stale near-duplicate model entries ---------------------
    names = [mid for _, mid in all_names]
    for i, a in enumerate(names):
        close = difflib.get_close_matches(a, names[i + 1:], n=1, cutoff=0.85)
        for b in close:
            findings.append((a, "L3", "WARN",
                f"near-duplicate model entries '{a}' ~ '{b}': stale/typo "
                "entry will be selectable and is usually the BROKEN old one",
                f"remove one of them"))

    return findings

# --------------------------------------------------------------------------- #
# lint command
# --------------------------------------------------------------------------- #

def detect_kind(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        head_txt = fh.read(4096)
    if head_txt.lstrip().startswith("{"):
        return "opencode"
    return "llamaswap"

def cmd_lint(args):
    path = args.config
    kind = args.kind if args.kind != "auto" else detect_kind(path)
    if kind == "opencode":
        findings = lint_opencode(path, only=set(args.model) or None)
    else:
        findings = lint_llamaswap(path, only=set(args.model) or None,
                                  check_fs=not args.no_fs)

    print(head(f"anti-loop lint -- {path} ({kind})"))
    print()

    by_model = collections.OrderedDict()
    for model, layer, level, msg, fix in findings:
        by_model.setdefault(model, []).append((layer, level, msg, fix))

    fails = warns = 0
    for model, items in by_model.items():
        print(head(f"  {model}"))
        for layer, level, msg, fix in items:
            fn = {"OK": ok, "WARN": warn, "FAIL": bad}[level]
            print(f"    [{layer}] {fn(msg)}")
            if level == "FAIL" and fix:
                print(f"           fix: --add  {fix}")
            elif level == "WARN" and fix:
                print(f"           fix:       {fix}")
            fails += level == "FAIL"
            warns += level == "WARN"
        print()

    print(f"summary: {fails} FAIL, {warns} WARN "
          f"({'all layers defended' if fails == 0 else 'DEFENSES MISSING'})")
    return 1 if fails else 0

# --------------------------------------------------------------------------- #
# detect command -- loop signatures in an OpenCode session DB
# --------------------------------------------------------------------------- #

def _msg_signature(data):
    """Stable signature of an assistant message: text + thinking + tool calls."""
    try:
        d = json.loads(data)
    except Exception:
        return None, None
    msg = d.get("message", d)
    texts, tools = [], []
    for part in (msg.get("content") or []):
        if not isinstance(part, dict):
            continue
        t = part.get("type")
        if t in ("text", "thinking", "reasoning"):
            body = part.get("text") or part.get("thinking") or part.get("reasoning") or ""
            texts.append(body)
        elif t == "tool":
            st = part.get("state") or {}
            tools.append({
                "tool": part.get("tool"),
                "input": st.get("input"),
            })
    blob = json.dumps({"t": texts, "c": tools}, sort_keys=True, ensure_ascii=False)
    sig = hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:12]
    return sig, "\n".join(texts)

def _ngram_rep(text, n=6, k=3):
    words = text.split()
    if len(words) < n * k:
        return 0, ""
    counter = collections.Counter(
        tuple(words[i:i + n]) for i in range(len(words) - n))
    gram, cnt = counter.most_common(1)[0]
    return cnt, " ".join(gram)

def cmd_detect(args):
    db = os.path.expanduser(args.db)
    if not os.path.exists(db):
        print(bad(f"database not found: {db}"), file=sys.stderr)
        return 2
    since_ms = int((__import__("time").time() - args.since * 3600) * 1000)

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT session_id, data, time_updated FROM session_message "
        "WHERE type='assistant' AND time_updated > ? "
        "ORDER BY time_updated", (since_ms,)).fetchall()
    con.close()

    print(head(f"anti-loop detect -- {db}"))
    print(info(f"{len(rows)} assistant messages in the last {args.since:g}h"))
    print()

    incidents = 0
    by_session = collections.OrderedDict()
    for ses, data, tu in rows:
        sig, text = _msg_signature(data)
        if sig is None:
            continue
        by_session.setdefault(ses, []).append((tu, sig, text or ""))

    for ses, msgs in by_session.items():
        # 1) identical consecutive messages (tool calls included)
        run, run_start = 1, 0
        for i in range(1, len(msgs) + 1):
            if i < len(msgs) and msgs[i][1] == msgs[i - 1][1]:
                if run == 1:
                    run_start = i - 1
                run += 1
            else:
                if run >= 3:
                    incidents += 1
                    ts = datetime.datetime.fromtimestamp(msgs[run_start][0] / 1000)
                    print(bad(
                        f"IDENTICAL LOOP  session={ses[:18]}...  "
                        f"{run} byte-identical messages @ {ts:%m-%d %H:%M}"))
                    preview = (msgs[run_start][2] or "[tool-only]").strip()
                    print(f"                 first: {preview[:90]!r}")
                run, run_start = 1, 0

        # 2) intra-message n-gram repetition (tables filtered out)
        for tu, sig, text in msgs:
            clean = "\n".join(
                ln for ln in text.splitlines() if not ln.strip().startswith("|"))
            cnt, gram = _ngram_rep(clean)
            if cnt >= 3:
                incidents += 1
                ts = datetime.datetime.fromtimestamp(tu / 1000)
                print(bad(
                    f"PHRASE LOOP     session={ses[:18]}...  "
                    f"6-gram x{cnt} @ {ts:%m-%d %H:%M}"))
                print(f"                 ... {gram[:80]!r}")

    print()
    if incidents:
        print(f"summary: {incidents} incident(s) -- LOOP SIGNATURES FOUND")
        return 1
    print("summary: clean -- no loop signatures")
    return 0

# --------------------------------------------------------------------------- #
# patch command -- add missing anti-repetition flags to llama-swap.yaml
# --------------------------------------------------------------------------- #

def cmd_patch(args):
    path = args.config
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    cur = None
    changes = []  # (model, [flags added])
    out = []
    for line in lines:
        m = MODEL_RE.match(line)
        if m:
            cur = m.group(1)
        if (cur and line.startswith("    cmd: ")
                and "--embedding" not in line
                and (not args.model or cur in args.model)):
            added = []
            # DRY: safe for every family
            for flag, text in (("--dry-multiplier", "--dry-multiplier 0.8"),
                               ("--dry-penalty-last-n",
                                "--dry-penalty-last-n 128")):
                if flag not in line:
                    added.append(text)
            # repeat-penalty: MiMo-only by default (Qwen guidance: DRY instead)
            want_rp = ("mimo" in cur.lower()) or args.with_repeat_penalty
            if want_rp and "--repeat-penalty" not in line:
                added.append("--repeat-penalty 1.05")
            # never lower an existing value: only fill gaps
            if added:
                line = line.rstrip("\n") + " " + " ".join(added) + "\n"
                changes.append((cur, added))
        out.append(line)

    if not changes:
        print(ok("nothing to patch -- every generative model already has "
                 "the L2 anti-repetition flags"))
        return 0

    print(head(f"anti-loop patch -- {path}"))
    for model, added in changes:
        print(f"  {model}: + {' '.join(added)}")
    print()

    if args.dry_run:
        print(info("dry run -- nothing written"))
        return 0

    if not args.yes:
        print(bad("refusing to write without --yes (backup is taken anyway)"))
        return 2

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = f"{path}.bak-antiloop-{ts}"
    shutil.copy2(path, backup)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    print(ok(f"patched {len(changes)} model(s) -- backup: {backup}"))
    print(info("restart llama-swap to reload (kill <pid>; systemd respawns)"))
    return 0

# --------------------------------------------------------------------------- #
# verify command -- the tool-call loop test harness
# --------------------------------------------------------------------------- #

TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Execute a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]

PROMPT = ("Create a Python script that counts the models in the local model "
          "configuration and writes the result to /tmp/count.md. Use the "
          "execute tool to run it.")

def _post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def cmd_verify(args):
    url = args.url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url + "/v1/chat/completions"

    messages = [{"role": "user", "content": PROMPT}]
    sigs, commands = [], []
    verdict_loop = False
    tool_null_ok = True

    print(head(f"anti-loop verify -- {url} model={args.model}"))
    print(info("condition: tool results returned as null (the incident "
               "scenario: executed=False, output=None)"))
    print()

    for i in range(1, args.iterations + 1):
        payload = {
            "model": args.model,
            "messages": messages,
            "tools": TOOLS,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": False,
        }
        try:
            data = _post(url, payload, args.timeout)
        except urllib.error.HTTPError as e:
            if e.code == 400 and tool_null_ok and any(
                    m.get("role") == "tool" for m in messages):
                # server rejects null tool content: fall back to "null"
                tool_null_ok = False
                for m in messages:
                    if m.get("role") == "tool" and m.get("content") is None:
                        m["content"] = "null"
                continue
            print(bad(f"HTTP {e.code}: {e.read()[:300]}"))
            return 2
        except Exception as e:
            print(bad(f"request failed: {e}"))
            return 2

        if "error" in data:
            print(bad(f"api error: {data['error']}"))
            return 2

        choice = data["choices"][0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        sig = hashlib.sha256(json.dumps(
            {"c": content, "t": tool_calls}, sort_keys=True,
            ensure_ascii=False).encode()).hexdigest()[:12]
        sigs.append(sig)

        preview = ""
        if tool_calls:
            fn = tool_calls[0].get("function") or {}
            args_ = fn.get("arguments") or "{}"
            try:
                preview = (json.loads(args_) or {}).get("command", "")
            except Exception:
                preview = str(args_)
            preview = preview.replace("\n", " ")[:70]
        print(f"  [{i:2d}] finish={finish} hash={sig} "
              f"ntools={len(tool_calls)} run={_current_run(sigs)}"
              + (f"  cmd={preview!r}" if preview else ""))

        # append assistant turn + null tool results (incident condition)
        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": None if tool_null_ok else "null",
            })

        if finish == "stop" and not tool_calls:
            print(info("model ended its turn (healthy stop)"))
            break

    max_run, run = 1, 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    print()
    print(f"iterations: {len(sigs)} | max consecutive identical: {max_run}")
    if max_run >= 3:
        print(bad("VEREDICT: LOOP"))
        verdict_loop = True
    elif max_run == 2:
        print(warn("VEREDICT: borderline (one duplicate pair) -- watch it"))
    else:
        print(ok("VEREDICT: SIN LOOP"))
    return 1 if verdict_loop else 0

def _current_run(sigs):
    if not sigs:
        return 0
    run = 1
    for i in range(len(sigs) - 1, 0, -1):
        if sigs[i] == sigs[i - 1]:
            run += 1
        else:
            break
    return run

# --------------------------------------------------------------------------- #
# journal command -- cross-check llama-swap logs for loop evidence
# --------------------------------------------------------------------------- #

def cmd_journal(args):
    cmd = ["journalctl", "-u", args.unit, "--no-pager",
           "--since", f"-{args.since}h"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30).stdout
    except FileNotFoundError:
        print(bad("journalctl not available on this OS (Linux only)"))
        return 2
    except subprocess.TimeoutExpired:
        print(bad("journalctl timed out"))
        return 2

    disc = [l for l in out.splitlines()
            if "upstream disconnection" in l or "no valid JSON data" in l]
    posts = [l for l in out.splitlines() if "chat/completions" in l]

    print(head(f"anti-loop journal -- unit={args.unit} last {args.since}h"))
    print(f"  chat/completions events : {len(posts)}")
    print(f"  stream disconnections   : {len(disc)}")
    if disc:
        print(warn("stream disconnections are the typical END of a loop "
                   "(client abort / crash):"))
        for l in disc[-5:]:
            print("    " + l.strip()[:120])
    else:
        print(ok("no loop-end signatures in the window"))
    return 1 if disc else 0

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(
        prog="antiloop",
        description="Defense-in-depth against LLM agent repetition loops "
                    "(7-layer model).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("lint", help="audit configs across the 7 layers")
    pl.add_argument("config", help="path to llama-swap.yaml or opencode.json")
    pl.add_argument("--kind", choices=["auto", "llamaswap", "opencode"],
                    default="auto")
    pl.add_argument("--model", action="append", default=[],
                    help="only check these model ids")
    pl.add_argument("--no-fs", action="store_true",
                    help="skip template/file existence checks (remote configs)")
    pl.set_defaults(func=cmd_lint)

    pd = sub.add_parser("detect", help="find loop signatures in opencode.db")
    pd.add_argument("--db", default="~/.local/share/opencode/opencode.db")
    pd.add_argument("--since", type=float, default=48, help="hours back")
    pd.set_defaults(func=cmd_detect)

    pp = sub.add_parser("patch", help="add missing L2 flags to llama-swap.yaml")
    pp.add_argument("config")
    pp.add_argument("--model", action="append", default=[])
    pp.add_argument("--with-repeat-penalty", action="store_true",
                    help="also add --repeat-penalty to non-MiMo models "
                         "(default: MiMo only; Qwen prefers DRY)")
    pp.add_argument("--dry-run", action="store_true")
    pp.add_argument("--yes", action="store_true", help="actually write")
    pp.set_defaults(func=cmd_patch)

    pv = sub.add_parser("verify", help="tool-call loop test vs live endpoint")
    pv.add_argument("--url", default="http://127.0.0.1:8080")
    pv.add_argument("--model", required=True)
    pv.add_argument("--iterations", type=int, default=12)
    pv.add_argument("--max-tokens", type=int, default=1500)
    pv.add_argument("--temperature", type=float, default=0.5)
    pv.add_argument("--timeout", type=float, default=180)
    pv.set_defaults(func=cmd_verify)

    pj = sub.add_parser("journal", help="cross-check service logs (Linux)")
    pj.add_argument("--unit", default="llama-swap")
    pj.add_argument("--since", type=float, default=24, help="hours back")
    pj.set_defaults(func=cmd_journal)

    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130

if __name__ == "__main__":
    sys.exit(main())
