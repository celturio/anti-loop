# anti-loop

**Defense-in-depth against repetition loops in local LLM agents.**

A loop — the agent re-emitting the same text or the same tool call forever —
is not one bug. It is the *emergent* failure of seven independent layers that
are all permissive by default. This repo gives you:

1. a **model of the failure** (the 7 layers + the fixed-point core),
2. a **fix matrix** with exact flags, verified in production,
3. a **toolkit** (`antiloop.py`, stdlib-only) to `lint`, `detect`, `patch`
   and `verify` your own deployment.

> Case study: MiMo-V2.6-Distill-9B serving on a single RTX 3060 12 GB through
> llama-swap + OpenCode. 6 byte-identical assistant messages in production →
> zero loops after the full stack was applied (evidence below).

---

## The failure in one picture

```
   L1 WEIGHTS      the model never learned WHEN TO STOP
                   (pre-MOPD MiMo: P(end-of-turn) = 5% at tool-call #12)
        +
   L2 SAMPLING     penalties are OFF by default
                   (llama.cpp: repeat-penalty 1.0 = disabled,
                                dry-multiplier  0.0 = disabled)
        +
   L3 HARNESS      the tool result never enters the context
                   (executed=False, output=None → context is INVARIANT)
        +
   L4 BUDGET       reasoning eats max_tokens → visible answer is empty
                   → the harness retries → apparent "repetitions"
        +
   L5 TEMPLATE     broken chat template → thinking→tool_use transition fails
                   → the model dumps its thinking as text, then repeats it
        +
   L6 RUNTIME      eval bugs, KV-cache degradation at long context
        +
   L7 QUANT        aggressive quantization degrades logits
```

### The math core: a loop is a fixed point

Generation is a dynamical system: `stateₙ₊₁ = f(stateₙ)` where `f` is
(model + sampler) over a context.

* If the **context does not change** between iterations (L3) and
* the **token distribution is peaked** (L2: low temperature, no penalties),

then `f` has a **fixed point inside the previously generated sequence**:
the sampler re-emits exactly the same tokens. Byte-identical messages,
one every few seconds, until something external breaks the cycle.

Every defense below works by breaking one of those two conditions:

| Break | Mechanism |
|---|---|
| peaked distribution | DRY / repeat-penalty / higher temperature |
| invariant context | make tool results actually enter the context |
| never-stops weights | MOPD-style fine-tuning (or an anti-loop LoRA) |
| starved budget | raise `limit.output` so thinking never eats the answer |

---

## The 7 layers and the fix matrix

| Layer | Variable | Default state | Defense | Exact fix |
|---|---|---|---|---|
| **L1** weights | model doesn't know when to stop | pre-MOPD checkpoints | retrain / distill a stop signal | Xiaomi **MOPD** recipe; or anti-loop **LoRA/QLoRA** (reward=0 on repetition) |
| **L2** sampling | anti-repetition OFF | `repeat-penalty=1.0`, `DRY=0.0` | turn both on, keep temp ≥ 0.4 | `--dry-multiplier 0.8 --dry-penalty-last-n 128 --repeat-penalty 1.05 --temp 0.5` |
| **L3** harness | tool output not in context | `executed=False` loops | execute tools for real; cap & dedupe | per-step tool cap + identical `{tool,input}` hash guard + doom-loop detector |
| **L4** budget | thinking eats the answer | `output ≤ 4096` | give reasoning room | `"limit": {"output": 16384…32768}` in opencode.json |
| **L5** template | upstream template broken | official MiMo jinja | use a corrected template | `--chat-template-file` → corrected `.jinja` (verify md5) |
| **L6** runtime | KV degradation / eval bugs | `q4_0` KV at 41K ctx | q8_0 KV + current llama.cpp | `--cache-type-k q8_0 --cache-type-v q8_0` |
| **L7** quant | degraded logits | Q3/IQ2 quants loop | Q6_K minimum | re-quantize to `Q6_K`/`Q8_0` |

**Precedence:** L1 is the only true fix (the model itself stops). Everything
else is *emulation* of the stop signal at the layers you control. For small
distills where no fixed checkpoint exists (e.g. MiMo-V2.6-Distill-9B has
**no MOPD release**), the inference-side stack is the only option — and it
is sufficient when applied as a whole.

---

## Quick start

```bash
# 1. audit your configs across the layers
python3 antiloop.py lint /path/to/llama-swap.yaml
python3 antiloop.py lint ~/.config/opencode/opencode.json

# 2. find loop signatures in past sessions (OpenCode DB)
python3 antiloop.py detect --since 48

# 3. patch missing L2 flags into llama-swap.yaml (backup is taken)
python3 antiloop.py patch /path/to/llama-swap.yaml --dry-run
python3 antiloop.py patch /path/to/llama-swap.yaml --yes

# 4. validate under the exact incident condition
#    (tool results returned as null → no new context → fixed-point risk)
python3 antiloop.py verify --url http://m002:8080 --model mimo-heretic

# 5. cross-check service logs for loop-end signatures (Linux)
python3 antiloop.py journal --unit llama-swap --since 24
```

Exit codes: `0` clean / defended, `1` problems found, `2` usage or runtime
error — so you can drop `lint` and `detect` into CI.

No dependencies. Python ≥ 3.8.

---

## Case study — MiMo-V2.6-Distill-9B on RTX 3060 12 GB

### The incident (2026-11-10, measured, not assumed)

* 6 assistant messages **byte-identical** (reasoning + text + same tool call),
  one every ~6.5 s, in an OpenCode session.
* Each: `finish=tool-calls`, tool `executed=False`, `output=None`
  → **the result never entered the context** (L3).
* `repeat-penalty` absent everywhere (= disabled), `dry-multiplier` absent
  (= disabled), `temp 0.3` (near-deterministic) (L2).
* Journal: 7 POSTs → 200 OK with identical response sizes.

### What was applied (all 7 layers)

```
L1  mimo-qlora anti-loop QLoRA (local stand-in for MOPD)
L2  --dry-multiplier 0.8 --dry-penalty-last-n 128
    --repeat-penalty 1.05            ← community-verified for this model
    --temp 0.5 --top-p 0.8 --top-k 20
L3  tool results guaranteed to enter the context (harness side)
L4  limit.output 32768 on every mimo entry (= the reference cloud model)
L5  corrected chat template (md5-verified against the community fix)
L6  --cache-type-k/v q8_0, current llama.cpp
L7  Q8_0 (17.9 GB) and Q6_K (6.9 GB) only
```

### Validation (test harness, incident-exact condition)

```
iterations: 13 | max consecutive identical: 1 | VEREDICT: SIN LOOP
```

13 distinct tool-call hashes, 0 warnings/disconnects in the journal during
the run — vs 6 byte-identical messages before the fix.

Reproduce it:

```bash
python3 antiloop.py verify --url http://127.0.0.1:8080 --model mimo-heretic
```

---

## Why does OpenCode show the highest loop rate?

Xiaomi's own measurements (blog below), response-level repetition rate by
harness for the pre-fix checkpoints:

| Harness | MiMo-V2.6-Flash-RL | MiMo-V2.6-Pro-RL |
|---|---|---|
| **OpenCode** | **1.02 %** | **0.54 %** |
| Claude Code | 0.27 % | 0.10 % |
| Codex | 0.23 % | 0.07 % |
| … | … | … |

OpenCode maximizes L3 (rich tool surface, results sometimes dropped) and L4
(reasoning-heavy turns) — exactly the two layers that need the most care.

---

## Sources

* Xiaomi MiMo — *Diagnosing and Mitigating Tool-Call Repetition in MiMo-V2.6*
  (MOPD fix, per-harness rates): <https://mimo.xiaomi.com/blog/mimo-v2-6-tool-call-repetition>
* XiaomiMiMo/MiMo issue #59 — *infinite text repetition loop when tool_use
  transition fails*: <https://github.com/XiaomiMiMo/MiMo/issues/59>
* oh-my-pi issue #12784 — *sampler falls into a repetition attractor;
  repetition_penalty is the missing ingredient*:
  <https://github.com/can1357/oh-my-pi/issues/12784>
* anomalyco/opencode issue #50678 — *no per-step tool cap, no identical-call
  dedupe, no doom-loop detector*:
  <https://github.com/anomalyco/opencode/issues/50678>
* llama.cpp — `--repeat-penalty` default **1.00 = disabled**,
  `--dry-multiplier` default **0.00 = off** (server README).
* Production evidence & test harness: Ecosistema Mazingers vault,
  `GOLDEN_FIX_LoopRepeticion_MiMo_OpenCode_2026-11-10.md`.

---

## License

MIT — see [LICENSE](LICENSE).
