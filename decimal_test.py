#!/usr/bin/env python3
"""Decimal trap test — inspired by the Qwen3.8 Flash Next video lab (task 3):
float 2.675 stores 2.6749... so round-half-up must be done in Decimal from text.
Also checks the 'lazy fix' trap (int 3 -> 3.0) from task 1's config bug."""
import json, urllib.request, time

URL = "http://127.0.0.1:8080/v1/chat/completions"
QUESTIONS = [
    ("float-trap",
     "En Python, ¿que imprime round(2.675, 2)? Explica en 2 lineas por que no es 2.68."),
    ("spec-fix",
     "Especificacion: montos en dolares, redondear la mitad hacia arriba, asi que 2.675 DEBE quedar en 2.68. "
     "El codigo actual hace la suma en float y luego round(...,2) y da 2.67. "
     "Dame el arreglo exacto en Python para cumplir la spec SIEMPRE (desde la lectura del texto). Codigo + 1 linea de por que."),
    ("lazy-fix",
     "Tengo en un YAML: limit_retries: 3. Un bug de redondeo hizo que un colega sugiera 'convertir todos los valores "
     "a decimales' como arreglo. ¿Debo convertir limit_retries a 3.0? Responde si/no en 1 linea con la razon."),
    ("eps",
     "¿Por que 0.1 + 0.2 == 0.3 es False en Python? 2 lineas maximo."),
    ("exact-repr",
     "¿Que valor exacto (en decimal) tiene 2.675 guardado en float64 de doble precision? Responde con el numero y 1 linea."),
    ("loop-guard",
     "Si un agente AI repite el mismo tool call 20 veces seguidas sin que el contexto cambie, ¿cual es la causa mas "
     "probable y que 3 medidas la previenen? Lista numerada, maximo 6 lineas."),
]

def ask(model, q, max_tokens=1600):
    payload = {"model": model, "messages": [{"role": "user", "content": q}],
               "max_tokens": max_tokens, "temperature": 0.5, "stream": False}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    c = d["choices"][0]
    msg = c["message"]
    txt = (msg.get("content") or "").strip()
    if not txt:
        txt = "[thinking-only] " + (msg.get("reasoning") or msg.get("reasoning_content") or "")[:400]
    return txt, c["finish_reason"], time.time() - t0

results = {}
for model in ("mimo-q8-0", "mimo-heretic"):
    print(f"\n{'='*70}\n### MODELO: {model}\n{'='*70}", flush=True)
    results[model] = []
    for tag, q in QUESTIONS:
        try:
            ans, fin, dt = ask(model, q)
        except Exception as e:
            ans, fin, dt = f"ERROR: {e}", "-", 0
        results[model].append((tag, ans))
        print(f"\n[{tag}] ({fin}, {dt:.1f}s)\n{ans[:700]}", flush=True)

json.dump(results, open("/home/juanpablo/decimal_test_results.json", "w"), ensure_ascii=False, indent=1)
print("\n\nguardado -> decimal_test_results.json")
