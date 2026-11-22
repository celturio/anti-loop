#!/usr/bin/env python3
"""A/B blind reasoning test: mimo-q8-0 vs mimo-heretic (same base, same params)."""
import json, urllib.request, time

URL = "http://127.0.0.1:8080/v1/chat/completions"
QUESTIONS = [
    ("logica",   "Tres amigos revisan 3 cajas. Cada caja tiene 3 frutas. Si una fruta esta podrida, toda la caja se tira. Encuentran 2 podridas. ¿Cuantas cajas se tiran como maximo y como minimo? Explica breve."),
    ("mates",    "¿Cuanto es 17 * 23 - 89 / 7 + 15? Muestra solo el calculo final aproximado a 2 decimales."),
    ("trampa",   "Un tren sale de Santiago a 120 km/h. Otro de Valparaiso a 180 km/h hacia Santiago, separados 300 km. ¿Se encuentran antes o despues de 1 hora? Razona paso a paso."),
    ("codex",    "Encuentra el bug: def suma(l): t=0; for x in l: if type(x)==int: t+=x; return t. ¿Que pasa con suma([1, 2.5, 3])?"),
    ("abstract", "Si globo:flotar entonces piedra:??? Elige entre: (A) flotar (B) hundirse (C) rebotar (D) volar y razona en 1 linea."),
]

def ask(model, q, max_tokens=1800):
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
    if not txt:  # reasoning se comio el budget: mostrar thinking como fallback
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
        print(f"\n[{tag}] ({fin}, {dt:.1f}s)\n{ans[:600]}", flush=True)

json.dump(results, open("/home/juanpablo/ab_test_results.json", "w"), ensure_ascii=False, indent=1)
print("\n\nguardado -> ab_test_results.json")
