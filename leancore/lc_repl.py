#!/usr/bin/env python3
"""lc_repl.py — интерактивная обвязка для lc_stream (chat-протокол).
Токенизатор = тот же, что в prep_data.py; словарь из data/prep/meta.json.
Фичи обвязки: системный промпт/конституция (--constitution FILE, prepended как контекст),
бюджет генерации (max tokens), sampling (temp/top-k), seed, reset."""
import json, re, sys, subprocess, argparse, os

ROOT = os.path.dirname(os.path.abspath(__file__))
VOCAB = json.load(open(os.path.join(ROOT, "data/prep/meta.json")))
STOI = VOCAB["stoi"]; ITOS = {i: w for w, i in STOI.items()}
WORD = re.compile(r"[a-z']+|[0-9]+|[^\s\w]")

def tok(text: str):
    return [STOI.get(t, 2) for t in WORD.findall(text.lower())]

def detok(ids):
    out = []
    for i in ids:
        w = ITOS.get(i, "<unk>")
        if out and not (len(w) == 1 and not w.isalnum()): out.append(" ")
        out.append(w)
    return "".join(out)

class Engine:
    def __init__(self, model, seed=42):
        self.p = subprocess.Popen([os.path.join(ROOT, "lc_stream"), model, "chat"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.cmd(".seed %d" % seed)
    def cmd(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()
        return self.p.stdout.readline().strip()
    def step(self, ids):
        for i in ids: self.cmd(f".step {i}")
    def gen(self, n, temp, topk):
        return [int(t) for t in self.cmd(f".gen {n} {temp} {topk}").split() if t.lstrip("-").isdigit()]
    def reset(self): self.cmd(".reset")
    def tau(self, v): return self.cmd(f".tau {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=os.path.join(ROOT, "results/champ4k_qat.lcw2"))
    ap.add_argument("--temp", type=float, default=0.8); ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max", type=int, default=96); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--constitution", default=None, help="файл с системным промптом-конституцией")
    ap.add_argument("--tau", type=float, default=None, help="compute-бюджет ADR гейта (аналог thinking-budget)")
    ap.add_argument("--revise", action="store_true", help="draft → critique-pass → revise (CAI-lite)")
    a = ap.parse_args()
    e = Engine(a.model, a.seed)
    if a.tau is not None: e.tau(a.tau)
    sys_prompt = open(a.constitution).read().strip() if a.constitution else \
        "the following is a dialogue between a person and an assistant, wise and honest:"
    print("[lc_repl] модель:", a.model, "| temp", a.temp, "topk", a.topk, "| /reset /exit")
    hist_on = True
    while True:
        try: q = input("you> ").strip()
        except EOFError: break
        if q in ("/exit", "/quit"): break
        if q == "/reset": e.reset(); print("[ok]"); continue
        e.reset()
        e.step([1] + tok(sys_prompt))
        e.step(tok("\n" + q + "\n"))
        ans = e.gen(a.max, a.temp, a.topk)
        if a.revise:
            # CAI-lite: модель критикует черновик (конституция = системный промпт) и переписывает
            e.step(tok("\ncritique of the above:\n"))
            crit = e.gen(a.max // 2, a.temp, a.topk)
            e.step(tok("\nrevised answer:\n"))
            ans = e.gen(a.max, a.temp, a.topk)
            print("lc!(draft→critique→revise) critique:", detok(crit))
        print("lc>  " + detok(ans))
    e.cmd(".quit")

if __name__ == "__main__":
    main()
