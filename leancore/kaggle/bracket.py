#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bracket.py — движок турнира конфигов LeanCore (ASHA-брекет по непрерывным скалярам).
Принципы (зафиксированы в TRICKS ADV-R2 и брифе):
  1. Пространство = непрерывные скаляры (mulr, ssfull, ssk, ssalpha, lr), НЕ архитектуры.
  2. Парность: фиксированный инит у всех (crc32 у numpy-форка / INIT_SEED у torch),
     negrng-разделение, seeds парные [1,42] → финал [1,42,7,99].
  3. Предохранители: авто-DQ (‖W‖/‖W₀‖<3 после шага 100 = «мёртвый»; NaN), контроль CTRL
     (рецепт чемпиона) шагает ВСЕ рунги как референс и не выбывает; страховочный слот —
     лучший НАКЛОН (late-bloomer, наша измеренная ловушка ss/full: ранний лидер ≠ поздний).
  4. Возобновляемость: state.json + файловые локи; переживает таймаут сессии Kaggle
     (attach output прошлой версии как dataset → init перенесёт state в working).
Режимы: init | worker | status | summary. QUICK=1 (env) — трубопроводный прогон (4 конфига).
"""
import os, sys, json, math, time, glob, shutil, subprocess, argparse
import numpy as hnp

QUICK = os.environ.get("BRACKET_QUICK", "0") == "1"

CTRL = dict(name="CTRL", mulr=0.02, ssfull=0.12, ssk=512, ssalpha=1.0, lr=6e-4)

if QUICK:
    RUNGS = [dict(steps=20, seeds=[1], quota=3), dict(steps=30, seeds=[1], quota=2)]
    N_SAMPLES = 3
else:
    RUNGS = [dict(steps=60, seeds=[1, 42], quota=12),
             dict(steps=200, seeds=[1, 42], quota=5),
             dict(steps=500, seeds=[1, 42], quota=3),
             dict(steps=1500, seeds=[1, 42, 7, 99], quota=3)]
    N_SAMPLES = 23

PREDICTIONS = """# ПРЕДСКАЗАНИЯ БРЕКЕТА (зафиксированы ДО запуска)
1. CTRL останется в пределах ±2.8% (пол) от домашнего 111.21 на финальном рунге [проверка движка].
2. mulr: оптимум в [0.015, 0.030]; края 0.010/0.040 проиграют центру.
3. ssfull 0.25 проиграет 0.12 на R3/R4 (избыток полного CE на коротком бюджете).
4. ssalpha 0.75 ≥ 1.0 на ранних рунгах, разница сожмётся к финалу.
5. ssk: 512 не хуже 768/256 в пределах пола; 256 не выиграет у 512 больше чем на 3%.
6. lr×1.4 (8.4e-4) хуже базы на R2+ (muon и так на краю), lr×0.7 — в пределах пола.
7. Страховочный слот сработает ≥1 раза (конфиг с лучшим наклоном не войдёт в топ по PPL).
8. Победитель финала отстанет от CTRL меньше чем на 4% → раунд 2 с суженными диапазонами.
"""


def cfgs_samples():
    rng = hnp.random.default_rng(20240830)
    n, d = N_SAMPLES, 5
    pts = hnp.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.random(n)) / n
    cfgs = []
    for i in range(n):
        mulr = 0.010 * (0.040 / 0.010) ** pts[i, 0]
        ssfull = 0.0 + 0.30 * pts[i, 1]
        ssk = int(round((256 * (768 / 256) ** pts[i, 2]) / 32) * 32)
        ssalpha = 0.5 + 0.5 * pts[i, 3]
        lr = 4e-4 * (9e-4 / 4e-4) ** pts[i, 4]
        cfgs.append(dict(name=f"C{i:02d}", mulr=round(mulr, 4), ssfull=round(ssfull, 3),
                         ssk=min(ssk, 768), ssalpha=round(ssalpha, 3), lr=round(lr, 7)))
    return [CTRL] + cfgs


def wdir(workdir, *p):
    return os.path.join(workdir, *p)


def load_state(workdir):
    p = wdir(workdir, "state.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"jobs": {}, "decided": {}, "created": int(time.time())}


def save_state(workdir, st):
    tmp = wdir(workdir, "state.json.tmp")
    json.dump(st, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, wdir(workdir, "state.json"))


def job_key(rung, cfg_name, seed):
    return f"r{rung}_{cfg_name}_s{seed}"


def cfg_cli(engine, cfg, steps, eval_every, seed, tag, data, saveckpt):
    if engine == "torch":
        cmd = [sys.executable, "train_torch.py", "--steps", str(steps), "--eval_every", str(eval_every),
               "--lr", str(cfg["lr"]), "--mulr", str(cfg["mulr"]), "--ssk", str(cfg["ssk"]),
               "--ssfull", str(cfg["ssfull"]), "--ssalpha", str(cfg["ssalpha"]), "--seed", str(seed),
               "--negrng", "1", "--trunknorm", "1", "--tag", tag, "--data", data,
               "--saveckpt", str(saveckpt)]
    else:
        cmd = [sys.executable, "nano_lc_kg.py", "--kind", "ema", "--opt", "muon", "--steps", str(steps),
               "--eval_every", str(eval_every), "--lr", str(cfg["lr"]), "--mulr", str(cfg["mulr"]),
               "--ssk", str(cfg["ssk"]), "--ssfull", str(cfg["ssfull"]), "--ssalpha", str(cfg["ssalpha"]),
               "--seed", str(seed), "--negrng", "1", "--trunknorm", "1", "--tag", tag, "--data", data]
    return cmd


def ensure_rung(workdir, st, cfgs_all, rung, cfg_names):
    """Создать джобы рунга (идемпотентно)."""
    r = RUNGS[rung]
    ee = max(10, r["steps"] // 6)
    for cname in cfg_names:
        cfg = next(c for c in cfgs_all if c["name"] == cname)
        for seed in r["seeds"]:
            k = job_key(rung, cname, seed)
            if k not in st["jobs"]:
                st["jobs"][k] = dict(rung=rung, cfg=cname, seed=seed, steps=r["steps"],
                                     eval_every=ee, status="pending",
                                     tag=f"bk_{k}")


def rung_done(st, rung):
    js = [j for j in st["jobs"].values() if j["rung"] == rung]
    return js and all(j["status"] in ("done", "dq") for j in js)


def evaluate_rung(workdir, st, cfgs_all, rung):
    """Решение рунга: скоринг, DQ-причины, квота + страховочный слот + CTRL."""
    r = RUNGS[rung]
    by_cfg = {}
    for j in st["jobs"].values():
        if j["rung"] == rung and j["status"] in ("done", "dq"):
            by_cfg.setdefault(j["cfg"], []).append(j)
    rows = []
    for cname, jobs in by_cfg.items():
        ok = [j for j in jobs if j["status"] == "done"]
        if not ok:
            rows.append((cname, float("inf"), 0.0, "все джобы dq")); continue
        ppl = hnp.mean([j["val_ppl"] for j in ok])
        slope = hnp.mean([j.get("slope", 0.0) for j in ok])
        rows.append((cname, float(ppl), float(slope), ""))
    rows.sort(key=lambda t: t[1])
    quota = r["quota"]
    survivors = ["CTRL"] if "CTRL" in by_cfg else []
    nslots = max(0, quota - len(survivors))
    picked = [c for c, p, _, _ in rows if c != "CTRL" and math.isfinite(p)][:nslots]
    rest = [c for c in [r_[0] for r_ in rows] if c != "CTRL" and c not in picked]
    insurance = None
    if rest and rung < len(RUNGS) - 1:
        insurance = max(rest, key=lambda c: next(s for cn, _, s, _ in rows if cn == c))
    survivors += picked + ([insurance] if insurance else [])
    # merge-сохранение: воркеры могли параллельно дописать джобы — не затираем
    st2 = load_state(workdir)
    st2["decided"][str(rung)] = dict(
        table=[dict(cfg=c, ppl=round(p, 2), slope=round(s, 2), note=n) for c, p, s, n in rows],
        survivors=survivors, insurance=insurance)
    save_state(workdir, st2)
    return survivors


def cmd_init(args):
    workdir = args.workdir
    os.makedirs(wdir(workdir, "jobs"), exist_ok=True)
    os.makedirs(wdir(workdir, "results"), exist_ok=True)
    for stale in glob.glob(wdir(workdir, "jobs", "*.lock")):
        shutil.rmtree(stale, ignore_errors=True)
    # перенос state из приаттаченного прошлого вывода (Kaggle resume)
    if not os.path.exists(wdir(workdir, "state.json")):
        for cand in glob.glob("/kaggle/input/*/state.json"):
            shutil.copy(cand, wdir(workdir, "state.json"))
            print(f"[init] восстановлен state из {cand}", flush=True)
            break
    st = load_state(workdir)
    # сброс зависших "running" (воркер умер посреди джобы / таймаут сессии)
    n_reset = 0
    for j in st["jobs"].values():
        if j["status"] == "running":
            j["status"] = "pending"; n_reset += 1
    if n_reset:
        print(f"[init] вернул в pending зависших джоб: {n_reset}", flush=True)
    cfgs_all = cfgs_samples()
    if not os.path.exists(wdir(workdir, "configs.json")):
        open(wdir(workdir, "configs.json"), "w").write(json.dumps(cfgs_all, indent=1))
        ensure_rung(workdir, st, cfgs_all, 0, [c["name"] for c in cfgs_all])
    save_state(workdir, st)
    open(wdir(workdir, "PREDICTIONS.md"), "w").write(PREDICTIONS)
    print(f"[init] конфигов: {len(cfgs_all)}, рунгов: {len(RUNGS)}, джоб в пуле: {len(st['jobs'])}", flush=True)


def parse_result(workdir, job):
    """Прочитать jsonl прогона → (status, val_ppl, slope, wall, dq_reason).
    Тренер пишет в <script_dir>/results (ROOT тренера), не в workdir."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"run_{job['tag']}.jsonl")
    if not os.path.exists(path):
        return "dq", None, None, None, "нет jsonl (падение процесса)"
    rows = [json.loads(l) for l in open(path) if l.strip().startswith("{")]
    if not rows:
        return "dq", None, None, None, "пустой jsonl"
    last = rows[-1]
    ppl = last.get("val_ppl")
    if ppl is None or not math.isfinite(ppl):
        return "dq", None, None, None, "NaN финал"
    for r_ in rows:
        if r_["step"] >= 100 and "wn" in r_ and r_["wn"] < 3.0:
            return "dq", float(ppl), None, last.get("wall"), f"мертв: wn={r_['wn']} @step {r_['step']}"
    if len(rows) >= 2:
        slope = (rows[0]["val_ppl"] - last["val_ppl"]) / max(1, last["step"] - rows[0]["step"])
    else:
        slope = 0.0
    return "done", float(ppl), float(slope), last.get("wall"), ""


def cmd_worker(args):
    workdir, cfgs_all = args.workdir, json.load(open(wdir(args.workdir, "configs.json")))
    wid = args.id
    engine = args.engine
    env = dict(os.environ)
    if args.backend:
        env["LC_BACKEND"] = args.backend
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    while True:
        st = load_state(workdir)
        # все ли рунги решены?
        if str(len(RUNGS) - 1) in st["decided"]:
            print(f"[w{wid}] брекет завершён", flush=True); return
        # попытаться решить текущий рунг
        for ri in range(len(RUNGS)):
            if str(ri) in st["decided"]:
                continue
            if rung_done(st, ri):
                lock = wdir(workdir, "jobs", f"decide{ri}.lock")
                try:
                    os.mkdir(lock)
                    st = load_state(workdir)
                    if str(ri) not in st["decided"]:
                        surv = evaluate_rung(workdir, st, cfgs_all, ri)
                        if ri + 1 < len(RUNGS):
                            st3 = load_state(workdir)            # свежий объект, без затирания
                            ensure_rung(workdir, st3, cfgs_all, ri + 1, surv)
                            save_state(workdir, st3)
                        print(f"[w{wid}] рунг {ri} решён → {surv}", flush=True)
                finally:
                    shutil.rmtree(lock, ignore_errors=True)
            break
        st = load_state(workdir)
        pend = sorted([j for j in st["jobs"].values() if j["status"] == "pending"],
                      key=lambda j: (j["rung"], j["cfg"], j["seed"]))
        claimed = None
        for j in pend:
            # работаем только по неоткрытому рунгу
            if any(str(r0) not in st["decided"] and r0 < j["rung"] for r0 in range(len(RUNGS))):
                continue
            lock = wdir(workdir, "jobs", f"{job_key(j['rung'], j['cfg'], j['seed'])}.lock")
            try:
                os.mkdir(lock)
                claimed = (j, lock); break
            except FileExistsError:
                continue
        if claimed is None:
            print(f"[w{wid}] нет доступных джоб; выход (другие воркеры добивают)", flush=True)
            return
        job, lock = claimed
        key = job_key(job["rung"], job["cfg"], job["seed"])
        st = load_state(workdir)
        st["jobs"][key]["status"] = "running"
        save_state(workdir, st)
        cfg = next(c for c in cfgs_all if c["name"] == job["cfg"])
        cmd = cfg_cli(engine, cfg, job["steps"], job["eval_every"], job["seed"],
                      job["tag"], args.data, saveckpt=(1 if job["rung"] == len(RUNGS) - 1 else 0))
        print(f"[w{wid}] СТАРТ {key} (steps={job['steps']} seed={job['seed']})", flush=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                                  env=env, timeout=args.job_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        status, ppl, slope, wall, dq = parse_result(workdir, job)
        if rc != 0 and status == "done":
            status, dq = "dq", f"rc={rc}"
        st = load_state(workdir)
        st["jobs"][key].update(status=status, val_ppl=ppl, slope=slope, wall=wall,
                               dq=dq, worker=wid, elapsed=round(time.time() - t0, 1))
        save_state(workdir, st)
        shutil.rmtree(lock, ignore_errors=True)
        print(f"[w{wid}] ФИН {key}: {status} ppl={ppl} {dq}", flush=True)


def cmd_status(args):
    st = load_state(args.workdir)
    for ru in range(len(RUNGS)):
        js = [j for j in st["jobs"].values() if j["rung"] == ru]
        if not js:
            continue
        done = sum(1 for j in js if j["status"] in ("done", "dq"))
        print(f"rung{ru}: {done}/{len(js)}", end="")
        if str(ru) in st["decided"]:
            print(f"  решён → {st['decided'][str(ru)]['survivors']}", end="")
        print()
    if args.verbose:
        for k, j in sorted(st["jobs"].items()):
            print(f"  {k}: {j['status']} ppl={j.get('val_ppl')} {j.get('dq', '')}")


def cmd_summary(args):
    st = load_state(args.workdir)
    cfgs_all = json.load(open(wdir(args.workdir, "configs.json")))
    L = ["# ТУРНИР LeanCore — итоги", "", f"engine/data: см. лог воркеров; конфигов: {len(cfgs_all)}", ""]
    for ru in range(len(RUNGS)):
        dec = st["decided"].get(str(ru))
        if not dec:
            continue
        L.append(f"## Рунг {ru} ({RUNGS[ru]['steps']} шагов, seeds={RUNGS[ru]['seeds']})")
        L.append("| cfg | PPL(ср по сидам) | наклон | прим |")
        L.append("|---|---|---|---|")
        for row in dec["table"]:
            L.append(f"| {row['cfg']} | {row['ppl']} | {row['slope']} | {row['note']} |")
        L.append(f"\nвыжившие: {dec['survivors']}  (страховка: {dec.get('insurance')})")
        L.append("")
    fin_jobs = [j for j in st["jobs"].values()
                if j["rung"] == len(RUNGS) - 1 and j["status"] == "done"
                and j["cfg"] in st["decided"].get(str(len(RUNGS) - 1), {}).get("survivors", [])]
    if fin_jobs:
        L.append("## Финал — по сидам")
        L.append("| cfg | seed | PPL |")
        L.append("|---|---|---|")
        for j in sorted(fin_jobs, key=lambda j: (j["cfg"], j["seed"])):
            L.append(f"| {j['cfg']} | {j['seed']} | {j['val_ppl']} |")
    out = "\n".join(L) + "\n"
    open(wdir(args.workdir, "SUMMARY.md"), "w").write(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "worker", "status", "summary"):
        sp = sub.add_parser(name)
        sp.add_argument("--workdir", default="bracket_work")
        if name == "status":
            sp.add_argument("--verbose", action="store_true")
        if name in ("init", "worker"):
            sp.add_argument("--data", default="data/prep")
        if name == "worker":
            sp.add_argument("--id", type=int, default=0)
            sp.add_argument("--engine", choices=["torch", "numpy"], default="numpy")
            sp.add_argument("--backend", default=None)
            sp.add_argument("--cuda", default=None)
            sp.add_argument("--job_timeout", type=int, default=6 * 3600)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    {"init": cmd_init, "worker": cmd_worker, "status": cmd_status, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
