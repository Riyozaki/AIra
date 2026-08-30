#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evo.py — эволюционный движок поиска конфигов LeanCore (стадии A→B→C).

Зачем (замена «просто перебору»): ASHA-брекет выбирает один раз из фиксированного
облака и ничему не учится. Здесь популяция РАЗВИВАЕТСЯ: селекция → кроссовер →
мутации с адаптивным шагом (правило 1/5), зал справедливости (HOF) по всей истории.

Методология материнского проекта сохранена:
  - парный детерминированный инит (crc32-имена у numpy-форка; train_torch с v3 —
    битово тот же инит), negrng-разделение, парные сиды;
  - CTRL (рецепт чемпиона) в каждом поколении как якорь; фитнес = ΔPPL к CTRL
    ЭТОГО ЖЕ поколения (снимает дрейф батча/данных между поколениями);
  - авто-DQ «мёртвых» (wn<3 после шага 100), финал 1500 × сиды [1,42,7,99];
  - возобновляемость: evo_state.json + локи, переживает таймаут сессии Kaggle.

Стадии: A qual (16 конфигов: CTRL+15 LHS @60, сид 1 — дешёвый отсев дна) →
B evo (популяция 8, поколений 3 @200, сиды [1,42]) → C final (HOF-3 + CTRL
@1500 × 4 сида). Прогнозы — PREDICTIONS ниже, зафиксированы ДО запуска.

Режимы CLI: init | worker | status | summary. Env EVO_QUICK=1 — трубопроводный
дым (qual 4 @20, 2 поколения по 4, финал @30).
"""
import os, sys, json, math, time, glob, shutil, signal, argparse, subprocess
import numpy as hnp

QUICK = os.environ.get("EVO_QUICK", "0") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))

# точка отсчёта — рецепт чемпиона (как в bracket.py)
CTRL = dict(name="CTRL", mulr=0.02, ssfull=0.12, ssk=512, ssalpha=1.0, lr=6e-4, origin="ctrl")

BOUNDS = dict(mulr=(0.010, 0.040), ssfull=(0.0, 0.30), ssk=(256, 768), ssalpha=(0.5, 1.0), lr=(4e-4, 9e-4))
SSK_STEPS = [256, 320, 384, 512, 768]

if QUICK:
    QUAL_N, QUAL_STEPS, QUAL_SEEDS = 3, 20, [1]
    POP, GENS, EVO_STEPS, EVO_SEEDS = 4, 2, 20, [1]
    FINAL_TOP, FINAL_STEPS, FINAL_SEEDS = 2, 30, [1]
else:
    QUAL_N, QUAL_STEPS, QUAL_SEEDS = 15, 60, [1]
    POP, GENS, EVO_STEPS, EVO_SEEDS = 8, 3, 200, [1, 42]
    FINAL_TOP, FINAL_STEPS, FINAL_SEEDS = 3, 1500, [1, 42, 7, 99]

MASTER_SEED = 0xE90

PREDICTIONS = """# ПРЕДСКАЗАНИЯ EVO-КОНТУРА (зафиксированы ДО запуска, снять после SUMMARY)
E1. Технический: история поколений при том же MASTER_SEED воспроизводима битово
    (трасса конфигов, не весов) в двух независимых сессиях [контроль механизма].
E2. Поисковый: к поколению 3 медиана ssalpha популяции ≤ 1.0 — механизм САМ сместится
    вниз по α (α=0.75 дал −1.30% @1500 в чистой паре [измерено дома]; подсказки нет).
E3. Итоговый: лучший HOF-конфиг на финале (4 сида) будет НЕ хуже CTRL более чем на
    +0.5% и, ожидаемо, −0.5…−2.5% к CTRL (якорь — α-пара). Если хуже +0.5% — сигнал
    @200 ниже нашего шума, контур в таком виде паркингуется [отрицательная ветка].
E4. Структурный: успешность кроссоверов ≤ успешности мутаций (пространство почти
    сепарабельное — наш архив это поддерживает) [проверится по трассе].
E5. Якорный: CTRL не выбывает из топ-3 ни в одном поколении больше чем по вине ≤1 мутанта
    (пол одно-парного шума у нас ±1.5–2.5% — систематический каскад выбиваний означал бы
    переобучение механизма в шум).
"""


# ----------------------------------------------------------------- геном и операторы
def _ckey(c):
    return (c["mulr"], c["ssfull"], c["ssk"], c["ssalpha"], c["lr"])


def _add_unique(pop, cand):
    if all(_ckey(cand) != _ckey(c) for c in pop):
        pop.append(cand)
        return True
    return False
def to_gene(c):
    """Конфиг → точка в гене-пространстве (лог-координаты там, где масштаб мультипликативен)."""
    return dict(l_mulr=math.log(c["mulr"]), l_lr=math.log(c["lr"]),
                l_ssfull=math.log(c["ssfull"] + 0.02), a= c["ssalpha"], k=SSK_STEPS.index(
                    min(SSK_STEPS, key=lambda s: abs(s - c["ssk"]))))


def from_gene(g):
    def cl(x, lo, hi):
        return max(lo, min(hi, x))
    return dict(mulr=round(cl(math.exp(g["l_mulr"]), *BOUNDS["mulr"]), 4),
                ssfull=round(cl(math.exp(g["l_ssfull"]) - 0.02, *BOUNDS["ssfull"]), 3),
                ssk=SSK_STEPS[int(round(cl(g["k"], 0, len(SSK_STEPS) - 1)))],
                ssalpha=round(cl(g["a"], *BOUNDS["ssalpha"]), 3),
                lr=round(cl(math.exp(g["l_lr"]), *BOUNDS["lr"]), 7))


def mutate(cfg, rng, sigma):
    g = to_gene(cfg)
    g["l_mulr"] += rng.normal(0, sigma)
    g["l_lr"] += rng.normal(0, sigma * 0.8)
    g["l_ssfull"] += rng.normal(0, sigma)
    g["a"] += rng.normal(0, sigma * 0.35)
    if rng.random() < min(0.75, sigma * 3.0):       # ступенчатая координата
        g["k"] += int(rng.choice([-1, 1]))
    out = from_gene(g)
    out.update(name="", origin=f"mut({cfg['name']})")
    return out


def crossover(ca, cb, rng):
    out = {}
    for key in ("mulr", "ssfull", "ssk", "ssalpha", "lr"):
        out[key] = ca[key] if rng.random() < 0.5 else cb[key]
    out.update(name="", origin=f"x({ca['name']}×{cb['name']})")
    return out


# ----------------------------------------------------------------- state / storage
def wp(workdir, *p):
    return os.path.join(workdir, *p)


def load(workdir):
    p = wp(workdir, "evo_state.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"stage": "qual", "gen": 0, "sigma": 0.18, "jobs": {}, "pop": [],
            "history": [], "hof": {}, "qual_cfgs": [], "created": int(time.time())}


def save(workdir, st):
    tmp = wp(workdir, "evo_state.json.tmp")
    json.dump(st, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, wp(workdir, "evo_state.json"))


def _lock(workdir, name, wait=120.0):
    """mkdir-based мьютекс (атомарен на POSIX)."""
    p = wp(workdir, "jobs", name)
    t0 = time.time()
    while True:
        try:
            os.mkdir(p)
            return p
        except FileExistsError:
            if time.time() - t0 > wait:
                raise TimeoutError(f"lock {name}: >{wait:.0f}s")
            time.sleep(0.05)


def _unlock(p):
    shutil.rmtree(p, ignore_errors=True)


def state_update(workdir, mutate):
    """Атомарный read-modify-write evo_state.json под глобальным локом.
    Без него два воркера, закончившие джобу одновременно, затирали записи друг друга
    (load→update→save всем файлом) → повторные прогоны и сожжённые часы [вердикт
    архитектурного разбора агента-2; у нас такая же конструкция — патч принят]."""
    p = _lock(workdir, "state.lock")
    try:
        st = load(workdir)
        mutate(st)
        save(workdir, st)
        return st
    finally:
        _unlock(p)


def jkey(stage, gen, name, seed):
    return f"{stage}{gen}_{name}_s{seed}"


def cfg_cli(engine, cfg, steps, eval_every, seed, tag, data, saveckpt):
    if engine == "torch":
        return [sys.executable, "train_torch.py", "--steps", str(steps), "--eval_every", str(eval_every),
                "--lr", str(cfg["lr"]), "--mulr", str(cfg["mulr"]), "--ssk", str(cfg["ssk"]),
                "--ssfull", str(cfg["ssfull"]), "--ssalpha", str(cfg["ssalpha"]), "--seed", str(seed),
                "--negrng", "1", "--trunknorm", "1", "--tag", tag, "--data", data,
                "--saveckpt", str(saveckpt)]
    return [sys.executable, "nano_lc_kg.py", "--kind", "ema", "--opt", "muon", "--steps", str(steps),
            "--eval_every", str(eval_every), "--lr", str(cfg["lr"]), "--mulr", str(cfg["mulr"]),
            "--ssk", str(cfg["ssk"]), "--ssfull", str(cfg["ssfull"]), "--ssalpha", str(cfg["ssalpha"]),
            "--seed", str(seed), "--negrng", "1", "--trunknorm", "1", "--tag", tag, "--data", data]


def lhs_cfgs(n):
    """CTRL + n LHS-точек пространства (тот же генератор, что в bracket, — сопоставимость)."""
    rng = hnp.random.default_rng(20240830)
    pts = hnp.zeros((n, 5))
    for j in range(5):
        perm = rng.permutation(n)
        pts[:, j] = (perm + rng.random(n)) / n
    out = [dict(CTRL)]
    for i in range(n):
        q = dict(name=f"Q{i:02d}",
                 mulr=round(0.010 * 4.0 ** pts[i, 0], 4),
                 ssfull=round(0.30 * pts[i, 1], 3),
                 ssk=min(int(round(256 * 3.0 ** pts[i, 2] / 32) * 32), 768),
                 ssalpha=round(0.5 + 0.5 * pts[i, 3], 3),
                 lr=round(4e-4 * (9e-4 / 4e-4) ** pts[i, 4], 7), origin="lhs")
        out.append(q)
    return out


def queue_stage_jobs(st, stage, gen, cfgs, steps, seeds):
    ee = max(10, steps // 6)
    for cfg in cfgs:
        for seed in seeds:
            k = jkey(stage, gen, cfg["name"], seed)
            if k not in st["jobs"]:
                st["jobs"][k] = dict(stage=stage, gen=gen, cfg=dict(cfg), steps=steps,
                                     eval_every=ee, seed=seed, status="pending",
                                     tag=f"ev_{k}")


def parse_result(job):
    path = os.path.join(HERE, "results", f"run_{job['tag']}.jsonl")
    if not os.path.exists(path):
        return "dq", None, 0.0, "нет jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip().startswith("{")]
    if not rows:
        return "dq", None, 0.0, "пустой jsonl"
    last = rows[-1]
    ppl = last.get("val_ppl")
    if ppl is None or not math.isfinite(ppl):
        return "dq", None, 0.0, "NaN"
    for r in rows:
        if r["step"] >= 100 and "wn" in r and r["wn"] < 3.0:
            return "dq", float(ppl), 0.0, f"мертв: wn={r['wn']}@{r['step']}"
    nst = int(job.get("steps", 0))
    if nst >= 20 and last.get("step", nst) < 0.6 * nst:
        return "dq", float(ppl), 0.0, f"недопрогон: {last.get('step')}/{nst} шагов"
    p0 = rows[0].get("val_ppl")
    if p0 and float(ppl) >= 0.9 * float(p0):
        return "dq", float(ppl), 0.0, f"нет движения: ppl={ppl} @0={p0}"
    slope = ((rows[0]["val_ppl"] - ppl) / max(1, last["step"] - rows[0]["step"])) if len(rows) > 1 else 0.0
    return "done", float(ppl), float(slope), ""


# ----------------------------------------------------------------- решения стадий
def gen_jobs(st, stage, gen):
    return [j for j in st["jobs"].values() if j["stage"] == stage and j["gen"] == gen]


def stage_done(js):
    return js and all(j["status"] in ("done", "dq") for j in js)


def score_by_cfg(js):
    by = {}
    for j in js:
        if j["status"] == "done":
            by.setdefault(j["cfg"]["name"], []).append(j["val_ppl"])
    return {k: float(hnp.mean(v)) for k, v in by.items()}


def advance(workdir, st):
    """Лок нужен снаружи. Решает завершённую стадию и раскладывает следующую."""
    stage, gen = st["stage"], st["gen"]
    js = gen_jobs(st, stage, gen)
    if stage == "qual":
        sc = score_by_cfg(js)
        cjob = {j["cfg"]["name"]: j["cfg"] for j in js}
        ranked = sorted(sc.items(), key=lambda t: t[1])
        st["history"].append(dict(stage="qual", gen=0,
                                  table=[dict(cfg=k, ppl=round(v, 2)) for k, v in ranked]))
        top2 = [cjob[k] for k, _ in ranked if k != "CTRL"][:2]
        rng = hnp.random.default_rng(hnp.random.SeedSequence([MASTER_SEED, 0]))
        pop = [dict(CTRL)] + top2
        guard = 0
        while len(pop) < POP - 3 and guard < 24:
            guard += 1
            _add_unique(pop, mutate(rng.choice(top2), rng, st["sigma"]))
        for cand in [crossover(top2[0], dict(CTRL), rng), crossover(top2[-1], dict(CTRL), rng),
                     mutate(dict(CTRL), rng, st["sigma"])]:
            _add_unique(pop, cand)
        guard = 0
        while len(pop) < POP and guard < 24:
            guard += 1
            _add_unique(pop, mutate(rng.choice(top2), rng, st["sigma"]))
        pop = pop[:POP]
        for i, c in enumerate(pop):
            if not c["name"].startswith(("CTRL", "Q")) or sum(p["name"] == c["name"] for p in pop) > 1:
                c["name"] = f"G1.{i}"
        st["pop"] = [dict(c) for c in pop]
        st["stage"], st["gen"] = "evo", 1
        queue_stage_jobs(st, "evo", 1, pop, EVO_STEPS, EVO_SEEDS)
        return f"qual решён → популяция G1: {[c['name'] for c in pop]}"
    if stage == "evo":
        sc = score_by_cfg(js)
        ctrl_ppl = sc.get("CTRL")
        genes = {j["cfg"]["name"]: j["cfg"] for j in js}
        tbl = []
        for name, ppl in sorted(sc.items(), key=lambda t: t[1]):
            tbl.append(dict(cfg=name, ppl=round(ppl, 2),
                            dctrl=round(ppl - ctrl_ppl, 2) if ctrl_ppl else None,
                            origin=genes[name].get("origin", "")))
        ranked = [t["cfg"] for t in tbl if t["cfg"] != "CTRL"]
        top3_names = ranked[:3]
        elites = [genes[n] for n in ranked[:2]] + [dict(CTRL)]   # CTRL — вечный якорный элит
        if ctrl_ppl:  # HOF: лучший dctrl конфига за всю историю
            for name, ppl in sc.items():
                if name == "CTRL":
                    continue
                fit = ctrl_ppl - ppl
                if fit > st["hof"].get("fit", float("-inf")):
                    st["hof"] = dict(cfg=dict(genes[name]), fit=round(fit, 2), gen=gen)
        succ = sum(1 for j in js if j["status"] == "done" and "mut" in j["cfg"].get("origin", "")
                   and j["cfg"]["name"] in top3_names)
        nmut = max(1, sum(1 for j in js if "mut" in j["cfg"].get("origin", "")))
        ps = succ / nmut
        st["sigma"] = round(min(0.5, max(0.06, st["sigma"] * (1.22 if ps > 0.2 else 0.82))), 3)
        st["history"].append(dict(stage="evo", gen=gen, table=tbl, sigma=st["sigma"],
                                  ps=round(ps, 2), top3=top3_names))
        if gen >= GENS:
            st["stage"], st["gen"] = "final", 0
            finalists = [dict(CTRL)]
            if st["hof"] and _ckey(st["hof"]["cfg"]) != _ckey(CTRL):
                finalists.append(dict(st["hof"]["cfg"]))
            for e in elites:
                if all(_ckey(e) != _ckey(f) for f in finalists):
                    finalists.append(dict(e))
                if len(finalists) >= FINAL_TOP + 1:
                    break
            queue_stage_jobs(st, "final", 0, finalists, FINAL_STEPS, FINAL_SEEDS)
            return f"evo финиш → финал: {[f['name'] for f in finalists]}, HOF={st['hof'] and st['hof']['cfg']['name']}"
        rng = hnp.random.default_rng(hnp.random.SeedSequence([MASTER_SEED, gen]))
        pop = [dict(e) for e in elites]
        guard = 0
        while len(pop) < POP - 1 and guard < 24:
            guard += 1
            _add_unique(pop, mutate(elites[rng.integers(0, len(elites))], rng, st["sigma"]))
        if not _add_unique(pop, crossover(elites[0], elites[-1], rng)):
            _add_unique(pop, mutate(elites[0], rng, st["sigma"]))
        for i, c in enumerate(pop):
            if c["name"] != "CTRL":
                c["name"] = f"G{gen+1}.{i}"
        st["pop"] = [dict(c) for c in pop]
        st["gen"] = gen + 1
        queue_stage_jobs(st, "evo", gen + 1, pop, EVO_STEPS, EVO_SEEDS)
        return f"поколение {gen} решено (ps={ps:.2f}, σ→{st['sigma']}) → G{gen+1}"
    return "final завершён"


# ----------------------------------------------------------------- CLI
def cmd_init(args):
    os.makedirs(wp(args.workdir, "jobs"), exist_ok=True)
    for stale in glob.glob(wp(args.workdir, "jobs", "*.lock")):
        shutil.rmtree(stale, ignore_errors=True)
    if not os.path.exists(wp(args.workdir, "evo_state.json")):
        for cand in glob.glob("/kaggle/input/*/evo_state.json"):
            shutil.copy(cand, wp(args.workdir, "evo_state.json"))
            print(f"[evo-init] state восстановлен из {cand}", flush=True)
            break
    st = load(args.workdir)
    n_reset = 0
    for j in st["jobs"].values():
        if j["status"] == "running":
            j["status"] = "pending"; n_reset += 1
    if n_reset:
        print(f"[evo-init] зависшие running → pending: {n_reset}", flush=True)
    if not st["jobs"] and not st["history"]:
        qual = lhs_cfgs(QUAL_N)
        queue_stage_jobs(st, "qual", 0, qual, QUAL_STEPS, QUAL_SEEDS)
        print(f"[evo-init] qual: {len(qual)} конфигов @{QUAL_STEPS}", flush=True)
    save(args.workdir, st)
    open(wp(args.workdir, "PREDICTIONS.md"), "w").write(PREDICTIONS)


def cmd_worker(args):
    env = dict(os.environ)
    if args.backend:
        env["LC_BACKEND"] = args.backend
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if args.engine == "torch" or (args.backend or "") == "cupy":
        env.setdefault("LC_SKIP_KERNELS", "1")     # ctypes-ядра x86-специфичны
        env.setdefault("OPENBLAS_NUM_THREADS", "1")  # воркеры не дерутся за BLAS-потоки
        env.setdefault("OMP_NUM_THREADS", "1")
    while True:
        st = load(args.workdir)
        if st["stage"] == "final" and stage_done(gen_jobs(st, "final", 0)):
            print(f"[w{args.id}] финал завершён", flush=True)
            return
        cur = gen_jobs(st, st["stage"], st["gen"])
        if stage_done(cur) and st["stage"] != "final":
            lock = wp(args.workdir, "jobs", "advance.lock")
            try:
                os.mkdir(lock)            # атомарная заявка
            except FileExistsError:
                lock = None               # поколение решает сосед — его лок не трогаем
            if lock is not None:
                try:
                    st = load(args.workdir)
                    cur = gen_jobs(st, st["stage"], st["gen"])
                    if stage_done(cur):
                        box = {}
                        def _adv(st_):
                            box["msg"] = advance(args.workdir, st_)
                        state_update(args.workdir, _adv)
                        print(f"[w{args.id}] {box['msg']}", flush=True)
                finally:
                    _unlock(lock)
        elif stage_done(cur) and st["stage"] == "final":
            return
        claimed = None
        for _attempt in range(10):          # терпим: другой воркер может сейчас advance'нуть
            st = load(args.workdir)
            for j in sorted((j for j in st["jobs"].values() if j["status"] == "pending"),
                            key=lambda j: j["tag"]):
                lock = wp(args.workdir, "jobs", f"{j['tag']}.lock")
                try:
                    os.mkdir(lock)
                    claimed = (j, lock); break
                except FileExistsError:
                    continue
            if claimed or st["stage"] == "final" or all(
                    j["status"] in ("done", "dq") for j in st["jobs"].values()):
                break
            time.sleep(20)
        if claimed is None:
            print(f"[w{args.id}] нет доступных джоб; выход", flush=True)
            return
        job, lock = claimed
        key = [k for k, v in st["jobs"].items() if v["tag"] == job["tag"]][0]
        def _run(st_, key=key):
            st_["jobs"][key]["status"] = "running"
        state_update(args.workdir, _run)
        cmd = cfg_cli(args.engine, job["cfg"], job["steps"], job["eval_every"], job["seed"],
                      job["tag"], args.data, saveckpt=(1 if st_stage_is_final(job) else 0))
        print(f"[w{args.id}] СТАРТ {key} (steps={job['steps']} seed={job['seed']} cfg={job['cfg']})", flush=True)
        t0 = time.time()
        logp = wp(args.workdir, "jobs", f"{key}.log")
        box = {"ch": None}

        def _bail(signum=None, _frame=None):
            """SIGTERM/SIGINT: гасим тренера (иначе сирота держит GPU), джоба → pending.
            Без этого воркер, убитый по таймауту сессии/монитора, оставлял прогон running навсегда."""
            ch = box["ch"]
            if ch is not None and ch.poll() is None:
                try:
                    ch.terminate()
                    for _ in range(50):
                        if ch.poll() is not None:
                            break
                        time.sleep(0.1)
                    if ch.poll() is None:
                        ch.kill()
                except Exception:
                    pass
            def _rev(st_):
                if key in st_["jobs"]:
                    st_["jobs"][key]["status"] = "pending"
            try:
                state_update(args.workdir, _rev)
            finally:
                _unlock(lock)
            print(f"[w{args.id}] {'сигнал ' + str(signum) if signum else 'стоп'} → {key} в pending", flush=True)
            sys.exit(143)

        old_h = [signal.signal(signal.SIGTERM, _bail), signal.signal(signal.SIGINT, _bail)]
        try:
            with open(logp, "wb") as lf:
                ch = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
                box["ch"] = ch
                try:
                    rc = ch.wait(timeout=args.job_timeout)
                except subprocess.TimeoutExpired:
                    _bail()
                    rc = -9
        except Exception as e:
            rc = -8
            try:
                open(logp, "ab").write(f"\n[worker] запуск упал: {type(e).__name__}: {e}\n".encode())
            except Exception:
                pass
        finally:
            signal.signal(signal.SIGTERM, old_h[0]); signal.signal(signal.SIGINT, old_h[1])
            box["ch"] = None
        status, ppl, slope, dq = parse_result(job)
        if rc != 0:
            tail = ""
            try:
                tail = "".join(open(logp, errors="replace").readlines()[-4:])
            except Exception:
                pass
            dq = ((dq + " | ") if dq else "") + f"rc={rc}" + (f" | {tail.strip()[:250]}" if tail else "")
            status = "dq"
        el = round(time.time() - t0, 1)
        def _fin(st_, key=key, status=status, ppl=ppl, slope=slope, dq=dq, wid=args.id, el=el):
            st_["jobs"][key].update(status=status, val_ppl=ppl, slope=slope, dq=dq,
                                    worker=wid, elapsed=el)
        state_update(args.workdir, _fin)
        _unlock(lock)
        print(f"[w{args.id}] ФИН {key}: {status} ppl={ppl} {dq}", flush=True)


def st_stage_is_final(job):
    return job["stage"].startswith("final")   # этап называется "final0", не "final"


def cmd_status(args):
    st = load(args.workdir)
    print(f"stage={st['stage']} gen={st['gen']} σ={st['sigma']} популяция={len(st.get('pop') or [])}")
    for h in st["history"]:
        print(f"  история: {h['stage']}{h.get('gen')} " +
              (f"σ={h.get('sigma')} ps={h.get('ps')} топ={h.get('top3')}" if h["stage"] == "evo" else ""))
    for stage in ("qual", "evo", "final"):
        for g in (0, 1, 2, 3):
            js = [j for j in st["jobs"].values() if j["stage"] == stage and j["gen"] == g]
            if js:
                done = sum(1 for j in js if j["status"] in ("done", "dq"))
                print(f"  {stage}{g}: {done}/{len(js)}")


def cmd_summary(args):
    st = load(args.workdir)
    L = ["# EVO-ТУРНИР LeanCore — итоги", "",
         f"поколений сыграно: {sum(1 for h in st['history'] if h['stage'] == 'evo')}, финальный σ={st['sigma']}",
         f"HOF: {json.dumps(st['hof'], ensure_ascii=False)}", ""]
    for h in st["history"]:
        if h["stage"] == "qual":
            L += [f"## Квалификация @{QUAL_STEPS}ш (сид {QUAL_SEEDS})", "| cfg | PPL |", "|---|---|"]
            L += [f"| {t['cfg']} | {t['ppl']} |" for t in h["table"]]
        else:
            L += [f"## Поколение {h['gen']} @{EVO_STEPS}ш (сиды {EVO_SEEDS}) — σ={h['sigma']}, ps={h['ps']}",
                  "| cfg | PPL | ΔPPL к CTRL | происхождение |", "|---|---|---|---|"]
            L += [f"| {t['cfg']} | {t['ppl']} | {t['dctrl']} | {t['origin']} |" for t in h["table"]]
            L += [f"топ-3: {h['top3']}"]
        L.append("")
    fin = [j for j in st["jobs"].values() if j["stage"] == "final"]
    if fin:
        L += ["## Финал @1500 × сиды [1,42,7,99]", "| cfg | seed | PPL |", "|---|---|---|"]
        for j in sorted([j for j in fin if j["status"] == "done"],
                        key=lambda j: (j["cfg"]["name"], j["seed"])):
            c = j["cfg"]
            L.append(f"| {j['cfg']['name']} | {j['seed']} | {j['val_ppl']} |"
                     f"  (mulr={c['mulr']}, ssfull={c['ssfull']}, ssk={c['ssk']}, "
                     f"ssalpha={c['ssalpha']}, lr={c['lr']}, {c.get('origin','')})")
        L.append("")
    L += ["## Контроль предсказаний (E1–E5 — см. PREDICTIONS.md) — заполняется агентом проекта"]
    out = "\n".join(L) + "\n"
    open(wp(args.workdir, "EVO_SUMMARY.md"), "w").write(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "worker", "status", "summary"):
        sp = sub.add_parser(name)
        sp.add_argument("--workdir", default="evo_work")
        if name in ("init", "worker"):
            sp.add_argument("--data", default="data/prep")
        if name == "worker":
            sp.add_argument("--id", type=int, default=0)
            sp.add_argument("--engine", choices=["torch", "numpy"], default="numpy")
            sp.add_argument("--backend", default=None)
            sp.add_argument("--cuda", default=None)
            sp.add_argument("--job_timeout", type=int, default=6 * 3600)
    args = ap.parse_args()
    os.chdir(HERE)
    {"init": cmd_init, "worker": cmd_worker, "status": cmd_status, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
