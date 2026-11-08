#!/usr/bin/env python3
"""cost_model.py -- order-of-magnitude cost model for architecture screening.

Назначение: перебрать тысячи вариантов архитектуры *на бумаге* за секунды,
чтобы на GPU уходили только выжившие. Это screening-инструмент, а не измерение.

Модель (roofline для декодирования одного токена):

    active_params      = params - (неиспользуемые при инференсе веса)
    weight_bytes       = active_params * bits_w / 8
    kv_per_token_bytes = full_frac * n_layers * 2 * n_kv_heads * head_dim * bits_kv / 8
    state_bytes        = (1 - full_frac) * n_layers * state_size * bits_kv / 8   [фиксировано!]
    flops(token)       = 2 * active_params + full_frac * n_layers * 4 * d_model * T
    bytes(token)       = weight_bytes + full_frac * kv_per_token * T + state_bytes
    latency            = max(bytes / BW, flops / peak) / efficiency + overhead
    energy             = bytes * dram_pJ_per_byte + flops * mac_pJ

Константы -- публичные порядковые оценки (энергия доступа к DRAM и MAC
в духе Horowitz; BW/TOPS по классам устройств). Они НЕ откалиброваны под
конкретный телефон, поэтому модель годится для сравнения вариантов между
собой, но не для абсолютных обещаний.

Ключевое ограничение модели: в ней нет качества. Она умеет отвечать только
"что дешевле при равном числе параметров", а не "что умнее". Вторая половина
(качество / scaling law) добавляется отдельно -- см. docs/00-research-framework.md.

Использование:
    python3 tools/cost_model.py table              # сравнение при ~1B параметров
    python3 tools/cost_model.py sweep              # перебор пространства -> CSV
    python3 tools/cost_model.py table --device laptop
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Устройства
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Device:
    name: str
    bw_gbs: float          # ГБ/с, доступная полоса памяти
    peak_tops: float       # TOPS (int8), реальная для класса устройства
    dram_pj_per_byte: float
    mac_pj: float
    overhead_ms: float     # запуск кернелов, синхронизация, оверхед рантайма
    ram_gb: float          # бюджет памяти под модель + состояние

    def __str__(self) -> str:
        return (f"{self.name}: {self.bw_gbs:.0f} GB/s, {self.peak_tops:.0f} TOPS, "
                f"RAM {self.ram_gb:.1f} GB")


PHONE = Device("phone", bw_gbs=60, peak_tops=35, dram_pj_per_byte=20,
               mac_pj=0.5, overhead_ms=6.0, ram_gb=2.0)
LAPTOP = Device("laptop", bw_gbs=200, peak_tops=100, dram_pj_per_byte=20,
                mac_pj=0.5, overhead_ms=2.0, ram_gb=8.0)
H100 = Device("h100", bw_gbs=3350, peak_tops=1000, dram_pj_per_byte=20,
              mac_pj=0.3, overhead_ms=0.3, ram_gb=80.0)

DEVICES = {d.name: d for d in (PHONE, LAPTOP, H100)}


# ----------------------------------------------------------------------------
# Архитектура
# ----------------------------------------------------------------------------

MIXERS = ("full", "gqa", "hybrid", "state")

# Доля слоёв, у которых есть KV-кэш (и растёт с контекстом).
# hybrid 3:1 -- мейнстрим 2025-2026 (Gated DeltaNet + gated attention).
FULL_FRACTION = {"full": 1.0, "gqa": 1.0, "hybrid": 0.25, "state": 0.0}

# Множитель ширины state для линейных слоёв (фиксированное состояние).
STATE_FACTOR = {"full": 0.0, "gqa": 0.0, "hybrid": 1.0, "state": 1.0}


@dataclass
class Arch:
    name: str
    d_model: int
    n_layers: int
    vocab: int = 32000
    ff_mult: float = 4.0          # SwiGLU: 3 * d * d_ff параметров
    mixer: str = "gqa"
    n_heads: int = 16
    n_kv_heads: int = 4           # GQA
    head_dim: int = 64
    state_dim: int = 64           # размерность состояния на голову (linear attn / SSM)
    bits_w: float = 4.0
    bits_kv: float = 8.0
    engram_gb: float = 0.0        # внешняя память O(1); читается частично, не целиком

    # --- параметры ---------------------------------------------------------
    @property
    def d_ff(self) -> int:
        return int(self.ff_mult * self.d_model)

    def mixer_params(self) -> float:
        # q,k,v,o + гейты. Для сравнения держим ~равным между миксерами,
        # чтобы изолировать эффект памяти, а не числа параметров.
        d = self.d_model
        base = 4.0 * d * d
        if self.mixer in ("hybrid", "state"):
            base += 2.0 * d * self.state_dim      # проекции в состояние
        return base

    def mlp_params(self) -> float:
        return 3.0 * self.d_model * self.d_ff     # SwiGLU

    def params(self) -> float:
        emb = self.vocab * self.d_model
        per_layer = self.mixer_params() + self.mlp_params()
        return emb + self.n_layers * per_layer

    # --- память ------------------------------------------------------------
    def weight_bytes(self) -> float:
        return self.params() * self.bits_w / 8.0

    @property
    def n_kv_effective(self) -> int:
        """У full-внимания кэшируются все головы, у GQA/hybrid -- только группы."""
        return self.n_heads if self.mixer == "full" else self.n_kv_heads

    def kv_per_token(self) -> float:
        frac = FULL_FRACTION[self.mixer]
        return (frac * self.n_layers * 2 * self.n_kv_effective * self.head_dim
                * self.bits_kv / 8.0)

    def state_bytes(self) -> float:
        f = STATE_FACTOR[self.mixer]
        per_layer = self.n_heads * self.head_dim * self.state_dim
        return f * self.n_layers * per_layer * self.bits_kv / 8.0

    def engram_bytes(self) -> float:
        return self.engram_gb * 1e9

    def working_set(self, ctx: int) -> float:
        """Сколько реально надо держать в RAM при данном контексте."""
        return self.weight_bytes() + self.kv_per_token() * ctx + self.state_bytes()

    # --- вычисления --------------------------------------------------------
    def flops_decode(self, ctx: int) -> float:
        frac = FULL_FRACTION[self.mixer]
        quad = frac * self.n_layers * 4.0 * self.d_model * ctx
        return 2.0 * self.params() + quad

    def bytes_decode(self, ctx: int) -> float:
        frac = FULL_FRACTION[self.mixer]
        return self.weight_bytes() + frac * self.kv_per_token() * ctx + self.state_bytes()

    # --- метрики -----------------------------------------------------------
    def latency_ms(self, ctx: int, dev: Device, efficiency: float = 0.5) -> float:
        t_bytes = self.bytes_decode(ctx) / (dev.bw_gbs * 1e9)
        t_flops = self.flops_decode(ctx) / (dev.peak_tops * 1e12 * efficiency)
        return 1000.0 * max(t_bytes, t_flops) + dev.overhead_ms

    def energy_mj(self, ctx: int, dev: Device) -> float:
        pj = (self.bytes_decode(ctx) * dev.dram_pj_per_byte
              + self.flops_decode(ctx) * dev.mac_pj)
        return pj / 1e9

    def report(self, ctx: int, dev: Device) -> dict:
        lat = self.latency_ms(ctx, dev)
        return {
            "arch": self.name,
            "mixer": self.mixer,
            "d_model": self.d_model,
            "layers": self.n_layers,
            "bits_w": self.bits_w,
            "params_M": self.params() / 1e6,
            "weights_MB": self.weight_bytes() / 1e6,
            "kv_tok_KB": self.kv_per_token() / 1e3,
            "state_MB": self.state_bytes() / 1e6,
            "ctx": ctx,
            "working_set_MB": self.working_set(ctx) / 1e6,
            "latency_ms": lat,
            "tok_per_s": 1000.0 / lat if lat > 0 else math.inf,
            "mJ_per_tok": self.energy_mj(ctx, dev),
            "fits": self.working_set(ctx) < dev.ram_gb * 1e9,
        }


# ----------------------------------------------------------------------------
# Сравнение при равном числе параметров
# ----------------------------------------------------------------------------

def build_baselines(target_params: float = 1.0e9) -> list[Arch]:
    """Один и тот же масштаб, четыре способа тратить байты."""

    def tuned(mixer: str, name: str) -> Arch:
        # подбираем (ширина, глубина) так, чтобы попасть в target_params
        best: tuple[float, Arch] | None = None
        for d in range(384, 4097, 32):
            n_heads = max(4, d // 128)
            for layers in range(6, 49, 2):
                a = Arch(name, d, layers, n_heads=n_heads,
                         n_kv_heads=max(1, n_heads // 4), head_dim=128,
                         state_dim=128, mixer=mixer)
                err = abs(a.params() - target_params)
                if best is None or err < best[0]:
                    best = (err, a)
        return best[1]

    return [
        tuned("full", "Трансформер (MHA)"),
        tuned("gqa", "Трансформер (GQA)"),
        tuned("hybrid", "Гибрид 3:1 linear:full"),
        tuned("state", "Полностью state-based"),
    ]


def cmd_table(args) -> None:
    dev = DEVICES[args.device]
    archs = build_baselines(args.params * 1e9)

    for ctx in (4096, 32768):
        rows = [a.report(ctx, dev) for a in archs]
        print(f"\n=== {dev} | контекст {ctx} | ~{args.params:g}B параметров ===")
        hdr = (f"{'архитектура':28s} {'пар.':>7s} {'веса':>8s} {'KV/ток':>8s} "
               f"{'state':>7s} {'RAM':>8s} {'мс/ток':>8s} {'ток/с':>7s} {'мДж':>7s}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['arch']:28s} {r['params_M']/1000:6.2f}B "
                  f"{r['weights_MB']:7.0f}M {r['kv_tok_KB']:7.1f}K "
                  f"{r['state_MB']:6.0f}M {r['working_set_MB']:7.0f}M "
                  f"{r['latency_ms']:8.1f} {r['tok_per_s']:7.2f} {r['mJ_per_tok']:7.1f}")

    # эффект битности -- самый недооценённый рычаг
    print(f"\n=== Биты против байтов (гибрид 3:1, {dev}) ===")
    base = archs[2]
    print(f"{'bits_w':>7s} {'bits_kv':>8s} {'веса,MB':>9s} {'мс/ток':>8s} {'ток/с':>7s} {'мДж':>7s}")
    for bw, bkv in ((16, 16), (8, 8), (4, 8), (4, 4), (2, 4), (2, 2)):
        a = Arch(base.name, base.d_model, base.n_layers,
                 n_heads=base.n_heads, n_kv_heads=base.n_kv_heads,
                 head_dim=base.head_dim, state_dim=base.state_dim,
                 mixer="hybrid", bits_w=bw, bits_kv=bkv)
        r = a.report(8192, dev)
        print(f"{bw:7.0f} {bkv:8.0f} {r['weights_MB']:9.0f} "
              f"{r['latency_ms']:8.1f} {r['tok_per_s']:7.2f} {r['mJ_per_tok']:7.1f}")


# ----------------------------------------------------------------------------
# Перебор пространства
# ----------------------------------------------------------------------------

def cmd_sweep(args) -> None:
    dev = DEVICES[args.device]
    ctx = args.ctx

    grid = itertools.product(
        MIXERS,                      # 4
        (384, 512, 768, 1024, 1536, 2048),   # 6
        (8, 12, 16, 24, 32),         # 5
        (2, 4, 8),                   # 3
        (0.0, 0.5, 2.0),             # engram, GB -- 3
    )
    rows = []
    for mixer, d, layers, bits, engram in grid:
        a = Arch(f"{mixer}-d{d}-L{layers}-w{bits:g}-e{engram:g}", d, layers,
                 n_heads=max(4, d // 128), n_kv_heads=max(1, max(4, d // 128) // 4),
                 head_dim=128, state_dim=128, mixer=mixer,
                 bits_w=bits, bits_kv=max(4, bits), engram_gb=engram)
        r = a.report(ctx, dev)
        r["engram_GB"] = engram
        rows.append(r)

    total = len(rows)
    fits = [r for r in rows if r["fits"]]
    print(f"перебрано вариантов: {total}, влезает в {dev.ram_gb:g} ГБ: {len(fits)}")

    # Парето по (память, мДж/ток) при нижней границе по числу параметров:
    # иначе побеждает самая маленькая модель, и сравнение бессмысленно.
    cands = [r for r in fits if r["params_M"] >= args.min_params]
    print(f"кандидатов с >= {args.min_params:g}M параметров: {len(cands)}")

    cands.sort(key=lambda r: r["mJ_per_tok"])
    print(f"\n=== Топ-15 по энергии на токен (контекст {ctx}, {dev}) ===")
    print(f"{'архитектура':30s} {'пар,М':>7s} {'RAM,МБ':>8s} {'мс/ток':>8s} {'ток/с':>7s} {'мДж':>7s}")
    for r in cands[:15]:
        print(f"{r['arch']:30s} {r['params_M']:7.0f} {r['working_set_MB']:8.0f} "
              f"{r['latency_ms']:8.1f} {r['tok_per_s']:7.2f} {r['mJ_per_tok']:7.1f}")

    # Внутри бюджета параметров сравнение честное: качество примерно привязано
    # к числу параметров, поэтому минимум энергии в бюджете -- осмысленный выбор.
    print(f"\n=== Лучший вариант в каждом бюджете параметров ({dev}, ctx={ctx}) ===")
    hdr = (f"{'бюджет':>8s} {'миксер':10s} {'пар,М':>7s} {'w-бит':>6s} {'RAM,МБ':>8s} "
           f"{'ток/с':>7s} {'мДж':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for budget in (150, 300, 600, 1200):
        for mixer in MIXERS:
            pool = [r for r in fits if r["mixer"] == mixer]
            if not pool:
                continue
            near = [r for r in pool if abs(r["params_M"] - budget) <= 0.25 * budget]
            if not near:
                near = sorted(pool, key=lambda r: abs(r["params_M"] - budget))[:10]
            best = min(near, key=lambda r: r["mJ_per_tok"])
            print(f"{budget:7.0f}M {mixer:10s} {best['params_M']:7.0f} "
                  f"{best['bits_w']:6.0f} {best['working_set_MB']:8.0f} "
                  f"{best['tok_per_s']:7.1f} {best['mJ_per_tok']:6.1f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nполная таблица: {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("table", help="сравнение при равном числе параметров")
    t.add_argument("--device", default="phone", choices=list(DEVICES))
    t.add_argument("--params", type=float, default=1.0, help="целевой масштаб, B")
    t.set_defaults(func=cmd_table)

    s = sub.add_parser("sweep", help="перебор пространства -> CSV")
    s.add_argument("--device", default="phone", choices=list(DEVICES))
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--min-params", type=float, default=100.0)
    s.add_argument("--out", default="runs/design_space.csv")
    s.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
