"""Аналитическая модель затрат базового подхода (плотный декодер-трансформер + backprop + AdamW).

Код служит только КАЛЬКУЛЯТОРОМ для формул из docs/01_cost_model.md.
Все результаты этого скрипта — ТЕОРЕТИЧЕСКИЕ ОЦЕНКИ, а не измерения.
Допущения перечислены в docs/01_cost_model.md, раздел 0.

Запуск:  python tools/cost_model.py  > results/cost_model_tables.md
"""
from dataclasses import dataclass
import math

# ---------------------------------------------------------------- железо
@dataclass
class HW:
    name: str
    peak_flops: float      # FLOP/s в рабочей точности (dense)
    bw: float              # байт/с, основная память
    mem: float             # байт
    power_w: float         # Вт (TDP платы/ CPU-пакета), для грубой оценки энергии
    note: str

HWS = {
    # Паспортные данные (vendor spec), см. docs/01 §0
    "H100-SXM": HW("H100-SXM bf16", 989e12, 3.35e12, 80e9, 700, "спецификация вендора"),
    "T4":       HW("T4 fp16 tensor", 65e12, 320e9, 16e9, 70, "спецификация вендора (из памяти, не проверено в этой сессии)"),
    # ИЗМЕРЕНО tools/roofline_probe.py на текущей песочнице (2 vCPU, numpy/OpenBLAS fp32)
    "sandbox":  HW("sandbox 2vCPU fp32", 240e9, 23e9, 3e9, 0, "измерено: GEMM fp32 ~240 GFLOP/s, GEMV ~23 GB/s"),
}

# ---------------------------------------------------------------- модель
@dataclass
class Model:
    name: str
    L: int          # слои
    d: int          # ширина
    V: int          # словарь
    kv_ratio: float = 1.0   # n_kv_heads / n_heads (GQA)
    tied: bool = True

    @property
    def n_nonemb(self):            # 12 L d^2: attn 4d^2 + MLP 8d^2 (d_ff=4d или SwiGLU 8/3 d)
        return 12 * self.L * self.d ** 2 * (1 - (1 - self.kv_ratio) * 2 / 12)  # GQA уменьшает K,V проекции

    @property
    def n_emb(self):
        return self.V * self.d * (1 if self.tied else 2)

    @property
    def n_total(self):
        return self.n_nonemb + self.n_emb

    def fwd_flops_per_token(self, T):
        """2*N_nonemb + 2*V*d (логиты) + 2*L*T*d (каузальное внимание, QK^T и AV, среднее по позициям)."""
        return 2 * self.n_nonemb + 2 * self.V * self.d + 2 * self.L * T * self.d

    def attn_fraction(self, T):
        f = self.fwd_flops_per_token(T)
        return 2 * self.L * T * self.d / f

    def kv_bytes_per_token(self, b_kv=2):
        return 2 * self.L * self.d * self.kv_ratio * b_kv


MODELS = [
    Model("GPT-2 small 124M", 12, 768, 50257),
    Model("~1.3B", 24, 2048, 50257),
    Model("~7B (Llama-like)", 32, 4096, 32000, kv_ratio=1.0, tied=False),
    Model("~57B (L=80, d=8192, GQA 1/8, d_ff=8/3·d)", 80, 8192, 128000, kv_ratio=0.125, tied=False),
]

def fmt(x, unit=""):
    for s, k in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= k:
            return f"{x/k:.3g}{s}{unit}"
    return f"{x:.3g}{unit}"

def training_table(T=2048, tokens_per_param=20, mfu=0.40, pue=1.2):
    hw = HWS["H100-SXM"]
    rows = ["| Модель | N (всего) | D=20N токенов | C обучения, FLOP | доля внимания при T | GPU-часы H100 @MFU 40% | Энергия, кВт·ч (700 Вт·PUE 1.2) | Память состояния, ГБ (16 Б/пар.) | Активации, ГБ на 1 последовательность (без recompute) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        N = m.n_total
        D = tokens_per_param * N
        C = 3 * m.fwd_flops_per_token(T) * D
        gpu_h = C / (hw.peak_flops * mfu) / 3600
        energy_kwh = gpu_h * hw.power_w * pue / 1e3
        state_gb = 16 * N / 1e9
        a = max(1, m.d // 128)  # число голов при head_dim=128
        act_gb = T * m.d * m.L * (34 + 5 * a * T / m.d) / 1e9   # Korthikanti et al. 2022, b=1, bf16
        rows.append(f"| {m.name} | {fmt(N)} | {fmt(D)} | {C:.2e} | {m.attn_fraction(T)*100:.1f}% | {gpu_h:,.0f} | {energy_kwh:,.0f} | {state_gb:,.1f} | {act_gb:,.1f} |")
    return "\n".join(rows)

def decode_table(T_ctx=4096, b_w=2, batch=1):
    rows = ["| Модель | Железо | байт/токен (веса+KV) | t_mem, мс | t_compute, мс | Нижняя граница, мс/токен | Утилизация вычислений | Ток/с (верх. граница) |",
            "|---|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for key in ("H100-SXM", "T4", "sandbox"):
            hw = HWS[key]
            bw_bytes = 4 if key == "sandbox" else b_w   # в песочнице измерен только fp32 GEMV
            w_bytes = m.n_total * bw_bytes
            if w_bytes > hw.mem:
                rows.append(f"| {m.name} | {hw.name} | {fmt(w_bytes,'B')} | — | — | не помещается в память | — | — |")
                continue
            kv = m.kv_bytes_per_token(bw_bytes) * T_ctx * batch
            bytes_step = w_bytes + kv
            flops_step = m.fwd_flops_per_token(T_ctx) * batch
            t_mem = bytes_step / hw.bw
            t_cmp = flops_step / hw.peak_flops
            t = max(t_mem, t_cmp)
            util = t_cmp / t
            rows.append(f"| {m.name} | {hw.name} | {fmt(bytes_step,'B')} | {t_mem*1e3:.3g} | {t_cmp*1e3:.3g} | {t*1e3:.3g} | {util*100:.2f}% | {batch/t:,.0f} |")
    return "\n".join(rows)

def batch_sweep(model_idx=2, T_ctx=4096, b_w=2):
    m = MODELS[model_idx]; hw = HWS["H100-SXM"]
    rows = ["| batch B | арифм. интенсивность, FLOP/байт | утилизация H100 (оценка) | KV-кэш, ГБ |", "|---|---|---|---|"]
    for B in (1, 8, 32, 128, 512):
        bytes_step = m.n_total * b_w + m.kv_bytes_per_token() * T_ctx * B
        flops = m.fwd_flops_per_token(T_ctx) * B
        ai = flops / bytes_step
        util = min(1.0, ai / (hw.peak_flops / hw.bw))
        rows.append(f"| {B} | {ai:.1f} | {util*100:.1f}% | {m.kv_bytes_per_token()*T_ctx*B/1e9:.1f} |")
    return f"Модель {m.name}, контекст {T_ctx}, веса bf16. Порог H100 (ridge point) = {hw.peak_flops/hw.bw:.0f} FLOP/байт.\n\n" + "\n".join(rows)

def sandbox_budget(hours=(1, 8, 24), util=0.30, tokens_per_param=20):
    hw = HWS["sandbox"]
    rows = ["| Бюджет | C = F·u·t, FLOP | N* (Chinchilla, D=20N) | D* токенов | Память AdamW fp32 (16 Б/пар.) |", "|---|---|---|---|---|"]
    for h in hours:
        C = hw.peak_flops * util * h * 3600
        N = math.sqrt(C / (6 * tokens_per_param))
        D = tokens_per_param * N
        rows.append(f"| {h} ч | {C:.2e} | {fmt(N)} | {fmt(D)} | {16*N/1e6:.0f} МБ |")
    return "\n".join(rows)

def kaggle_budget(hours=(12, 30), util=0.35, tokens_per_param=20):
    hw = HWS["T4"]
    rows = ["| Бюджет (1×T4) | C, FLOP | N* | D* | Память состояния |", "|---|---|---|---|---|"]
    for h in hours:
        C = hw.peak_flops * util * h * 3600
        N = math.sqrt(C / (6 * tokens_per_param))
        rows.append(f"| {h} ч | {C:.2e} | {fmt(N)} | {fmt(tokens_per_param*N)} | {16*N/1e9:.2f} ГБ |")
    return "\n".join(rows)

if __name__ == "__main__":
    print("## A. Обучение (оценка), T=2048, D=20N, MFU 40%\n")
    print(training_table())
    print("\n## B. Декодирование, batch=1, контекст 4096, веса и KV в bf16 (в песочнице fp32) — оценка по roofline\n")
    print(decode_table())
    print("\n## C. Зависимость утилизации декодирования от batch (оценка)\n")
    print(batch_sweep())
    print("\n## D. Что можно обучить в песочнице (оценка, утилизация 30% от измеренного пика)\n")
    print(sandbox_budget())
    print("\n## E. Что можно обучить на Kaggle T4 (оценка, утилизация 35%)\n")
    print(kaggle_budget())
