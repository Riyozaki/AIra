"""Two-timescale Markov source (synthetic task with known ground truth tau).

Fast level:  x_{t+1} = (x_t + step_s) mod V, step_s = 1 + s
Slow level:  s switches with prob 1/tau per step (mean dwell tau)
Noise:       with prob p_noise the emission is uniform random (irreducible floor).
"""
import torch


class TwoTimeSource:
    def __init__(self, vocab: int = 24, n_modes: int = 4, tau: float = 10.0,
                 p_noise: float = 0.05, seed: int = 0):
        self.V = vocab
        self.n_modes = n_modes
        self.tau = tau
        self.p_noise = p_noise
        self.g = torch.Generator().manual_seed(seed)

    def batch(self, B: int, T: int):
        """Returns x[B,T+1] inputs, y[B,T] targets, hard[B,T] bool mask of hard targets."""
        g = self.g
        s = torch.randint(0, self.n_modes, (B,), generator=g)
        x = torch.randint(0, self.V, (B, T + 1), generator=g)
        hard = torch.zeros(B, T, dtype=torch.bool)
        p_switch = 1.0 / self.tau
        for t in range(T):
            # switch FIRST: y[t] = x[t+1] is emitted by the (possibly new) mode,
            # so a switch at step t makes target y[t] hard (mode unknown from past context).
            sw = torch.rand(B, generator=g) < p_switch
            hard[:, t] = sw
            new_s = torch.randint(0, self.n_modes - 1, (B,), generator=g)
            new_s = new_s + (new_s >= s).long()
            s = torch.where(sw, new_s, s)
            step = 1 + s  # [B]
            nxt = (x[:, t] + step) % self.V
            noise = torch.rand(B, generator=g) < self.p_noise
            nxt = torch.where(noise, torch.randint(0, self.V, (B,), generator=g), nxt)
            x[:, t + 1] = nxt
        return x[:, :-1].contiguous(), x[:, 1:].contiguous(), hard


# Wolfram rule 110 lookup: code = left*4 + center*2 + right
RULE110 = torch.tensor([0, 1, 1, 1, 0, 1, 1, 0], dtype=torch.long)


class CASource:
    """Cellular automaton rule 110, r evolution steps — true compute-depth task.

    Sequence: [init cells (L tokens in {0,1}), SEP=2, evolved cells (L)]. LM teacher-forced;
    'hard' mask = answer region (cells after SEP). Predicting r CA steps needs ~r composed
    transformations: loops are necessary, not optional.
    """

    def __init__(self, L: int = 32, r: int = 8, vocab: int = 3, seed: int = 0):
        self.L = L
        self.r = r
        self.V = vocab
        self.SEP = vocab - 1
        self.g = torch.Generator().manual_seed(seed)
        self.table = RULE110

    def ca_step(self, s: torch.Tensor) -> torch.Tensor:
        left = torch.roll(s, 1, dims=-1)
        right = torch.roll(s, -1, dims=-1)
        return self.table[left * 4 + s * 2 + right]

    def batch(self, B: int, T: int = 0):
        g = self.g
        init = torch.randint(0, 2, (B, self.L), generator=g)
        state = init
        for _ in range(self.r):
            state = self.ca_step(state)
        sep = torch.full((B, 1), self.SEP, dtype=torch.long)
        seq = torch.cat([init, sep, state], dim=1)  # [B, 2L+1]
        x = seq[:, :-1].contiguous()
        y = seq[:, 1:].contiguous()
        hard = torch.zeros(B, y.shape[1], dtype=torch.bool)
        hard[:, self.L:] = True  # targets in the answer region
        return x, y, hard
