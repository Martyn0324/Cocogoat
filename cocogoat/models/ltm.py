"""Surprise-gated long-term memory for the latent world model (Task 2.1, from the ViVAE design notes).

The world model sees the last ``W`` frames exactly (sliding-window attention) and a multi-scale EMA
of all past frames.  On top of that, a bank of ``K`` slots keeps whole past frame latents chosen by
*surprise*: the world model's prediction error for that frame, ``||z_hat_t - z_t||^2``.  This is the
analogue of Prioritized Experience Replay: the frames the model could not predict are the ones worth
remembering.  Priorities decay with age (``decay`` per frame), so stale frames are eventually replaced.

The sequence is processed in chunks of ``W`` frames.  Within a chunk the bank is fixed; after the
chunk, its frames are offered to the bank with their surprise and the ``K`` highest (decayed)
priorities are kept.  Nothing in the bank carries gradients: it stores data latents and detached
scores, so every chunk is an independent graph (gradients are accumulated chunk by chunk).

During generation there is no ground truth, so a generated frame's surprise cannot be measured.
Generated frames are offered with a fallback score, the latent change ``||z_t - z_{t-1}||^2``
rescaled by the ratio (mean surprise / mean change) measured on the context frames.

``slots == 0`` gives the baseline: the same chunked training and rollout without a memory bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cocogoat.models.world import WorldModel, ema_memory


class SurpriseMemory:
    """Per-sequence bank of K frame latents with write times and surprise scores (no gradients)."""

    def __init__(self, batch: int, slots: int, shape: tuple, decay: float, device, dtype=torch.float32):
        self.k, self.decay = slots, decay
        self.z = torch.zeros(batch, slots, *shape, device=device, dtype=dtype)
        self.t = torch.zeros(batch, slots, dtype=torch.long, device=device)
        self.score = torch.zeros(batch, slots, device=device)
        self.valid = torch.zeros(batch, slots, dtype=torch.bool, device=device)

    def priority(self, score: torch.Tensor, t: torch.Tensor, valid: torch.Tensor, now: int) -> torch.Tensor:
        p = score * self.decay ** (now - t).clamp(min=0).float()
        return p.masked_fill(~valid, float("-inf"))

    @torch.no_grad()
    def write(self, z: torch.Tensor, times: torch.Tensor, scores: torch.Tensor, now: int) -> None:
        """Offer frames z (B,T,C,H,W) written at ``times`` (T,) with surprise ``scores`` (B,T)."""
        if self.k == 0 or z.shape[1] == 0:
            return
        b, t = z.shape[:2]
        all_z = torch.cat([self.z, z.to(self.z.dtype)], dim=1)
        all_t = torch.cat([self.t, times[None].expand(b, t)], dim=1)
        all_s = torch.cat([self.score, scores.float()], dim=1)
        all_v = torch.cat([self.valid, torch.ones(b, t, dtype=torch.bool, device=z.device)], dim=1)
        keep = self.priority(all_s, all_t, all_v, now).topk(self.k, dim=1).indices
        gather = lambda x: torch.gather(x, 1, keep.view(b, self.k, *([1] * (x.dim() - 2))).expand(  # noqa: E731
            b, self.k, *x.shape[2:]))
        self.z, self.t, self.score, self.valid = gather(all_z), gather(all_t), gather(all_s), gather(all_v)

    def select(self, idx: torch.Tensor) -> "SurpriseMemory":
        out = SurpriseMemory.__new__(SurpriseMemory)
        out.k, out.decay = self.k, self.decay
        out.z, out.t, out.score, out.valid = self.z[idx], self.t[idx], self.score[idx], self.valid[idx]
        return out

    def stats(self, now: int, window: int) -> dict:
        if self.k == 0:
            return {}
        v = self.valid.float()
        age = (now - self.t).float()
        n = v.sum().clamp(min=1)
        return {"ltm_fill": float(v.mean()), "ltm_mean_age": float((age * v).sum() / n),
                "ltm_beyond_window": float(((age > window).float() * v).sum() / n)}


def surprise(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-frame squared prediction error (B,T)."""
    return (pred.float() - target.float()).square().mean(dim=(2, 3, 4))


@dataclass
class LTMConfig:
    slots: int = 8           # 0 = baseline (no memory bank)
    window: int = 16         # frames per chunk = attention window
    decay: float = 0.99      # priority decay per frame of age


class ChunkedWorldModel(nn.Module):
    """Runs a ``WorldModel`` (rotary time positions) chunk by chunk with an optional memory bank.

    Chunk-aligned context, identical in training and generation: the prediction made at frame ``t`` uses
    full causal attention over frames ``[s - W, t]``, where ``s = W * (t // W)`` is the start of t's chunk,
    so it sees between W + 1 and 2W frames (fewer at the start of a sequence).  The memory bank used for
    t's chunk contains exactly the frames ``<= s``; banks are updated only at chunk boundaries.  The
    multi-scale EMA memory always covers the whole history.

    ``rollout(context, steps)`` has the signature of ``WorldModel.rollout``, so the Task 2 evaluation
    (metrics, PNG grids, GIFs) works unchanged.
    """

    def __init__(self, wm: WorldModel, cfg: LTMConfig):
        super().__init__()
        assert wm.cfg.rope_time and not wm.cfg.time_window, "world model must use rope_time and no sliding mask"
        assert (cfg.slots > 0) == bool(wm.cfg.ltm_every), "ltm_every must be set exactly when slots > 0"
        self.wm, self.cfg = wm, cfg
        self.last_stats: dict = {}

    def new_bank(self, z: torch.Tensor) -> SurpriseMemory:
        return SurpriseMemory(z.shape[0], self.cfg.slots, tuple(z.shape[2:]), self.cfg.decay, z.device)

    def ltm_tokens(self, bank: SurpriseMemory, now: int):
        if self.cfg.slots == 0:
            return None
        return self.wm.embed_ltm(bank.z, now - bank.t, bank.valid)

    def step_chunk(self, z_in: torch.Tensor, mem: torch.Tensor, start: int, end: int, bank: SurpriseMemory,
                   cond=None) -> torch.Tensor:
        """Predictions for frames start+1..end (outputs at positions start..end-1), shape (B,end-start,...)."""
        lo = max(0, start - self.cfg.window)
        out = self.wm(z_in[:, lo:end], cond, memory=mem[:, lo:end], ltm=self.ltm_tokens(bank, start))
        return out[:, start - lo:]

    def teacher_forced(self, z_in: torch.Tensor, z: torch.Tensor, backward_fn=None, cond=None):
        """Chunked teacher forcing over a whole clip.

        z_in: (possibly noised) inputs, z: clean targets, both (B,T,C,H,W).  For each chunk, ``backward_fn(loss)``
        is called with that chunk's share of the MSE (graphs are freed chunk by chunk).  Returns
        (mean MSE, {chunk start: bank used by that chunk}).
        """
        T, W = z.shape[1], self.cfg.window
        mem = ema_memory(z_in, self.wm.cfg.ema_decays)
        bank = self.new_bank(z)
        banks, total, n = {}, 0.0, 0
        for start in range(0, T - 1, W):
            end = min(start + W, T - 1)
            banks[start] = bank.select(slice(None))
            pred = self.step_chunk(z_in, mem, start, end, bank, cond)
            tgt = z[:, start + 1:end + 1]
            loss = F.mse_loss(pred.float(), tgt)
            if backward_fn is not None:
                backward_fn(loss * (end - start) / (T - 1))
            total += float(loss.detach()) * (end - start)
            n += end - start
            if self.cfg.slots:  # frames start+1..end become available to the next chunk (whose start is end)
                times = torch.arange(start + 1, end + 1, device=z.device)
                bank.write(z[:, start + 1:end + 1], times, surprise(pred.detach(), tgt), now=end)
        self.last_stats = bank.stats(T - 1, W)
        return total / max(n, 1), banks

    def generate(self, seq: torch.Tensor, steps: int, bank: SurpriseMemory, pending: list | None = None,
                 ratio: torch.Tensor | None = None, cond=None) -> torch.Tensor:
        """Autoregressive generation of ``steps`` latents after ``seq`` (B,T0,...); differentiable if grads are on.

        ``bank`` must hold the frames ``<= W * ((T0 - 1) // W)``; ``pending`` lists (times, latents, scores) of
        later frames, written when their chunk boundary is reached.  Generated frames are offered with the
        fallback score ``ratio * ||z_t - z_{t-1}||^2`` (no writes if ``ratio`` is None).
        """
        W, decays = self.cfg.window, self.wm.cfg.ema_decays
        pending = list(pending or [])
        mem = ema_memory(seq, decays)
        c = seq.shape[2]
        preds = []
        for _ in range(steps):
            t = seq.shape[1] - 1
            s = W * (t // W)
            if self.cfg.slots and pending:
                ready = [p for p in pending if int(p[0][-1]) <= s]
                pending = [p for p in pending if int(p[0][-1]) > s]
                for times, zs, sc in ready:
                    bank.write(zs, times, sc, now=s)
            lo = max(0, s - W)
            out = self.wm(seq[:, lo:t + 1], cond, memory=mem[:, lo:t + 1], ltm=self.ltm_tokens(bank, s))
            nxt = out[:, -1:]
            preds.append(nxt)
            if decays:
                last = mem[:, -1]
                new = torch.cat([a * last[:, k * c:(k + 1) * c] + (1 - a) * nxt[:, 0] for k, a in enumerate(decays)],
                                dim=1)
                mem = torch.cat([mem, new[:, None]], dim=1)
            else:
                mem = torch.cat([mem, nxt[:, :, :0]], dim=1)
            seq = torch.cat([seq, nxt], dim=1)
            if self.cfg.slots and ratio is not None:
                change = (nxt.float() - seq[:, -2:-1].float()).square().mean(dim=(2, 3, 4))
                pending.append((torch.tensor([t + 1], device=seq.device), nxt.detach(),
                                (change * ratio[:, None]).detach()))
        return torch.cat(preds, dim=1)

    @torch.no_grad()
    def rollout(self, context: torch.Tensor, steps: int, cond=None) -> torch.Tensor:
        """Evaluation: score the context frames by true surprise (chunk by chunk), then generate ``steps`` frames."""
        T, W = context.shape[1], self.cfg.window
        bank = self.new_bank(context)
        pending, ratio = [], None
        if self.cfg.slots and T > 1:
            mem = ema_memory(context, self.wm.cfg.ema_decays)
            s_all, c_all = [], []
            for start in range(0, T - 1, W):
                end = min(start + W, T - 1)
                for times, zs, sc in pending:  # same boundary rule as training: frames <= start
                    bank.write(zs, times, sc, now=start)
                pred = self.step_chunk(context, mem, start, end, bank, cond)
                tgt = context[:, start + 1:end + 1]
                s = surprise(pred, tgt)
                pending = [(torch.arange(start + 1, end + 1, device=context.device), tgt, s)]
                s_all.append(s)
                c_all.append(surprise(tgt, context[:, start:end]))
            ratio = torch.cat(s_all, 1).mean(1) / torch.cat(c_all, 1).mean(1).clamp(min=1e-6)
        preds = self.generate(context, steps, bank, pending, ratio, cond)
        self.last_stats = bank.stats(T + steps - 1, W)
        return preds
