from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import math
import pickle

AA = "ACDEFGHIKLMNPQRSTVWY"

KD = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

RES_MASS = {
    "A": 71.08,
    "C": 103.14,
    "D": 115.09,
    "E": 129.12,
    "F": 147.18,
    "G": 57.05,
    "H": 137.14,
    "I": 113.16,
    "K": 128.17,
    "L": 113.16,
    "M": 131.20,
    "N": 114.11,
    "P": 97.12,
    "Q": 128.13,
    "R": 156.19,
    "S": 87.08,
    "T": 101.11,
    "V": 99.13,
    "W": 186.21,
    "Y": 163.18,
}

PKA = {
    "Cterm": 3.1,
    "Nterm": 8.0,
    "C": 8.5,
    "D": 3.9,
    "E": 4.1,
    "H": 6.5,
    "K": 10.8,
    "R": 12.5,
    "Y": 10.1,
}

HYDROPHOBIC = set(list("AVILMFWY"))


def clean_seq(seq: str) -> str:
    seq = seq.strip().upper().replace(" ", "").replace("\n", "")
    bad = [c for c in seq if c not in AA]
    if bad:
        raise ValueError(
            f"Sequence contains non-standard amino acids: {sorted(set(bad))}"
        )
    return seq


def gravy(seq: str) -> float:
    return sum(KD[a] for a in seq) / max(1, len(seq))


def approx_mw(seq: str) -> float:
    return sum(RES_MASS[a] for a in seq)


def net_charge_at_pH(seq: str, pH: float = 7.4) -> float:
    def pos_frac(pKa):
        return 1.0 / (1.0 + 10 ** (pH - pKa))

    def neg_frac(pKa):
        return 1.0 / (1.0 + 10 ** (pKa - pH))

    counts = {aa: seq.count(aa) for aa in "CDEHKRY"}
    pos = pos_frac(PKA["Nterm"])
    pos += counts["K"] * pos_frac(PKA["K"])
    pos += counts["R"] * pos_frac(PKA["R"])
    pos += counts["H"] * pos_frac(PKA["H"])
    neg = neg_frac(PKA["Cterm"])
    neg += counts["D"] * neg_frac(PKA["D"])
    neg += counts["E"] * neg_frac(PKA["E"])
    neg += counts["C"] * neg_frac(PKA["C"])
    neg += counts["Y"] * neg_frac(PKA["Y"])
    return pos - neg


def estimate_pI(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ch = net_charge_at_pH(seq, pH=mid)
        if ch > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hydrophobic_patch_score(seq: str, window: int = 6) -> float:
    if len(seq) < window:
        return 0.0
    hits = 0
    total = len(seq) - window + 1
    for i in range(total):
        w = seq[i : i + window]
        if sum(1 for a in w if a in HYDROPHOBIC) >= 4:
            hits += 1
    return hits / total


@dataclass
class BBBProxyConfig:
    pI_target: float = 9.6
    pI_tol: float = 1.2
    pI_extreme: float = 11.2
    charge_target: float = 6.0
    charge_tol: float = 4.0
    charge_extreme: float = 14.0
    mw_ref: float = 15000.0
    mw_scale: float = 25000.0
    gravy_ref: float = 0.2
    gravy_scale: float = 0.6
    patch_ref: float = 0.12
    patch_scale: float = 0.18
    w_charge: float = 0.35
    w_pI: float = 0.25
    w_size: float = 0.15
    w_gravy: float = 0.15
    w_patch: float = 0.10


class BBBProxyModel:
    """
    Returns a BBB proxy score in [0, 1] (higher = better predicted BBB favorability)
    and a feature dict for debugging/ranking analysis.
    """

    def __init__(self, config: Optional[BBBProxyConfig] = None):
        self.config = config or BBBProxyConfig()

    def featurize(self, seq: str) -> Dict[str, float]:
        seq = clean_seq(seq)
        f = {}
        f["length"] = float(len(seq))
        f["mw"] = float(approx_mw(seq))
        f["gravy"] = float(gravy(seq))
        f["patch"] = float(hydrophobic_patch_score(seq))
        f["charge_pH7p4"] = float(net_charge_at_pH(seq, 7.4))
        f["pI"] = float(estimate_pI(seq))
        return f

    @staticmethod
    def _soft_band(x: float, target: float, tol: float) -> float:
        d = abs(x - target)
        return math.exp(-((d / max(1e-9, tol)) ** 2))

    @staticmethod
    def _soft_penalty(x: float, ref: float, scale: float) -> float:
        if x <= ref:
            return 0.0
        return 1.0 - math.exp(-(((x - ref) / max(1e-9, scale)) ** 2))

    def score(self, seq: str) -> Tuple[float, Dict[str, float]]:
        c = self.config
        f = self.featurize(seq)
        charge_reward = self._soft_band(
            f["charge_pH7p4"], c.charge_target, c.charge_tol
        )
        pI_reward = self._soft_band(f["pI"], c.pI_target, c.pI_tol)
        charge_ext_pen = self._soft_penalty(
            f["charge_pH7p4"], c.charge_extreme, c.charge_tol
        )
        pI_ext_pen = self._soft_penalty(f["pI"], c.pI_extreme, c.pI_tol)
        size_pen = self._soft_penalty(f["mw"], c.mw_ref, c.mw_scale)
        gravy_pen = self._soft_penalty(f["gravy"], c.gravy_ref, c.gravy_scale)
        patch_pen = self._soft_penalty(f["patch"], c.patch_ref, c.patch_scale)
        raw = (
            c.w_charge * charge_reward
            + c.w_pI * pI_reward
            - c.w_size * size_pen
            - c.w_gravy * gravy_pen
            - c.w_patch * patch_pen
            - 0.10 * charge_ext_pen
            - 0.10 * pI_ext_pen
        )
        score01 = 1.0 / (1.0 + math.exp(-4.0 * (raw - 0.3)))
        f.update(
            {
                "charge_reward": charge_reward,
                "pI_reward": pI_reward,
                "size_pen": size_pen,
                "gravy_pen": gravy_pen,
                "patch_pen": patch_pen,
                "charge_ext_pen": charge_ext_pen,
                "pI_ext_pen": pI_ext_pen,
                "raw": raw,
            }
        )
        return float(score01), f

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BBBProxyModel":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    model = BBBProxyModel()
    test_seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISWNSGSIGYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRYYGSSSWYFDVWGQGTLVTVSS"
    s, feats = model.score(test_seq)
    print("BBB_proxy_score:", s)
    for k in ["pI", "charge_pH7p4", "mw", "gravy", "patch", "raw"]:
        print(k, feats[k])
