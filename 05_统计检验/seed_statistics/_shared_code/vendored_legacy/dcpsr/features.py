"""Run-level time/frequency-domain features from a raw 1-D signal.

Dataset-independent. Channel names are supplied by the adapter, so the
resulting feature names are generated from the data rather than copied from
the PHM2010 feature set.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def time_domain(x: np.ndarray) -> dict[str, float]:
    x = _finite(x)
    if x.size < 4:
        return {k: np.nan for k in TIME_KEYS}
    ax = np.abs(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    rms = float(np.sqrt(np.mean(x ** 2)))
    absmean = float(ax.mean())
    peak = float(ax.max())
    p2p = float(x.max() - x.min())
    # sqrt-amplitude (root amplitude), used by clearance factor
    sra = float(np.mean(np.sqrt(ax)) ** 2)
    c = x - mean
    denom = std ** 3 if std > EPS else np.nan
    skew = float(np.mean(c ** 3) / denom) if np.isfinite(denom) else np.nan
    denom4 = std ** 4 if std > EPS else np.nan
    kurt = float(np.mean(c ** 4) / denom4) if np.isfinite(denom4) else np.nan
    return dict(
        mean=mean, std=std, rms=rms, abs_mean=absmean, peak=peak, p2p=p2p,
        skewness=skew, kurtosis=kurt,
        crest_factor=peak / (rms + EPS),
        shape_factor=rms / (absmean + EPS),
        impulse_factor=peak / (absmean + EPS),
        clearance_factor=peak / (sra + EPS),
        energy=float(np.sum(x ** 2)),
        variance=float(std ** 2),
        zero_crossing_rate=float(np.mean(np.diff(np.signbit(c)) != 0)),
    )


TIME_KEYS = ["mean", "std", "rms", "abs_mean", "peak", "p2p", "skewness",
             "kurtosis", "crest_factor", "shape_factor", "impulse_factor",
             "clearance_factor", "energy", "variance", "zero_crossing_rate"]


def freq_domain(x: np.ndarray, fs: float, n_bands: int = 5) -> dict[str, float]:
    x = _finite(x)
    if x.size < 16 or not np.isfinite(fs) or fs <= 0:
        keys = FREQ_KEYS + [f"band{i}_energy_ratio" for i in range(n_bands)]
        return {k: np.nan for k in keys}
    x = x - x.mean()
    n = int(x.size)
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft(x * win))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    p = sp ** 2
    tot = float(p.sum()) + EPS
    pn = p / tot
    centroid = float(np.sum(f * pn))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * pn)))
    ent = float(-np.sum(pn * np.log(pn + EPS)) / np.log(len(pn) + EPS))
    k = int(np.argmax(p[1:]) + 1) if len(p) > 1 else 0
    out = dict(
        dominant_freq=float(f[k]),
        dominant_amp=float(sp[k]),
        spectral_centroid=centroid,
        spectral_spread=spread,
        spectral_variance=float(spread ** 2),
        spectral_entropy=ent,
        spectral_rms=float(np.sqrt(np.mean(p))),
        spectral_skewness=float(np.sum(((f - centroid) ** 3) * pn) / (spread ** 3 + EPS)),
        spectral_kurtosis=float(np.sum(((f - centroid) ** 4) * pn) / (spread ** 4 + EPS)),
        spectral_rolloff95=float(f[np.searchsorted(np.cumsum(pn), 0.95)]
                                 if np.cumsum(pn)[-1] >= 0.95 else f[-1]),
        total_power=tot,
    )
    edges = np.linspace(0, f[-1], n_bands + 1)
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1] if i < n_bands - 1 else f <= edges[i + 1])
        out[f"band{i}_energy_ratio"] = float(p[m].sum() / tot)
    return out


FREQ_KEYS = ["dominant_freq", "dominant_amp", "spectral_centroid",
             "spectral_spread", "spectral_variance", "spectral_entropy",
             "spectral_rms", "spectral_skewness", "spectral_kurtosis",
             "spectral_rolloff95", "total_power"]


def channel_features(x: np.ndarray, fs: float, prefix: str,
                     n_bands: int = 5) -> dict[str, float]:
    """All features for one channel, prefixed with the channel name."""
    out = {}
    for k, v in time_domain(x).items():
        out[f"{prefix}__{k}"] = v
    for k, v in freq_domain(x, fs, n_bands=n_bands).items():
        out[f"{prefix}__{k}"] = v
    return out
