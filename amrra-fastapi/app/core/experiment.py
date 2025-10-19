from typing import List, Tuple
import math
from scipy.stats import t as t_dist

def _mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)

def _var(xs: List[float]) -> float:
    m = _mean(xs)
    return sum((x-m)*(x-m) for x in xs)/(len(xs)-1)

def _lcg(seed: int, n: int) -> List[float]:
    s = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        out.append((s % 1000)/1000.0)
    return out

def two_sample_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Pooled-variance two-sample t-test with proper Student-t p-value.
    Returns (t_stat, p_two_tailed).
    """
    ma, mb = _mean(a), _mean(b)
    va, vb = _var(a), _var(b)
    na, nb = len(a), len(b)
    df = na + nb - 2
    sp2 = (((na - 1) * va) + ((nb - 1) * vb)) / df
    t_stat = (ma - mb) / math.sqrt(sp2 * (1/na + 1/nb))
    p_two = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    # avoid printing 0.0 due to rounding underflows downstream
    p_two = max(p_two, 1e-12)
    return t_stat, p_two

def run_toy_experiment(seed: int = 12345):
    # Use a single LCG stream to avoid perfectly paired series
    r = _lcg(seed, 120)
    a = [10 + 0.8*x for x in r[:60]]
    b = [10 + 0.8*x + 0.2 for x in r[60:]]

    t_stat, p = two_sample_ttest(a, b)

    # Cohen's d (pooled SD)
    ma, mb = _mean(a), _mean(b)
    df = len(a) + len(b) - 2
    sp = math.sqrt((((len(a) - 1) * _var(a)) + ((len(b) - 1) * _var(b))) / df)
    d = (ma - mb) / sp

    # 95% CI for mean difference using Student-t critical value
    se = sp * math.sqrt(1/len(a) + 1/len(b))
    tcrit = t_dist.ppf(0.975, df)
    ci_low = (ma - mb) - tcrit * se
    ci_high = (ma - mb) + tcrit * se

    return {
        "design": "two-sample t-test",
        "n": len(a) + len(b),
        "alpha": 0.05,
        "effect_size": round(d, 3),
        "p": round(float(p), 6),
        "ci": [round(ci_low, 3), round(ci_high, 3)]
    }
