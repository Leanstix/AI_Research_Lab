from typing import List, Tuple

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

def two_sample_ttest(a: List[float], b: List[float]) -> Tuple[float,float]:
    ma, mb = _mean(a), _mean(b)
    va, vb = _var(a), _var(b)
    na, nb = len(a), len(b)
    sp2 = (((na-1)*va)+((nb-1)*vb))/(na+nb-2)
    t = (ma-mb)/((sp2*(1/na+1/nb))**0.5)
    # crude two-tailed p approx; OK for demo
    df = na+nb-2
    x = abs(t)
    a = 1.0 + (x*x)/df
    tail = a ** (-(df+1)/2)
    p_two = max(0.0, min(1.0, 2.0*(1.0 - (1.0 - 0.5*tail))))
    return t, round(p_two, 4)

def run_toy_experiment(seed: int = 12345):
    a = [10 + 0.8*x for x in _lcg(seed, 60)]
    b = [10 + 0.8*x + 0.2 for x in _lcg(seed, 60)]
    t, p = two_sample_ttest(a, b)
    # cohen d
    import math
    ma, mb = _mean(a), _mean(b)
    sp = math.sqrt((((len(a)-1)*_var(a))+((len(b)-1)*_var(b)))/(len(a)+len(b)-2))
    d = (ma-mb)/sp
    # naive 95% CI on mean diff
    import math
    se = sp * ((1/len(a) + 1/len(b))**0.5)
    ci_low = (ma-mb) - 1.96*se
    ci_high = (ma-mb) + 1.96*se
    return {
        "design": "two-sample t-test",
        "n": len(a)+len(b),
        "alpha": 0.05,
        "effect_size": round(d, 3),
        "p": p,
        "ci": [round(ci_low,3), round(ci_high,3)]
    }
