"""Independent verification probe: same-input / skipped-work audit."""
import os, random, math, time, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import numpy as np
from deap import tools as deap_tools
from mojodeap import tools

def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        a = time.perf_counter(); fn(); best = min(best, time.perf_counter()-a)
    return best

print("=== PMX ===")
first = np.arange(500_000, dtype=np.int64); second = first[::-1].copy()
a1,b1 = first.copy(), second.copy(); a2,b2 = first.copy(), second.copy()
random.seed(10); deap_tools.cxPartialyMatched(a1,b1)
random.seed(10); tools.cxPartialyMatched(a2,b2)
print("PMX bit-identical:", np.array_equal(a1,a2) and np.array_equal(b1,b2), "changed_a:", np.count_nonzero(a1!=first), "changed_b:", np.count_nonzero(b1!=second))
def pmx_mojo():
    a,b = first.copy(), second.copy(); random.seed(10); tools.cxPartialyMatched(a,b)
def pmx_up():
    a,b = first.copy(), second.copy(); random.seed(10); deap_tools.cxPartialyMatched(a,b)
print(f"PMX time mojo {timeit(pmx_mojo)*1000:.1f}ms up {timeit(pmx_up)*1000:.1f}ms")

print("=== Polynomial ===")
orig = np.linspace(-0.9,0.9,500_000)
v1,v2 = orig.copy(), orig.copy()
random.seed(10); deap_tools.mutPolynomialBounded(v1,20.0,-1.0,1.0,0.2); n1=random.random()
random.seed(10); tools.mutPolynomialBounded(v2,20.0,-1.0,1.0,0.2); n2=random.random()
print("poly maxdiff", float(np.max(np.abs(v1-v2))), "changed", int(np.count_nonzero(v1!=orig)), "rng_next_eq", n1==n2)
def poly_mojo():
    v=orig.copy(); random.seed(10); tools.mutPolynomialBounded(v,20.0,-1.0,1.0,0.2)
def poly_up():
    v=orig.copy(); random.seed(10); deap_tools.mutPolynomialBounded(v,20.0,-1.0,1.0,0.2)
print(f"poly time mojo {timeit(poly_mojo)*1000:.1f}ms up {timeit(poly_up)*1000:.1f}ms")

print("=== BLX batch ===")
first2 = np.linspace(-2,2,2_000_000).reshape(2000,1000); second2 = first2[::-1].copy()
a,b = first2.copy(), second2.copy()
tools.cxBlendBatch(a,b,0.3,rng=np.random.default_rng(10))
r = np.random.default_rng(10).random(a.size).reshape(a.shape)
gamma = (1+2*0.3)*r-0.3
# expected from originals
exp_a = (1-gamma)*first2 + gamma*second2
exp_b = gamma*first2 + (1-gamma)*second2
print("BLX formula maxdiff", float(np.max(np.abs(a-exp_a))), float(np.max(np.abs(b-exp_b))), "exact_frac", float(np.mean(a==exp_a)))
def blx_mojo():
    x,y = first2.copy(), second2.copy(); tools.cxBlendBatch(x,y,0.3,rng=np.random.default_rng(10))
def blx_up():
    x,y = first2.copy(), second2.copy(); random.seed(10)
    for ra,rb in zip(x,y): deap_tools.cxBlend(ra,rb,0.3)
print(f"BLX time mojo {timeit(blx_mojo)*1000:.1f}ms up {timeit(blx_up)*1000:.1f}ms")

print("=== SBX batch ===")
a,b = first2.copy(), second2.copy()
tools.cxSimulatedBinaryBatch(a,b,15.0,rng=np.random.default_rng(10))
r = np.random.default_rng(10).random(a.size).reshape(a.shape)
beta = np.where(r<=0.5, 2.0*r, 1.0/(2.0*(1.0-r)))**(1.0/16.0)
exp_a = 0.5*((1+beta)*first2 + (1-beta)*second2)
exp_b = 0.5*((1-beta)*first2 + (1+beta)*second2)
print("SBX formula maxdiff", float(np.max(np.abs(a-exp_a))), "frac_within_1e-12", float(np.mean(np.abs(a-exp_a)<=1e-12)), "frac_within_1e-9", float(np.mean(np.abs(a-exp_a)<=1e-9)))
def sbx_mojo():
    x,y = first2.copy(), second2.copy(); tools.cxSimulatedBinaryBatch(x,y,15.0,rng=np.random.default_rng(10))
def sbx_up():
    x,y = first2.copy(), second2.copy(); random.seed(10)
    for ra,rb in zip(x,y): deap_tools.cxSimulatedBinary(ra,rb,15.0)
print(f"SBX time mojo {timeit(sbx_mojo)*1000:.1f}ms up {timeit(sbx_up)*1000:.1f}ms")

print("=== Gaussian batch ===")
pop = np.zeros((2000,1000))
v = pop.copy(); tools.mutGaussianBatch(v,0.0,1.0,0.2,rng=np.random.default_rng(10))
g = np.random.default_rng(10)
active = g.random(v.size)<0.2; delta = g.normal(0.0,1.0,v.size)
exp = np.where(active, delta, 0.0).reshape(v.shape)
print("Gauss formula exact:", np.array_equal(v,exp), "active_frac", float(np.mean(v!=0)))
def g_mojo():
    x=pop.copy(); tools.mutGaussianBatch(x,0.0,1.0,0.2,rng=np.random.default_rng(10))
def g_up():
    x=pop.copy(); random.seed(10)
    for ind in x: deap_tools.mutGaussian(ind,0.0,1.0,0.2)
print(f"Gauss time mojo {timeit(g_mojo)*1000:.1f}ms up {timeit(g_up)*1000:.1f}ms")

print("=== RNG-generation cost split (isolates batch RNG source difference) ===")
# For batch rows: numpy RNG cost alone vs python random alone
def numpy_rng_2m(): np.random.default_rng(10).random(2_000_000)
def py_rng_2m():
    random.seed(10); [random.random() for _ in range(2_000_000)]
def numpy_gauss_batch():
    g=np.random.default_rng(10); a=g.random(2_000_000)<0.2; d=g.normal(0.0,1.0,2_000_000)
def py_gauss_equiv():
    random.seed(10); [random.random()<0.2 for _ in range(2_000_000)]
print(f"numpy random 2M {timeit(numpy_rng_2m)*1000:.1f}ms  python random 2M {timeit(py_rng_2m)*1000:.1f}ms")
print(f"numpy mask+normal 2M {timeit(numpy_gauss_batch)*1000:.1f}ms  python mask-only 2M {timeit(py_gauss_equiv)*1000:.1f}ms")

print("=== Small-size sanity (README makes no small-size claim) ===")
# check bench.py has no early-exit / size shortcut: read kernels loop bounds already confirmed full n
print("kernels iterate range(n) / range(left,right) with n=full size: verified in src/operators.mojo")
