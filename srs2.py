import numpy as np
from scipy import optimize

A, B = 0.5, 1.0
E = (1e-3, 1e-5)

f = lambda x: x**2 + x - 1
df = lambda x: 2 * x + 1
phi = lambda x: 1 / (1 + x)
dphi = lambda x: -1 / (1 + x) ** 2

def fmt(v):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.3e}" if 0 < abs(v) < 1e-6 else f"{v:.10f}"
    return str(v)

def print_table(title, headers, rows):
    print(f"\n  {title}")
    rows = [[fmt(v) for v in r] for r in rows]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    print(" | ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("-" * (sum(widths) + 3 * len(widths) - 3))
    for r in rows:
        print(" | ".join(v.rjust(w) for v, w in zip(r, widths)))

def bisection(a, b, e):
    rows = []
    while True:
        c, err = (a + b) / 2, (b - a) / 2
        rows.append([len(rows), a, b, c, f(c), err])
        if err < e: return c, rows
        a, b = (a, c) if f(a) * f(c) < 0 else (c, b)

def chord(a, b, e):
    rows, prev = [], None
    while True:
        x = a - f(a) * (b - a) / (f(b) - f(a))
        err = abs(x - prev) if prev is not None else None
        rows.append([len(rows), a, b, x, err, f(x)])
        if err is not None and err < e: return x, rows
        a, b = (a, x) if f(a) * f(x) < 0 else (x, b)
        prev = x

def iterate(step, x, e):
    rows = []
    while True:
        nx = step(x)
        err = abs(nx - x)
        rows.append([len(rows), x, nx, err, abs(f(nx))])
        x = nx
        if err < e: return x, rows

def lib_bisect(e):
    r = optimize.bisect(f, A, B, xtol=e, full_output=True)
    return r[0], r[1].iterations

def lib_secant(e):
    r = optimize.root_scalar(f, x0=A, x1=B, method="secant", xtol=e)
    return r.root, r.iterations

def lib_newton(x0, e):
    r = optimize.newton(f, x0, fprime=df, tol=e, full_output=True)
    return r[0], r[1].iterations

def lib_fixed_point(x0, e):
    counter = [0]
    def wrapped_phi(x):
        counter[0] += 1
        return phi(x)
    
    root = optimize.fixed_point(wrapped_phi, x0, xtol=e, method="iteration")
    return float(root), counter[0]

summary = []

def run(name, head, e, x0, own, libname, lib):
    x, rows = own
    title = f"{name}, E = {e:.0e}" + ("" if x0 == "-" else f", x0 = {x0}")
    print_table(title, head, rows)
    summary.append([name, f"{e:.0e}", str(x0), x, len(rows), abs(f(x)), libname, *lib])

xs = np.linspace(A, B, 11)
print_table("Табулирование", ["i", "x", "f(x)"], [[i, x, f(x)] for i, x in enumerate(xs)])

for a, b in zip(xs, xs[1:]):
    if f(a) * f(b) < 0:
        print(f"f({a:.2f}) * f({b:.2f}) < 0, корень на [{a:.2f}; {b:.2f}]")
print(f"f({A}) * f({B}) = {f(A) * f(B)} < 0")

h_bis = ["k", "a", "b", "c", "f(c)", "(b-a)/2"]
h_chord = ["k", "a", "b", "x_k", "|x_k-x_k-1|", "f(x_k)"]
h_iter = ["k", "x_k", "x_k+1", "|x_k+1-x_k|", "|f(x_k+1)|"]

for e in E:
    run("Бисекция", h_bis, e, "-", bisection(A, B, e), "bisect", lib_bisect(e))
    
for e in E:
    run("Хорды", h_chord, e, "-", chord(A, B, e), "secant", lib_secant(e))
    
for e in E:
    for x0 in (1.0, 0.5):
        run("Ньютон", h_iter, e, x0, iterate(lambda x: x - f(x) / df(x), x0, e), "newton", lib_newton(x0, e))

g = np.linspace(A, B, 1001)
print(f"\n  Условие сходимости\nmax|phi'(x)| на [{A}; {B}] = {abs(dphi(g)).max():.4f} < 1")
print(f"phi([{A}; {B}]) = [{phi(g).min():.4f}; {phi(g).max():.4f}]")

for e in E:
    for x0 in (0.5, 1.0):
        run("Итерация", h_iter, e, x0, iterate(phi, x0, e), "fixed_point", lib_fixed_point(x0, e))

print_table("Итоговая таблица", 
            ["Метод", "E", "x0", "Корень", "Итер.", "|f(x*)|", "Библ.", "Корень (SciPy)", "Итер. (SciPy)"], 
            summary)

print(f"\nТочный корень: {(5 ** 0.5 - 1) / 2:.15f}")