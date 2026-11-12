"""Circuit panels drawn with qiskit's own drawer (as in the CS591 HW4 figures),
restyled to the draw.io pastel palette and Times. Output: <name>.pdf next to this script (TrueType fonts embedded)."""
import os, copy
HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RYGate, XGate, ZGate

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # Times-like math
    "pdf.fonttype": 42,                  # embed TrueType, not Type 3 (IEEE PDF check)
})

# draw.io pastel pairs (fill, text)
BLUE, GREEN, YELLOW = "#dae8fc", "#d5e8d4", "#fff2cc"
PURPLE, ORANGE, TEAL, WHITE = "#e1d5e7", "#ffe6cc", "#b0e3e6", "#ffffff"
INK = "#000000"

STYLE = {
    "name": "bw",
    "textcolor": INK, "gatetextcolor": INK, "subtextcolor": "#333333",
    "linecolor": INK, "creglinecolor": "#666666", "edgecolor": INK,
    "gatefacecolor": WHITE, "barrierfacecolor": WHITE, "backgroundcolor": WHITE,
    "fontsize": 13, "subfontsize": 10,
    "displaytext": {"ry": "R_Y", "h": "H", "x": "X", "z": "Z", "measure": "M"},
    "displaycolor": {
        "ry": [BLUE, INK], "h": [WHITE, INK], "x": [WHITE, INK], "z": [ORANGE, INK],
        "measure": [WHITE, INK], "A": [BLUE, INK], "A_dg": [BLUE, INK],
        "S_chi": [ORANGE, INK], "S_0": [PURPLE, INK],
    },
}

def style(**color):
    s = copy.deepcopy(STYLE); s["displaycolor"].update(color); return s

import re
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
GREEK = {"θ": r"\theta", "ϑ": r"\vartheta", "χ": r"\chi"}

def _math(t):
    """'q_A' -> $q_{A}$, 'l1' -> $l_{1}$, 'θ_{T|00}' -> $\theta_{T|00}$ (register and parameter names)."""
    if "$" in t:
        return t
    m = re.fullmatch(r"([A-Za-zθϑχ]+)_?\{?([^{}]+?)\}?", t) if ("_" in t or re.fullmatch(r"[a-z]\d", t)) else None
    if not m:
        return t
    base, sub = m.groups()
    for k, v in GREEK.items():
        base = base.replace(k, v + " ")
    return f"${base}_{{{sub}}}$"

def tidy(fig, drop_indices=False):
    ax = fig.axes[0]
    for t in ax.texts:
        s = t.get_text()
        if drop_indices and re.fullmatch(r"\d+", s):
            t.set_visible(False); continue
        m = re.fullmatch(r"\$\{(.+?)\}\$ (\$\|0\\rangle\$)", s)   # register label "${q\_A}$ $|0>$"
        if m:
            t.set_text(f"${_math(m.group(1).replace(chr(92) + '_', '_')).strip('$')}$ {m.group(2)}")
            continue
        t.set_text(_math(s))
    for p in ax.patches:                  # barrier shading off: plain dashed line
        if p.get_alpha() == 0.6:
            p.set_visible(False)
    for p in ax.patches:                  # textbook target: white disc, black plus
        if isinstance(p, Circle) and p.get_radius() > 0.2 and p.get_facecolor()[:3] == (0, 0, 0):
            p.set_facecolor("white")
    for ln in ax.lines:
        if ln.get_color() in ("#ffffff", "white", "#FFFFFF"):
            ln.set_color(INK)

def save(qc, name, sty=STYLE, drop_indices=False, **kw):
    fig = qc.draw("mpl", style=sty, fold=-1, initial_state=True, cregbundle=False, **kw)
    tidy(fig, drop_indices)
    fig.savefig(os.path.join(HERE, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

def ucry(qc, label, controls, target, n):
    """Uniformly controlled RY written out as one controlled RY per control pattern."""
    for s in range(2 ** n):
        bits = format(s, f"0{n}b")          # bits[i] -> controls[i]
        g = RYGate(Parameter(f"θ_{{{label}|{bits}}}")).control(n, ctrl_state=bits[::-1])
        qc.append(g, controls + [target])

# ---- Fig 2 panel: state preparation A for A,B -> T -> E, plus evidence AND
def fig2():
    q = [QuantumRegister(1, n) for n in ("q_A", "q_B", "q_T", "q_E", "f")]
    qc = QuantumCircuit(*q)
    qA, qB, qT, qE, f = (r[0] for r in q)
    qc.ry(Parameter("θ_A"), qA); qc.ry(Parameter("θ_B"), qB)
    qc.barrier(label=r"$P(T\,|\,A,B)$")
    ucry(qc, "T", [qA, qB], qT, 2)
    qc.barrier(label=r"weighted $P(e\,|\,T)$")
    for t in "01":
        g = RYGate(Parameter(f"ϑ_{{w|{t}}}")).control(1, ctrl_state=t)
        g.name = "ryw"; qc.append(g, [qT, qE])
    qc.barrier(label="evidence AND")
    qc.cx(qE, f)
    save(qc, "c2_state_prep", style(ryw=[YELLOW, INK]), plot_barriers=True)

# ---- Fig 4 panels: one Grover step, then V0 / V1 key gates
def block(name, n, label):
    g = Gate(name, n, []); g.label = label; return g

def fig4():
    lits = [QuantumRegister(1, f"l{i}") for i in range(1, 5)]
    rest = [QuantumRegister(1, f"r{i}") for i in range(1, 3)]
    f = QuantumRegister(1, "f")
    allq = [r[0] for r in lits + rest] + [f[0]]
    n = len(allq)
    qc = QuantumCircuit(*lits, *rest, f)
    qc.append(ZGate(label="$S_\\chi$"), [f[0]])
    qc.append(block("A_dg", n, "A†"), allq)
    qc.append(block("S_0", n, "S₀"), allq)
    qc.append(block("A", n, "A"), allq)
    save(qc, "c4a_grover_step", style(z=[ORANGE, INK]), drop_indices=True)

    for tag, anc in (("v0", False), ("v1", True)):
        regs = lits + rest + [f] + ([QuantumRegister(1, "c")] if anc else [])
        qc = QuantumCircuit(*regs)
        pat = "1011"                                  # l1..l4 literal values
        qc.append(XGate().control(4, ctrl_state=pat[::-1]), allq[:4] + [f[0]])
        qc.barrier()
        qc.append(ZGate().control(n - 1, ctrl_state="0" * (n - 1)), allq)
        save(qc, f"c4_{tag}", style(z=[PURPLE, INK]))

# ---- Fig 5: coherent vs measurement-based uncompute of the flag
def fig5():
    lr = [QuantumRegister(1, f"l{i}") for i in range(1, 4)]; f = QuantumRegister(1, "f")
    l = [r[0] for r in lr]
    pat = "101"                                       # l1=1, l2=0, l3=1
    qc = QuantumCircuit(*lr, f)
    qc.append(ZGate(label="$S_\\chi$"), [f[0]]); qc.barrier()
    qc.append(XGate().control(3, ctrl_state=pat[::-1]), l + [f[0]])
    save(qc, "c5_coherent", style(z=[ORANGE, INK]))

    m = ClassicalRegister(1, "m")
    qc = QuantumCircuit(*lr, f, m)
    qc.append(ZGate(label="$S_\\chi$"), [f[0]]); qc.barrier()
    qc.h(f[0]); qc.measure(f[0], m[0])
    with qc.if_test((m, 1)):
        # phase -1 on the literal pattern: open control on l2, Z target on l3 (=1)
        qc.append(ZGate().control(2, ctrl_state=pat[:2][::-1]), [l[0], l[1], l[2]])
        qc.x(f[0])
    save(qc, "c5_mbu", style(z=[ORANGE, INK]))

if __name__ == "__main__":
    fig2(); fig4(); fig5()
    print(sorted(f for f in os.listdir(HERE) if f.endswith(".pdf")))
