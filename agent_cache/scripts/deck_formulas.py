"""Render the formulas of the "Pricing the Room" deck as transparent PNGs.

Each formula is typeset with LaTeX (standalone + amsmath, in the deck's colors), compiled with pdflatex and rasterized
by pdftocairo at SCALE times its size on the deck's 1920 x 1080 slide, so all PNGs share one scale. The DPI tag is
144 * SCALE, so PowerPoint's 16:9 slide (13.33 in wide) inserts each one at its size on the deck.

Files are named <slide number>_<slide>_<what>. The price formula sits on a dark slide in the deck, so it comes twice:
light ink as in the deck, and dark ink for light backgrounds.

Usage: python3 deck_formulas.py [out_dir]   (default ../results/pricing_room_formulas)
"""
import pathlib, subprocess, sys, tempfile

from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results" / "pricing_room_formulas"
SCALE = 4  # PNG pixels per slide pixel
SLIDE_PPI = 144  # 1920 px across PowerPoint's 13.33-in 16:9 slide
PT_PER_EM = 10.0  # standalone's default 10pt

PREAMBLE = r"""\documentclass[preview,border=1pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\definecolor{ink}{HTML}{1B2330}
\definecolor{light}{HTML}{F3F1EA}
\definecolor{muted}{HTML}{636B78}
\definecolor{mutedlight}{HTML}{BCC4D0}
\definecolor{accent}{HTML}{A94E12}
\begin{document}
"""
STRUT = r"\vphantom{(Ap}"
# The braces make the underbrace an ordinary atom, so the minus after it is spaced as binary.
PRICE = (r"\begin{aligned} &\mathbb{E}\Big[\sum_{\text{turns}} \mathrm{TTFT}\Big]"
         r" + \lambda\,\mathbb{E}\Big[\int \sum_{i} \big(h_i + p_i\big)\,dt\Big] - \lambda R T \\[8pt]"
         r" &\qquad = \sum_{i} {\underbrace{\Big(\mathbb{E}[S_i] + \frac{\mathbb{E}[U_i]}{\rho}"
         r" + \lambda\,\mathbb{E}\Big[\int (h_i + p_i)\,dt\Big]\Big)}_{J_i(\lambda)}}"
         r" - \lambda R T + \textcolor{%s}{\text{const}} \end{aligned}")

# file: (tex, color, slide px per em, strut for a uniform height)
F = {
    "03_setting_r": (r"r = bL/\beta", "ink", 28, True),
    "04_resources_N": (r"N", "ink", 30, True),
    "04_resources_C_GPU": (r"C_{\mathrm{GPU}}", "ink", 30, True),
    "04_resources_C_host": (r"C_{\mathrm{host}}", "ink", 30, True),
    "04_resources_R": (r"R = C_{\mathrm{host}} - C_{\mathrm{GPU}}", "ink", 30, True),
    "04_resources_b": (r"b", "ink", 30, True),
    "04_resources_beta": (r"\beta", "ink", 30, True),
    "04_resources_rho": (r"\rho", "ink", 30, True),
    "04_resources_b_over_beta": (r"b/\beta \approx 47\ \text{\textmu s}", "ink", 34, True),
    "04_resources_1_over_rho": (r"1/\rho \approx 110\ \text{\textmu s}", "ink", 34, True),
    "05_session_L": (r"L", "ink", 30, True),
    "05_session_Delta": (r"\Delta", "ink", 30, True),
    "05_session_e_a": (r"e,\quad a = t - e", "ink", 30, True),
    "05_session_k": (r"k", "ink", 30, True),
    "05_session_G": (r"G \sim \Phi_k", "ink", 30, True),
    "05_session_A": (r"A = e + G", "ink", 30, True),
    "05_session_D": (r"D = A + W", "ink", 30, True),
    "05_session_h_p": (r"h,\quad p", "ink", 30, True),
    # Shown shrunk from 482 to 450 px wide on its slide.
    "06_conditional_F": (r"\begin{aligned} F(x \mid t) &= \Pr(A \le t + x \mid A > t)\\[6pt]"
                         r" &= \frac{\Phi_k(a+x) - \Phi_k(a)}{1 - \Phi_k(a)} \end{aligned}", "ink", 34 * 450 / 482, False),
    "06_conditional_F_r_minus_W_hat": (r"F(r - \hat{W} \mid t)", "ink", 34, True),
    "06_conditional_F_r": (r"F(r \mid t)", "ink", 34, True),
    "06_conditional_F_H": (r"F(H \mid t)", "ink", 34, True),
    "07_turncost_TTFT": (r"\mathrm{TTFT} = W + S + \frac{\Delta + U}{\rho}", "ink", 40, False),
    "07_turncost_W": (r"W", "ink", 30, True),
    "07_turncost_S": (r"S = (\ell - D)^{+}", "ink", 30, True),
    "07_turncost_U": (r"U", "ink", 30, True),
    # Shown shrunk from 933 to 793 px wide on its slide.
    "08_problem": (r"\begin{alignedat}{2} \min_{\pi}\quad & \mathbb{E}\Big[\sum_{\text{turns}} \mathrm{TTFT}\Big]"
                   r" & & \\[2pt]"
                   r" \text{s.t.}\quad & \sum_{i} \big(h_i(t) + p_i(t)\big) \le R \quad \forall t & \qquad &"
                   r" \textcolor{muted}{\text{host room}} \\[2pt]"
                   r" & \text{SSD reads} + \text{writes} \le \beta & & \textcolor{muted}{\text{one SSD}} \\[2pt]"
                   r" & \text{one GPU, first come first served} & & \textcolor{muted}{\text{gives the queue wait } W}"
                   r" \end{alignedat}", "ink", 32 * 793 / 933, False),
    # Both shown shrunk from 901 to 766 px wide.
    "09_price_light_ink": (PRICE % "mutedlight", "light", 32 * 766 / 901, False),
    "09_price_dark_ink": (PRICE % "muted", "ink", 32 * 766 / 901, False),
    "10_tradeoff_J": (r"J(\sigma \mid t) = \mathbb{E}[\text{stall}] + \lambda L\,\mathbb{E}[\text{seconds pinned}]",
                      "ink", 36, False),
    # q is the stall risk given the turn has not arrived when the read starts, not a landing quantile.
    "11_rule1_main": (r"\begin{aligned} \frac{\partial J}{\partial \sigma} &= \underbrace{\Pr(\sigma < A \le \sigma + r)}_{\textcolor{muted}{\text{waiting adds stall}}}"
                      r" \;-\; \lambda L\,\underbrace{\Pr(A > \sigma + r)}_{\textcolor{muted}{\text{waiting saves rent}}}\\[6pt]"
                      r" &\propto\; F(r \mid \sigma) - q, \qquad q = \frac{\lambda L}{1 + \lambda L}\\[6pt]"
                      r" &\Longrightarrow\quad \text{start once } F(r \mid t) \ge q \end{aligned}", "ink", 32, False),
    "11_rule1_queue": (r"F(r - \hat{W} \mid t) \ge q", "ink", 30, True),
    "11_rule1_lambda": (r"\lambda = 5 \times 10^{-5}", "accent", 26, True),
    "09_rulesmap_bound": (r"\sum_{i} \min_{\pi_i} J_i(\lambda) - \lambda R T \;\le\; \mathbb{E}\Big[\sum_{\text{turns}} S + \frac{U}{\rho}\Big]",
                          "ink", 30, False),
    "12_rules234_rule2": (r"\text{take room from}\ \operatorname*{arg\,min}_{j}\, F_j(r_j \mid t)", "ink", 28, False),
    "12_rules234_rule3": (r"\text{evict}\ \operatorname*{arg\,min}_{i}\, F_i(H \mid t)\, c_i", "ink", 28, False),
    "12_rules234_rule4": (r"\lambda \leftarrow \lambda\, e^{\eta g}", "ink", 30, True),
    "12_rule3_keep": (r"\text{keep while}\quad \underbrace{\partial_x F_i(0 \mid t)}_{\textcolor{muted}{\text{return rate}}}\; r_i \;\ge\; q_i",
                      "ink", 28, False),
    "12_rule4_g": (r"g = \frac{\text{tokens refused room} - \text{idle room}}{R}", "ink", 28, False),
    "13_methods_A_plus_r": (r"A + r", "ink", 28, True),
    "13_methods_D_minus_1s": (r"D - 1\,\mathrm{s}", "ink", 28, True),
    "13_methods_p50_expiry": (r"p_{50} + \max(60\,\mathrm{s},\, p_{50})", "ink", 28, True),
    "13_methods_Phi_k": (r"\Phi_k", "ink", 28, True),
    "13_methods_q_i": (r"q_i = \frac{\lambda L_i}{1 + \lambda L_i}", "ink", 28, False),
    "13_methods_F_i_c_i": (r"F_i(H \mid t)\, c_i", "ink", 28, True),
    "14_limits_lambda_to_0": (r"\lambda \to 0\colon", "ink", 38, True),
    "14_limits_lambda_to_inf": (r"\lambda \to \infty\colon", "ink", 38, True),
}


def render(name, tex, color, em_px, strut, build):
    body = tex + (STRUT if strut else "")
    (build / f"{name}.tex").write_text(PREAMBLE + "\\color{%s}%%\n$\\displaystyle %s$\n\\end{document}\n" % (color, body))
    run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                         cwd=build, capture_output=True, text=True)
    if run.returncode:
        sys.exit(f"pdflatex failed on {name}:\n" + run.stdout[-1500:])
    png = OUT / f"{name}.png"
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-transp", "-r", f"{72 * SCALE * em_px / PT_PER_EM:.4f}",
                    f"{name}.pdf", str(png.with_suffix(""))], cwd=build, check=True)
    img = Image.open(png)
    img.load()
    img.save(png, dpi=(SLIDE_PPI * SCALE,) * 2)
    return img.size


OUT.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for name, (tex, color, em_px, strut) in F.items():
        w, h = render(name, tex, color, em_px, strut, pathlib.Path(tmp))
        print(f"{name:34s} {w:5d} x {h:4d} px")
print(len(F), "PNGs in", OUT)
