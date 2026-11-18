"""Render the formulas of the "Pricing the Room" deck as transparent PNGs, or as SVGs with --svg.

Each formula is typeset with LaTeX (standalone + amsmath, in the deck's colors) at its size on the deck's 1920 x 1080
slide, so all files share one scale. A PNG is rasterized by pdftocairo at SCALE times that size and tagged
144 * SCALE dpi. An SVG keeps the glyphs as outlines: its viewBox is in slide pixels and its size is in points, half
as many. Either way PowerPoint's 16:9 slide (13.33 in wide) inserts each one at its size on the deck.

Files are named <slide number>_<slide>_<what>, the number being the slide's place in ORDER. The price formula sits on
a dark slide in the deck, so it comes twice: light ink as in the deck, and dark ink for light backgrounds.

Usage: python3 deck_formulas.py [--svg] [out_dir]
  PNG, default ../results/pricing_room_formulas: needs pdflatex or tectonic, pdftocairo and Pillow.
  SVG, default ../results/pricing_the_room: needs pdflatex or tectonic, and PyMuPDF
       (uv run --no-project --with pymupdf python3 deck_formulas.py --svg).
"""
import pathlib, re, shutil, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
SVG = "--svg" in sys.argv[1:]
ARGS = [a for a in sys.argv[1:] if a != "--svg"]
OUT = pathlib.Path(ARGS[0]) if ARGS else HERE.parent / "results" / ("pricing_the_room" if SVG else "pricing_room_formulas")
SCALE = 4  # PNG pixels per slide pixel
SLIDE_PPI = 144  # 1920 px across PowerPoint's 13.33-in 16:9 slide
PT_PER_EM = 10.0  # standalone's default 10pt
# The deck's slide ids in order (project/deck.json, as published 2026-11-10); a formula's number is its slide's place.
ORDER = ("cover setting motivation roadmap case-late case-early case-lost case-nextuse case-gpu b-model turncost problem"
         " b-policy price onegap conditional tradeoff rule1 rule1keys rule2 rule3 rule4 b-proof proof floor replay methods"
         " limits b-better settings better open next resources session refs").split()

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
         r" &\qquad = \sum_{i} {\underbrace{\Big(\mathbb{E}[S_i] + \rho\,\mathbb{E}\big[U_i(2L_i - U_i)\big]"
         r" + \lambda\,\mathbb{E}\Big[\int (h_i + p_i)\,dt\Big]\Big)}_{J_i(\lambda)}}"
         r" - \lambda R T + \textcolor{%s}{\text{const}} \end{aligned}")

# <slide>_<what>: (tex, color, slide px per em, strut for a uniform height)
F = {
    "setting_r": (r"r = bL/\beta", "ink", 28, True),
    "resources_N": (r"N", "ink", 30, True),
    "resources_C_GPU": (r"C_{\mathrm{GPU}}", "ink", 30, True),
    "resources_C_host": (r"C_{\mathrm{host}}", "ink", 30, True),
    "resources_R": (r"R = C_{\mathrm{host}} - C_{\mathrm{GPU}}", "ink", 30, True),
    "resources_b": (r"b", "ink", 30, True),
    "resources_beta": (r"\beta", "ink", 30, True),
    "resources_rho": (r"\rho", "ink", 30, True),
    "resources_b_over_beta": (r"b/\beta \approx 47\ \text{\textmu s}", "ink", 34, True),
    "resources_2_rho_x": (r"2\rho x \approx 6.4\ \text{ns} \times x", "ink", 34, True),
    "session_L": (r"L", "ink", 30, True),
    "session_Delta": (r"\Delta", "ink", 30, True),
    "session_e_a": (r"e,\quad a = t - e", "ink", 30, True),
    "session_k": (r"k", "ink", 30, True),
    "session_G": (r"G \sim \Phi_k", "ink", 30, True),
    "session_A": (r"A = e + G", "ink", 30, True),
    "session_D": (r"D = A + W", "ink", 30, True),
    "session_h_p": (r"h,\quad p", "ink", 30, True),
    # Shown shrunk from 482 to 450 px wide on its slide.
    "conditional_F": (r"\begin{aligned} F(x \mid t) &= \Pr(A \le t + x \mid A > t)\\[6pt]"
                         r" &= \frac{\Phi_k(a+x) - \Phi_k(a)}{1 - \Phi_k(a)} \end{aligned}", "ink", 34 * 450 / 482, False),
    "conditional_F_r_minus_W_hat": (r"F(r - \hat{W} \mid t)", "ink", 34, True),
    "conditional_F_r": (r"F(r \mid t)", "ink", 34, True),
    "conditional_F_H": (r"F(H \mid t)", "ink", 34, True),
    # Prefill from position x0 to x1 takes rho (x1^2 - x0^2): the turn prefills from L - U to L + Delta.
    "turncost_TTFT": (r"\begin{aligned} \mathrm{TTFT} &= W + S + \rho\,\big[(L + \Delta)^2 - (L - U)^2\big] \\[6pt]"
                      r" &= W + S + \underbrace{\rho\,\Delta\,(2L + \Delta)}_{\textcolor{muted}{\text{new tokens}}}"
                      r" + \underbrace{\rho\,U\,(2L - U)}_{\textcolor{muted}{\text{recomputed prefix}}} \end{aligned}",
                      "ink", 34, False),
    "turncost_W": (r"W", "ink", 30, True),
    "turncost_S": (r"S = (\ell - D)^{+}", "ink", 30, True),
    "turncost_U": (r"U", "ink", 30, True),
    # Shown shrunk from 933 to 793 px wide on its slide.
    "problem": (r"\begin{alignedat}{2} \min_{\pi}\quad & \mathbb{E}\Big[\sum_{\text{turns}} \mathrm{TTFT}\Big]"
                   r" & & \\[2pt]"
                   r" \text{s.t.}\quad & \sum_{i} \big(h_i(t) + p_i(t)\big) \le R \quad \forall t & \qquad &"
                   r" \textcolor{muted}{\text{host room}} \\[2pt]"
                   r" & \text{SSD reads} + \text{writes} \le \beta & & \textcolor{muted}{\text{one SSD}} \\[2pt]"
                   r" & \text{one GPU, first come first served} & & \textcolor{muted}{\text{gives the queue wait } W}"
                   r" \end{alignedat}", "ink", 32 * 793 / 933, False),
    # Both at 32 px per em shrunk by 766/901, the scale the deck first showed them at.
    "price_light_ink": (PRICE % "mutedlight", "light", 32 * 766 / 901, False),
    "price_dark_ink": (PRICE % "muted", "ink", 32 * 766 / 901, False),
    "tradeoff_J": (r"J(\sigma \mid t) = \mathbb{E}[\text{stall}] + \lambda L\,\mathbb{E}[\text{seconds pinned}]",
                      "ink", 36, False),
    # q is the stall risk given the turn has not arrived when the read starts, not a landing quantile.
    "rule1_main": (r"\begin{aligned} \frac{\partial J}{\partial \sigma} &= \underbrace{\Pr(\sigma < A \le \sigma + r)}_{\textcolor{muted}{\text{waiting adds stall}}}"
                      r" \;-\; \lambda L\,\underbrace{\Pr(A > \sigma + r)}_{\textcolor{muted}{\text{waiting saves rent}}}\\[6pt]"
                      r" &\propto\; F(r \mid \sigma) - q, \qquad q = \frac{\lambda L}{1 + \lambda L}\\[6pt]"
                      r" &\Longrightarrow\quad \text{start once } F(r \mid t) \ge q \end{aligned}", "ink", 32, False),
    "rule1_queue": (r"F(r - \hat{W} \mid t) \ge q", "ink", 30, True),
    "rule1_lambda": (r"\lambda = 5 \times 10^{-5}", "accent", 26, True),
    "proof_bound": (r"\sum_{i} \min_{\pi_i} J_i(\lambda) - \lambda R T \;\le\; \mathbb{E}\Big[\sum_{\text{turns}} S + \rho\,U(2L - U)\Big]",
                          "ink", 30, False),
    "rule2_argmin": (r"\text{take room from}\ \operatorname*{arg\,min}_{j}\, F_j(r_j \mid t)", "ink", 28, False),
    "rule3_argmin": (r"\text{evict}\ \operatorname*{arg\,min}_{i}\, F_i(H \mid t)\, c_i", "ink", 28, False),
    "rule4_update": (r"\lambda \leftarrow \lambda\, e^{\eta g}", "ink", 30, True),
    "rule3_keep": (r"\text{keep while}\quad \underbrace{\partial_x F_i(0 \mid t)}_{\textcolor{muted}{\text{return rate}}}\; r_i \;\ge\; q_i",
                      "ink", 28, False),
    "rule4_g": (r"g = \frac{\text{tokens refused room} - \text{idle room}}{R}", "ink", 28, False),
    "methods_A_plus_r": (r"A + r", "ink", 28, True),
    "methods_D_minus_1s": (r"D - 1\,\mathrm{s}", "ink", 28, True),
    "methods_p50_expiry": (r"p_{50} + \max(60\,\mathrm{s},\, p_{50})", "ink", 28, True),
    "methods_Phi_k": (r"\Phi_k", "ink", 28, True),
    "methods_q_i": (r"q_i = \frac{\lambda L_i}{1 + \lambda L_i}", "ink", 28, False),
    "methods_F_i_c_i": (r"F_i(H \mid t)\, c_i", "ink", 28, True),
    "limits_lambda_to_0": (r"\lambda \to 0\colon", "ink", 38, True),
    "limits_lambda_to_inf": (r"\lambda \to \infty\colon", "ink", 38, True),
}


def typeset(name, tex, color, strut, build):
    body = tex + (STRUT if strut else "")
    (build / f"{name}.tex").write_text(PREAMBLE + "\\color{%s}%%\n$\\displaystyle %s$\n\\end{document}\n" % (color, body))
    cmd = (["pdflatex", "-interaction=nonstopmode", "-halt-on-error"] if shutil.which("pdflatex")
           else ["tectonic", "--chatter", "minimal"])
    run = subprocess.run(cmd + [f"{name}.tex"], cwd=build, capture_output=True, text=True)
    if run.returncode:
        sys.exit(f"{cmd[0]} failed on {name}:\n" + (run.stdout + run.stderr)[-1500:])


def write_png(name, em_px, build):
    from PIL import Image
    png = OUT / f"{name}.png"
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-transp", "-r", f"{72 * SCALE * em_px / PT_PER_EM:.4f}",
                    f"{name}.pdf", str(png.with_suffix(""))], cwd=build, check=True)
    img = Image.open(png)
    img.load()
    img.save(png, dpi=(SLIDE_PPI * SCALE,) * 2)
    return img.size


def write_svg(name, em_px, build):
    import pymupdf
    page = pymupdf.open(build / f"{name}.pdf")[0]
    px_per_pt = em_px / PT_PER_EM
    w, h = page.rect.width * px_per_pt, page.rect.height * px_per_pt
    svg = page.get_svg_image(matrix=pymupdf.Matrix(px_per_pt, px_per_pt), text_as_path=True)
    # MuPDF writes the size in the viewBox's units, slide pixels here; restate it in points.
    svg, sized = re.subn(r'width="[\d.]+" height="[\d.]+"',
                         f'width="{w * 72 / SLIDE_PPI:.2f}pt" height="{h * 72 / SLIDE_PPI:.2f}pt"', svg, count=1)
    if not sized:
        sys.exit(f"no size on the root of {name}.svg")
    (OUT / f"{name}.svg").write_text(svg)
    return round(w), round(h)


OUT.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    for key, (tex, color, em_px, strut) in F.items():
        name = f"{ORDER.index(key.split('_')[0]) + 1:02d}_{key}"
        typeset(name, tex, color, strut, pathlib.Path(tmp))
        w, h = (write_svg if SVG else write_png)(name, em_px, pathlib.Path(tmp))
        print(f"{name:34s} {w:5d} x {h:4d} px" + (" on the slide" if SVG else ""))
print(len(F), "SVGs" if SVG else "PNGs", "in", OUT)
