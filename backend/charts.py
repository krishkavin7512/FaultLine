"""Every graph from training, drawn from this run's own numbers, with a plain explanation.

Built only from artifacts/report.json, so the graphs can be redrawn without retraining:  python -m backend.charts
"""
import json
import math

from . import config

BLUE, AQUA, ORANGE, VIOLET, AMBER, MAGENTA = "#3987e5", "#199e70", "#e0621f", "#9085e9", "#c98500", "#d55181"
PAL = [BLUE, AQUA, ORANGE, VIOLET, AMBER, MAGENTA]    # validated against the #121417 chart surface, in this order
INK, INK2, MUTED, GRID, AXIS, REF, SURFACE = "#eef0f2", "#c3c8ce", "#878e97", "#1f2328", "#363c44", "#a7aeb6", "#121417"
RISK = [[0, "#1c1f23"], [0.45, "#a84a17"], [1, "#ffc29a"]]          # one hue, dark to light: share of houses collapsed
DIV = [[0, "#3987e5"], [0.5, "#2a2e33"], [1, "#e0621f"]]           # blue = fewer than predicted, orange = more
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
# Colour follows the model everywhere. Only six models get a hue; the rest are grey with a dash pattern.
MC = {"cart": BLUE, "forest": AQUA, "gboost": ORANGE, "gam": VIOLET, "crowd": AMBER, "stack": MAGENTA}
GREY = {"checklist": (REF, "dot"), "bagging": (REF, None), "adaboost": (MUTED, None), "logit": (MUTED, "dash"), "bma": (REF, "dashdot"),
        "shaking": (REF, "dash"), "rule": (INK2, "dot"), "random": (AXIS, "dash")}


def pct(v, d=0):
    return f"{v * 100:.{d}f}%"


def ax(title="", **k):
    return {"title": {"text": title, "font": {"color": MUTED, "size": 12}}, "gridcolor": GRID, "linecolor": AXIS, "zerolinecolor": AXIS, "tickcolor": AXIS,
            "ticks": "outside", "ticklen": 4, "automargin": True, **k}


def lay(x="", y="", xa=None, ya=None, **k):
    return {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"family": FONT, "color": INK2, "size": 12}, "margin": {"l": 60, "r": 18, "t": 26, "b": 48},
            "hoverlabel": {"bgcolor": "#1d2126", "bordercolor": "#454c55", "font": {"family": FONT, "color": INK, "size": 12}},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "bgcolor": "rgba(0,0,0,0)"},
            "xaxis": {**ax(x), **(xa or {})}, "yaxis": {**ax(y), **(ya or {})}, **k}


def style(key):
    if key in MC:
        return MC[key], None
    return GREY.get(key, (REF, None))


def line(x, y, name="", color=BLUE, width=2, dash=None, **k):
    return {"type": "scatter", "mode": "lines", "x": x, "y": y, "name": name, "line": {"color": color, "width": width, **({"dash": dash} if dash else {})}, **k}


def mline(key, x, y, name, **k):
    c, d = style(key)
    return line(x, y, name, c, 2, d, **k)


def dots(x, y, name="", color=BLUE, size=8, **k):
    return {"type": "scatter", "mode": "markers", "x": x, "y": y, "name": name, "marker": {"color": color, "size": size, "line": {"color": SURFACE, "width": 1}}, **k}


def bars(x, y, name="", color=BLUE, **k):
    return {"type": "bar", "x": x, "y": y, "name": name, "marker": {"color": color, "line": {"color": SURFACE, "width": 2}, "cornerradius": 4}, **k}


def hbars(x, y, name="", color=BLUE, **k):
    return bars(x, y, name, color, orientation="h", **k)


def vline(x, color=REF, dash="dash"):
    return {"type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1, "line": {"color": color, "width": 2, "dash": dash}}


def hline(y, color=REF, dash="dash"):
    return {"type": "line", "y0": y, "y1": y, "xref": "paper", "x0": 0, "x1": 1, "line": {"color": color, "width": 2, "dash": dash}}


def note(x, y, text, **k):
    return {"x": x, "y": y, "text": text, "showarrow": False, "font": {"color": INK2, "size": 11}, "bgcolor": "rgba(18,20,23,.8)", **k}


def heat(z, x, y, hover, scale=RISK, zmin=None, zmax=None, text=None, **k):
    t = {"type": "heatmap", "z": z, "x": x, "y": y, "colorscale": scale, "hovertemplate": hover + "<extra></extra>", "xgap": 2, "ygap": 2,
         "colorbar": {"thickness": 10, "outlinewidth": 0, "tickfont": {"color": MUTED}}, **k}
    if zmin is not None:
        t["zmin"], t["zmax"] = zmin, zmax
    if text is not None:
        t["text"], t["texttemplate"], t["textfont"] = text, "%{text}", {"color": INK, "size": 11}
    return t


def G(gid, cat, title, insight, what, how, read, where, data, layout, height=380):
    return {"id": gid, "category": cat, "title": title, "insight": insight, "height": height, "explain": {"what": what, "how": how, "read": read, "where": where},
            "figure": {"data": data, "layout": layout}}


def build_graphs(rep):
    D, S, L = rep["data"], rep["scores"], rep["labels"]
    HL = L[rep["headline"]]
    unseen = " · ".join(D["unseen_districts"])
    out = []
    add = lambda *a, **k: out.append(G(*a, **k))
    short = ["1 · slight", "2 · moderate", "3 · heavy", "4 · very heavy", "5 · destroyed"]

    # ================================================================== DATA
    cnt = D["grade_counts"]
    tot = sum(cnt)
    add("grades", "Data", "How badly 760,000 houses were damaged",
        f"{pct(cnt[4] / tot)} of surveyed houses were destroyed outright (grade 5). Only {pct(cnt[0] / tot)} got away with slight cracks.",
        "The number of houses in each of the five damage grades recorded by Nepal's door-to-door survey after the 25 April 2015 Gorkha earthquake.",
        f"Counted from {tot:,} building records across {D['districts']} districts. Each house was graded by survey engineers from 1 (hairline cracks) to 5 (collapsed).",
        "Taller bar = more houses. The orange bar is the answer every model in this project learns to predict: will this house be destroyed?",
        "Grade 5 is the yes/no target for all the classifiers. The five grades together are the five-way sorting task at the end.",
        [bars(short, cnt, color=[BLUE] * 4 + [ORANGE], hovertemplate="Grade %{x}<br>%{y:,} houses<extra></extra>", showlegend=False)],
        lay("Damage grade", "Houses"))

    ds = sorted(rep["districts"], key=lambda d: -d["rate"])
    tr = [d for d in ds if not d["unseen"]]
    ho = [d for d in ds if d["unseen"]]
    add("districts", "Data", f"Collapse rates differed {ds[0]['rate'] / ds[-1]['rate']:.0f}-fold between districts",
        f"{ds[0]['name']} lost {pct(ds[0]['rate'])} of its houses; {ds[-1]['name']} lost {pct(ds[-1]['rate'])}. The three orange districts were hidden from training.",
        "The share of houses destroyed (grade 5) in each surveyed district.",
        "Destroyed houses divided by surveyed houses, per district. Districts in orange (" + unseen + ") are never shown to any model during training: they are the spatial test.",
        "Longer bar = a district that suffered more. Gorkha held the epicentre, yet several districts to its east, closer to the big aftershock and on steeper ground, lost more.",
        "The orange districts are the main test of the whole project: can models trained on eight districts predict which houses fall in three districts they have never seen?",
        [hbars([d["rate"] for d in tr], [d["name"] for d in tr], "Training districts", BLUE, customdata=[[d["n"], d["shaking"]] for d in tr],
               hovertemplate="%{y}<br>%{x:.1%} destroyed<br>%{customdata[0]:,} houses · mean shaking %{customdata[1]:.2f}<extra></extra>"),
         hbars([d["rate"] for d in ho], [d["name"] for d in ho], "Held-out districts", ORANGE, customdata=[[d["n"], d["shaking"]] for d in ho],
               hovertemplate="%{y}<br>%{x:.1%} destroyed<br>%{customdata[0]:,} houses · mean shaking %{customdata[1]:.2f}<extra></extra>")],
        lay("Share of houses destroyed", "", xa={"tickformat": ".0%"}, ya={"categoryorder": "array", "categoryarray": [d["name"] for d in ds][::-1]}), 420)

    if "wall_rates" in rep:
        wr = sorted(rep["wall_rates"].items(), key=lambda kv: kv[1]["rate"])
        add("walls", "Data", "What the walls were made of mattered most",
            f"{wr[-1][0]} houses were destroyed {pct(wr[-1][1]['rate'])} of the time; {wr[0][0]} only {pct(wr[0][1]['rate'])}.",
            "The share of houses destroyed for each main construction type.",
            "Each house is given one main type, taking the strongest material it has (an engineered concrete frame beats brick, brick in cement beats brick in mud, and so on). Then destroyed / total within each type.",
            "Longer bar = more fragile construction. Hover for how many houses of each type there were.",
            "This is the single strongest fact in the data. It drives the inspector's checklist, the fragility curves and the upgrade what-ifs.",
            [hbars([v["rate"] for _, v in wr], [k for k, _ in wr], color=BLUE, customdata=[v["n"] for _, v in wr], showlegend=False,
                   hovertemplate="%{y}<br>%{x:.1%} destroyed<br>%{customdata:,} houses<extra></extra>")],
            lay("Share of houses destroyed", "", xa={"tickformat": ".0%"}), 400)

    if "floor_rates" in rep:
        fr = rep["floor_rates"]
        add("floors", "Data", "Up to three storeys, taller meant riskier",
            f"One-storey houses were destroyed {pct(fr['rate'][0])} of the time, three-storey houses {pct(fr['rate'][2])}. The few houses with four or more storeys fell far less ({pct(fr['rate'][3])}), because most of them have concrete frames.",
            "The share of houses destroyed, by number of storeys before the earthquake.",
            "Destroyed / total within each storey count; five or more storeys are pooled because they are rare.",
            "Bars rising from one to three storeys mean extra storeys of stone and mud add risk. The drop after that is about what tall houses are built from, not height itself: hover to see how few there are.",
            "Storeys and height enter every model. The GAM shape for storeys shows the same rise after the other facts are held fixed.",
            [bars(fr["floors"], fr["rate"], color=BLUE, customdata=fr["n"], showlegend=False, hovertemplate="%{x} storeys<br>%{y:.1%} destroyed<br>%{customdata:,} houses<extra></extra>")],
            lay("Storeys", "Share destroyed", ya={"tickformat": ".0%"}))

    fg = rep["fragility"]
    if "all_observed" in fg:
        add("shaking", "Data", "Stronger shaking, more collapse",
            "The collapse rate climbs steeply once ward shaking passes about 7.5 on the intensity scale.",
            "The share of houses destroyed in each band of estimated ground shaking.",
            "Every house takes the shaking intensity estimated for its ward (one number per ward, supplied with the dataset). Houses are grouped into bands 0.25 wide and the destroyed share is taken in each band with at least 150 houses.",
            "Left to right is stronger shaking. The curve is a fragility curve for the whole housing stock mixed together.",
            "The shaking estimate is the one fact that is not about the house itself. The fragility curves split this same line by wall type.",
            [line(fg["bands"], fg["all_observed"], "All houses", BLUE, mode="lines+markers", marker={"size": 8, "color": BLUE, "line": {"color": SURFACE, "width": 1}},
                  customdata=fg["all_n"], hovertemplate="Shaking %{x}<br>%{y:.1%} destroyed<br>%{customdata:,} houses<extra></extra>", showlegend=False)],
            lay("Estimated shaking intensity (ward)", "Share destroyed", ya={"tickformat": ".0%"}))

    # ================================================================== ONE MODEL
    C = rep["cart"]
    pth = C["path"]
    chosen, ck = pth[C["chosen"]], C["checklist"]
    small = next(r for r in pth if r["leaves"] <= ck["leaves"])
    add("cart_prune", "One model", "Pruning the tree back until it stops over-fitting",
        f"The full tree had {C['full_leaves']} leaves. Cutting it back to {C['leaves']} leaves gave the best score on validation houses.",
        "Validation error of every tree on the cost-complexity pruning path, from the full tree down to a single leaf.",
        "Grow one big CART tree on 60,000 training houses. Repeatedly collapse the 'weakest link', the branch that buys the least error reduction per leaf. Score each pruned tree by Brier score (mean squared error of the probability) on 20,000 validation houses.",
        "Lower is better. Far right = the full tree, which memorises noise. Far left = too few questions. The marked dot is the tree kept; the grey dot is the 10-leaf inspector's checklist.",
        "The chosen tree is the 'CART' model in every comparison. The small one becomes the checklist an engineer could follow on paper.",
        [line([r["leaves"] for r in pth], [r["val_brier"] for r in pth], "Pruning path", BLUE, mode="lines+markers", marker={"size": 5, "color": BLUE},
              hovertemplate="%{x} leaves<br>Brier %{y:.4f}<extra></extra>", showlegend=False),
         dots([chosen["leaves"]], [chosen["val_brier"]], "Kept: CART", ORANGE, 13, hovertemplate="Kept tree<br>%{x} leaves<br>Brier %{y:.4f}<extra></extra>"),
         dots([small["leaves"]], [small["val_brier"]], "Inspector's checklist", REF, 13, hovertemplate="Checklist<br>%{x} leaves<br>Brier %{y:.4f}<extra></extra>")],
        lay("Leaves in the tree (log scale)", "Validation Brier score", xa={"type": "log"}))

    nodes = {n["id"]: n for n in ck["nodes"]}
    ids, labels, parents, vals, colors, cd = [], [], [], [], [], []
    leafv = {}

    def total(i):
        n = nodes[i]
        if n["leaf"]:
            leafv[i] = max(1, round(n["share"] * 100000))
            return leafv[i]
        leafv[i] = total(n["left"]) + total(n["right"])
        return leafv[i]
    total(0)

    def walk(i, label, parent):
        n = nodes[i]
        ids.append(str(i)); labels.append(label); parents.append(parent); vals.append(leafv[i]); colors.append(n["p"]); cd.append([n["p"], n["share"]])
        if not n["leaf"]:
            walk(n["left"], n["yes"], str(i))
            walk(n["right"], n["no"], str(i))
    walk(0, "All houses", "")
    worst = ck["rules"][0]
    add("checklist", "One model", "The inspector's checklist: a 10-leaf tree",
        f"The riskiest branch ({pct(worst['share'])} of houses) was destroyed {pct(worst['p'])} of the time: " + " → ".join(worst["if"]) + ".",
        "The pruned classification tree small enough to print on one card. Each box is a group of houses; its colour is the share destroyed.",
        "This is the largest tree on the pruning path with at most 10 leaves. Each split was chosen by CART to reduce Gini impurity the most. Box width shows the share of training houses; colour shows the collapse rate (with a little Laplace smoothing).",
        "Read top to bottom. Each step down answers one question about the house. Brighter orange = more houses in that group were destroyed. Click a box to zoom into its branch.",
        "This is CART as a practical tool: a field engineer with no computer can sort houses into risk groups with these few questions.",
        [{"type": "icicle", "ids": ids, "labels": labels, "parents": parents, "values": vals, "branchvalues": "total", "customdata": cd,
          "marker": {"colors": colors, "colorscale": RISK, "cmin": 0, "cmax": 1, "line": {"color": SURFACE, "width": 2},
                     "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": ".0%", "tickfont": {"color": MUTED}, "title": {"text": "destroyed", "font": {"color": MUTED, "size": 11}}}},
          "texttemplate": "%{label}<br>%{customdata[0]:.0%} destroyed", "hovertemplate": "%{label}<br>%{customdata[1]:.1%} of houses<br>%{customdata[0]:.1%} destroyed<extra></extra>",
          "tiling": {"orientation": "v"}, "textfont": {"family": FONT, "size": 11}, "insidetextfont": {"color": INK}, "maxdepth": 5}],
        {**lay(), "margin": {"l": 4, "r": 4, "t": 8, "b": 4}}, 520)

    rules = ck["rules"]
    add("checklist_rules", "One model", "Every leaf of the checklist as a plain rule",
        f"The safest group was destroyed {pct(rules[-1]['p'])} of the time, the riskiest {pct(rules[0]['p'])}.",
        "Each of the checklist's leaves written out as an if-then rule, with the share of those houses destroyed.",
        "Follow every path from the top of the checklist tree to a leaf and join the questions with AND. The rate is the leaf's smoothed collapse share on training houses.",
        "Longer bar = a riskier group. Hover a bar to read the full rule.",
        "These rules are what a reconstruction programme could print: inspect or strengthen the top groups first.",
        [hbars([r["p"] for r in rules][::-1], [f"Rule {chr(65 + i)} · {pct(r['share'])} of houses" for i, r in enumerate(rules)][::-1], color=BLUE, showlegend=False,
               hovertext=["<br>AND ".join(r["if"]) for r in rules][::-1], hovertemplate="%{hovertext}<br><b>%{x:.1%} destroyed</b><extra></extra>")],
        lay("Share destroyed", "", xa={"tickformat": ".0%", "range": [0, 1]}), 420)

    GM = rep["gam"]
    shapes = {t["source"]: t for t in GM["terms"] if t["type"] == "spline"}
    text = {
        "pred_intensity": ("shaking", "Shaking: the GAM's learned curve", "shaking intensity", "Shaking"),
        "count_floors_pre_eq": ("floors", "Storeys: the GAM's learned curve", "storeys", "Storeys"),
        "age_building": ("age", "Age: the GAM's learned curve", "age", "Age (years)"),
        "plinth_area_sq_ft": ("area", "Floor area: the GAM's learned curve", "floor area", "Floor area (sq ft)"),
        "height_ft_pre_eq": ("height", "Height: the GAM's learned curve", "height", "Height (ft)"),
    }
    for src, (gid, title, word, xl) in text.items():
        t = shapes.get(src)
        if not t:
            continue
        c = t["curve"]
        rng_ = max(c) - min(c)
        hi_x = t["grid"][c.index(max(c))]
        add(f"gam_{gid}", "One model", title,
            f"Across the usual range of {word}, the odds of collapse change by up to ×{math.exp(rng_):.1f}, with all other facts held fixed. Highest near {hi_x:.4g}.",
            f"How the GAM's prediction moves with {word} alone, holding every other fact about the house fixed.",
            f"A logistic generalized additive model writes the log-odds of collapse as a sum of smooth curves, one per number, plus one weight per flag. Each curve is a cubic B-spline with a penalty on wiggliness (P-spline), fitted by local scoring and backfitting. λ = {GM['lam']} was chosen on validation houses. This curve used {t['edf']:.1f} effective degrees of freedom.",
            "Up = more likely to be destroyed. A rise of 0.69 on the y-axis doubles the odds. The grey bars underneath show how many training houses sit at each value, so you can see where the curve is well supported.",
            "The GAM is the explainable model: for any house the website splits its risk into one contribution per fact using these curves.",
            [line(t["grid"], c, "GAM curve", VIOLET, hovertemplate=f"{xl} %{{x:.3g}}<br>log-odds %{{y:+.2f}}<extra></extra>", showlegend=False),
             bars(t["hist_x"], t["hist_n"], "Houses", AXIS, yaxis="y2", hovertemplate="%{y:,} houses<extra></extra>", showlegend=False)],
            {**lay(xl, "Effect on log-odds"), "yaxis": {**ax("Effect on log-odds"), "domain": [0.3, 1]}, "yaxis2": {**ax(""), "domain": [0, 0.2], "showticklabels": False, "anchor": "x"}, "bargap": 0.05}, 400)

    lin = sorted([t for t in GM["terms"] if t["type"] == "linear"], key=lambda t: -abs(t["effect"]))[:16]
    lin.sort(key=lambda t: t["effect"])
    up = [t for t in lin if t["effect"] > 0]
    dn = [t for t in lin if t["effect"] <= 0]
    top_up = max(lin, key=lambda t: t["effect"])
    top_dn = min(lin, key=lambda t: t["effect"])
    add("gam_flags", "One model", "Which yes/no facts push risk up or down",
        f"'{top_up['name']}' multiplies the odds of collapse by {math.exp(top_up['effect']):.1f}; '{top_dn['name']}' cuts them to ×{math.exp(top_dn['effect']):.2f}.",
        "The GAM's weight for each yes/no fact: how the log-odds change when the fact is true, with everything else held fixed.",
        "Flags and category levels enter the GAM as straight-line terms. Each weight is compared against the most common level of its category (for example, a flat site, a mud-mortar foundation, a light roof).",
        "Right of zero (orange) raises collapse risk, left of zero (blue) lowers it. 0.69 = double the odds; −0.69 = half.",
        "These weights, with the shape curves, build the per-house explanation in the house builder.",
        [hbars([t["effect"] for t in up], [t["name"] for t in up], "Raises risk", ORANGE, hovertemplate="%{y}<br>log-odds %{x:+.2f}<extra></extra>"),
         hbars([t["effect"] for t in dn], [t["name"] for t in dn], "Lowers risk", BLUE, hovertemplate="%{y}<br>log-odds %{x:+.2f}<extra></extra>")],
        lay("Effect on log-odds of collapse", "", ya={"categoryorder": "array", "categoryarray": [t["name"] for t in lin]}), 480)

    lams = GM["lams"]
    add("gam_lambda", "One model", "Choosing how smooth the GAM's curves are",
        f"λ = {GM['lam']} gave the lowest validation log loss. Bigger λ means straighter curves.",
        "Validation log loss of the GAM fitted with five different smoothing penalties.",
        "Fit the full GAM at λ = 1, 3, 10, 30 and 100 on training houses, then score each on the 20,000 validation houses. Hover shows the total effective degrees of freedom (how flexible the model ended up).",
        "Lower is better. The dip marks the best balance between bending to fit the data and staying smooth.",
        "The chosen λ is the GAM used everywhere else.",
        [line([r["lam"] for r in lams], [r["val_log_loss"] for r in lams], "GAM", VIOLET, mode="lines+markers", marker={"size": 9, "color": VIOLET, "line": {"color": SURFACE, "width": 1}},
              customdata=[r["edf"] for r in lams], hovertemplate="λ %{x}<br>log loss %{y:.4f}<br>%{customdata:.1f} degrees of freedom<extra></extra>", showlegend=False)],
        lay("Smoothing penalty λ (log scale)", "Validation log loss", xa={"type": "log"}))

    # ================================================================== ENSEMBLES
    gr = rep["growth"]
    f_auc = gr["forest"]["auc"][-1]
    st = gr["single_tree_auc"]
    add("trees_alone", "Ensembles", "Weak alone, strong together",
        f"Each forest tree on its own ranks houses with AUC {min(st):.3f}–{max(st):.3f}. Averaged, the forest reaches {f_auc:.3f}.",
        "The AUC of 40 individual trees from the random forest, against the AUC of the whole forest.",
        "Score each tree alone on 30,000 houses from the held-out districts, then score the forest's average. Each tree saw a different bootstrap sample of houses and a random third of the facts at each split.",
        "Each dot is one tree. The line is the forest. The gap between the cloud of dots and the line is what averaging buys.",
        "This is the core idea of ensemble learning: combining many unstable models cancels out their separate mistakes.",
        [dots(list(range(1, len(st) + 1)), st, "Single forest trees", AQUA, 9, hovertemplate="Tree %{x}<br>AUC %{y:.3f}<extra></extra>"),
         dots(list(range(1, len(gr["single_bag_auc"]) + 1)), gr["single_bag_auc"], "Single bagging trees", REF, 7, hovertemplate="Bagging tree %{x}<br>AUC %{y:.3f}<extra></extra>")],
        lay("Tree number", "AUC on held-out districts", shapes=[hline(f_auc, AQUA, "solid")], annotations=[note(0.99, f_auc, f"Whole forest {f_auc:.3f}", xref="paper", xanchor="right", yanchor="bottom")]))

    traces = []
    for key in ("forest", "bagging", "adaboost", "gboost"):
        g = gr[key]
        traces.append(mline(key, g["n"], g["auc"], L[key], mode="lines+markers", marker={"size": 6, "color": style(key)[0]}, hovertemplate=f"{L[key]}<br>%{{x}} members<br>AUC %{{y:.3f}}<extra></extra>"))
    add("growth", "Ensembles", "Adding members: bagging, forest and boosting",
        f"Gradient boosting kept improving up to {gr['gboost']['best_n']} trees; the forest flattened after a few dozen.",
        "How ranking quality on the held-out districts grows as each ensemble gets more members.",
        "Bagging and the random forest average more and more trees. AdaBoost and gradient boosting add trees one after another, each fixing the last one's errors. Every point is scored on the same 30,000 held-out houses.",
        "Up is better. Averaging methods (bagging, forest) climb fast and plateau. Boosting methods climb more slowly but keep going.",
        "Shows the two families of ensemble methods side by side: averaging lowers variance, boosting lowers bias.",
        traces, lay("Members in the ensemble (log scale)", "AUC on held-out districts", xa={"type": "log"}))

    add("oob", "Ensembles", "Out-of-bag error: a free test set",
        f"The forest's out-of-bag error settled at {pct(gr['forest']['oob_error'][-1], 1)}, with no separate test data needed.",
        "Error measured on the training houses that each tree did not see in its bootstrap sample.",
        "Each tree trains on a bootstrap sample (houses drawn with replacement), which leaves out about 37% of houses. Each house is scored only by the trees that never saw it, and the share wrongly classified is plotted as trees are added.",
        "Down is better. The curve flattening means more trees no longer help.",
        "Out-of-bag error is how a forest can pick its own size without touching the test districts.",
        [mline("forest", list(range(1, len(gr["forest"]["oob_error"]) + 1)), gr["forest"]["oob_error"], "Random forest", hovertemplate="%{x} trees<br>OOB error %{y:.2%}<extra></extra>"),
         mline("bagging", list(range(1, len(gr["bagging"]["oob_error"]) + 1)), gr["bagging"]["oob_error"], "Bagging", hovertemplate="%{x} trees<br>OOB error %{y:.2%}<extra></extra>")],
        lay("Trees", "Out-of-bag error", ya={"tickformat": ".0%"}))

    ab = gr["adaboost"]
    add("ada_errors", "Ensembles", "AdaBoost: each stump faces harder houses",
        f"The first stump erred on {pct(ab['errors'][0])} of the weight; later stumps hover near 50% because the easy houses have been down-weighted.",
        "The weighted error of each successive AdaBoost stump (a one-question tree).",
        "AdaBoost gives every house a weight. After each stump, houses it got wrong are up-weighted by e^α and houses it got right are down-weighted. The next stump is fitted to the new weights. α = ½ ln((1−err)/err) is the stump's vote.",
        "A rising curve is expected and healthy: each new stump is asked to solve what the earlier ones could not. Hover for each stump's vote α.",
        "Shows boosting's mechanism: adaptive re-weighting toward the hard cases.",
        [line(list(range(1, len(ab["errors"]) + 1)), ab["errors"], "Weighted error", REF, customdata=ab["alphas"], showlegend=False,
              hovertemplate="Stump %{x}<br>weighted error %{y:.1%}<br>vote α %{customdata:.3f}<extra></extra>")],
        lay("Boosting round", "Weighted error", ya={"tickformat": ".0%"}, shapes=[hline(0.5, AXIS, "dot")]))

    stp = rep["stumps"]
    add("ada_stumps", "Ensembles", "AdaBoost's first 12 basis functions",
        f"The first question AdaBoost asked was '{stp[0]['collapse_if']}'." if stp else "",
        "The first twelve stumps AdaBoost chose, each with its vote α.",
        "Each stump is one yes/no question learned from the weighted houses. The model's score is Σ α_m · h_m(x), where h_m answers +1 (collapse) or −1 (stands).",
        "Longer bar = a stronger vote. Labels show the condition under which each stump votes 'collapse'. The same fact can appear several times at slightly different cut-offs.",
        "Each stump is a learned basis function. Their weighted sum is an adaptive basis function model, and because stumps ask one question each, AdaBoost with stumps is an additive model like the GAM.",
        [hbars([s["alpha"] for s in stp][::-1], [f"{i + 1}. {s['collapse_if']}" for i, s in enumerate(stp)][::-1], color=REF, showlegend=False, hovertemplate="%{y}<br>vote α %{x:.3f}<extra></extra>")],
        lay("Vote α", ""), 440)

    gb = gr["gboost"]
    add("gb_curves", "Ensembles", "Gradient boosting: stop before it over-fits",
        f"Validation log loss bottomed out at tree {gb['best_n']}; the model keeps only those trees.",
        "Log loss on training and on validation houses as gradient boosting adds depth-4 regression trees.",
        "Start from the overall log-odds. Each round, compute every house's residual y − p, fit a small regression tree to it on a random 80% of houses, set each leaf by a Newton step, and add the tree times a learning rate of 0.05.",
        "Both lines fall at first. Training keeps falling forever; validation turns flat or up when the model begins to memorise. The dashed line is where it stopped.",
        "Early stopping on validation is how the boosting model in every comparison was sized.",
        [line(list(range(1, len(gb["train"]) + 1)), gb["train"], "Training houses", REF, hovertemplate="%{x} trees<br>train log loss %{y:.4f}<extra></extra>"),
         mline("gboost", list(range(1, len(gb["val"]) + 1)), gb["val"], "Validation houses", hovertemplate="%{x} trees<br>validation log loss %{y:.4f}<extra></extra>")],
        lay("Trees", "Log loss", shapes=[vline(gb["best_n"])]))

    bs = rep["basis"]
    cum = bs["cumulative"]
    add("basis_cumulative", "Ensembles", "One Gorkha house, built up tree by tree",
        f"Starting from the average log-odds ({bs['f0']:+.2f}), {len(cum)} small trees moved this house to {cum[-1]:+.2f}, a {pct(bs['p_final'])} chance of collapse.",
        "The gradient-boosting score of one real house from Gorkha (a held-out district) after each tree is added.",
        "f(x) = f₀ + Σ ν·φ_m(x). Each φ_m is a regression tree learned from the data; ν is the learning rate. The line is the running total.",
        "Up = riskier. Early trees move the score a lot; later trees make small corrections. The final value, through the logistic function, is the predicted probability.",
        "This is an adaptive basis function model made visible: the basis functions are not chosen in advance, the data shapes them.",
        [line(list(range(0, len(cum) + 1)), [bs["f0"]] + cum, "Score", ORANGE, hovertemplate="after %{x} trees<br>log-odds %{y:+.3f}<extra></extra>", showlegend=False)],
        lay("Trees added", "Log-odds of collapse", shapes=[hline(0, AXIS, "dot")]))

    fst = bs["first"]
    add("basis_first", "Ensembles", "The first 40 basis functions for that house",
        "Each bar is one tree's push for this house. Most early trees push it toward collapse.",
        "What each of the first 40 boosting trees adds to the same Gorkha house's log-odds.",
        "For tree m, the bar is ν·φ_m(x): the learning rate times the value of the leaf the house lands in.",
        "Orange bars push toward collapse, blue bars away. Their running sum is the previous chart.",
        "Shows that every basis function is a different, data-chosen piece of the final answer.",
        [bars(list(range(1, len(fst) + 1)), fst, color=[ORANGE if v > 0 else BLUE for v in fst], showlegend=False, hovertemplate="Tree %{x}<br>adds %{y:+.4f}<extra></extra>")],
        lay("Tree", "Added log-odds"))

    pm = rep["importance"]["permutation"]
    add("importance", "Ensembles", "What gradient boosting relies on most",
        f"Scrambling '{pm[0]['name']}' costs the most ranking skill (AUC −{pm[0]['drop']:.3f}); then '{pm[1]['name']}'.",
        "How much the boosting model's AUC on held-out districts drops when one fact is shuffled across houses.",
        "Permutation importance: shuffle one fact (all its columns together) among 30,000 held-out houses, re-score, and average the AUC drop over two shuffles. A fact the model ignores costs nothing.",
        "Longer bar = the model leans on that fact more. Near-zero bars could be dropped with little loss.",
        "Used to explain the model in plain words, and to check it relies on sensible facts rather than accidents of the data.",
        [hbars([r["drop"] for r in pm][::-1], [r["name"] for r in pm][::-1], color=ORANGE, showlegend=False, hovertemplate="%{y}<br>AUC drop %{x:.4f}<extra></extra>")],
        lay("Drop in AUC when shuffled", ""), 460)

    imp = sorted(rep["importance"]["forest"].items(), key=lambda kv: kv[1])
    add("importance_forest", "Ensembles", "What the random forest split on most",
        f"'{imp[-1][0]}' accounts for {pct(imp[-1][1])} of all Gini decrease in the forest.",
        "Each fact's share of the total impurity decrease across every split in every forest tree.",
        "Every time a tree splits on a fact, the Gini impurity drops by some amount, weighted by the houses at that node. Add these up per fact over all trees and divide by the total.",
        "Longer bar = used more, and more usefully, by the forest. Compare with the permutation chart: the two methods mostly agree.",
        "A second, cheaper importance measure that comes for free with tree ensembles.",
        [hbars([v for _, v in imp], [k for k, _ in imp], color=AQUA, showlegend=False, hovertemplate="%{y}<br>%{x:.1%} of impurity decrease<extra></extra>")],
        lay("Share of impurity decrease", "", xa={"tickformat": ".0%"}), 460)

    # ================================================================== COMBINING
    sw = rep["stack"]["weights"]
    keys = list(sw)
    add("stack_weights", "Combining", "Stacking: how much to trust each model",
        f"The stacker leaned most on {L[max(sw, key=sw.get)]} (weight {max(sw.values()):.2f}).",
        "The weights a logistic regression learned for combining the seven base models' log-odds.",
        "Each base model predicts the 20,000 validation houses, which none of them trained on. A logistic regression then learns the best weights for their log-odds. Negative weights mean the stacker uses a model to correct the others.",
        "Taller bar = the stacker listens more to that model. Weights can be negative because the models are highly correlated.",
        "Stacking is one of the two combined models in the comparison. It learns its weights; the crowd average does not.",
        [bars([L[k] for k in keys], [sw[k] for k in keys], color=[MC.get(k, REF) for k in keys], showlegend=False, hovertemplate="%{x}<br>weight %{y:+.3f}<extra></extra>")],
        lay("", "Stacking weight", shapes=[hline(0, AXIS, "solid")]))

    fb = rep["family_bma"]
    names = fb["names"]
    ns = sorted(int(k) for k in fb["by_houses"])
    win = names[max(range(len(names)), key=lambda i: fb["weights"][i])]
    traces = []
    for i, k in enumerate(names):
        if k in MC:
            traces.append(mline(k, ns, [fb["by_houses"][str(n)][i] for n in ns], L[k], hovertemplate=f"{L[k]}<br>%{{x:,}} houses<br>weight %{{y:.3f}}<extra></extra>"))
    others = [sum(fb["by_houses"][str(n)][i] for i, k in enumerate(names) if k not in MC) for n in ns]
    traces.append(line(ns, others, "Other models", REF, dash="dot", hovertemplate="Other models<br>%{x:,} houses<br>weight %{y:.3f}<extra></extra>"))
    add("family_bma", "Combining", "Bayesian model averaging picks one winner",
        f"With all {ns[-1]:,} validation houses, {L[win]} takes {pct(max(fb['weights']), 1)} of the posterior weight.",
        "The Bayesian-model-averaging weight of each model family, computed from more and more validation houses.",
        "Each model's weight is proportional to how probable it makes the observed outcomes: exp(log-likelihood). With n houses the log-likelihood grows like n, so the best model's lead is multiplied n times.",
        "With a handful of houses the weights are spread out. With thousands, one model takes everything. That is why BMA here becomes model selection, and why stacking can beat it.",
        "This is the known limitation of BMA (Minka, 2000): it assumes one of the models is the truth and finds it, rather than combining them.",
        traces, lay("Validation houses used (log scale)", "Posterior weight", xa={"type": "log"}, ya={"range": [0, 1.02]}))

    bm = rep["bma"]
    inc = sorted(bm["inclusion"].items(), key=lambda kv: kv[1])
    add("bma_inclusion", "Combining", "Which facts matter? 4,096 models vote",
        f"{sum(v > 0.95 for v in bm['inclusion'].values())} of 12 groups of facts are almost certainly needed; {bm['models_for_99']} models hold 99% of the posterior weight.",
        "Posterior inclusion probability of each group of facts, from Bayesian model averaging over every possible logistic model.",
        f"Fit all 2¹² = {bm['n_models']:,} logistic models that use any subset of the 12 groups, on {bm['rows']:,} training houses. Approximate each model's evidence by BIC, weight models by exp(−BIC/2), then add up the weight of every model that includes a group.",
        "A bar near 100% means the data insist on that fact; near 0% means the models without it explain the data just as well with fewer parameters.",
        "Answers the planner's question 'which facts should a survey collect?' with a probability instead of a single model's yes/no.",
        [hbars([v for _, v in inc], [k for k, _ in inc], color=BLUE, showlegend=False, hovertemplate="%{y}<br>inclusion %{x:.1%}<extra></extra>")],
        lay("Posterior inclusion probability", "", xa={"tickformat": ".0%", "range": [0, 1.02]}), 420)

    groups = bm["groups"]
    top = bm["top"][:10]
    lbl = []
    for i, m in enumerate(top):
        miss = [g for g in groups if g not in m["factors"]]
        lbl.append(f"#{i + 1} · " + ("all 12 groups" if not miss else "without " + ", ".join(miss)))
    add("bma_top", "Combining", "The ten most probable models",
        f"The single best model holds {pct(top[0]['weight'], 1)} of the weight.",
        "The ten logistic models with the highest posterior weight among all 4,096.",
        "Posterior weight ∝ exp(−BIC/2), normalised over all subsets. Labels list which groups of facts each model leaves out.",
        "Longer bar = more probable. If the top bars share the same missing facts, those facts add little once the rest are known.",
        "The averaged prediction mixes these models by their weights; its score appears in the comparison as 'BMA'.",
        [hbars([m["weight"] for m in top][::-1], lbl[::-1], color=BLUE, showlegend=False, hovertemplate="%{y}<br>weight %{x:.1%}<extra></extra>")],
        lay("Posterior weight", "", xa={"tickformat": ".0%"}), 420)

    # ================================================================== COMPARISON
    order = ["checklist", "logit", "cart", "gam", "bagging", "adaboost", "forest", "gboost", "bma", "crowd", "stack"]
    order = [k for k in order if k in S]
    srt = sorted(order, key=lambda k: S[k]["unseen"]["auc"])
    best_u = srt[-1]
    add("auc_compare", "Comparison", "Every model, on seen and unseen districts",
        f"Best on the held-out districts: {L[best_u]} (AUC {S[best_u]['unseen']['auc']:.3f}). Every model loses about 0.1 AUC when it moves to new districts.",
        "AUC of every model on random new houses from the training districts (blue) and on every house in the three held-out districts (orange).",
        f"AUC is the chance that a randomly chosen destroyed house gets a higher risk score than a randomly chosen standing one. Seen test: {D['n_test_seen']:,} houses never used for training. Unseen test: all {D['n_test_unseen']:,} houses of {unseen}.",
        "Right is better. The gap in each row is how much a model's skill depends on knowing the local area. A shorter gap means the model travels better.",
        "This is the comparative analysis the review asks for: single models, ensembles and combined models scored on exactly the same houses.",
        [{"type": "scatter", "mode": "lines", "x": sum([[S[k]["seen"]["auc"], S[k]["unseen"]["auc"], None] for k in srt], []), "y": sum([[L[k], L[k], None] for k in srt], []),
          "line": {"color": AXIS, "width": 3}, "hoverinfo": "skip", "showlegend": False},
         dots([S[k]["seen"]["auc"] for k in srt], [L[k] for k in srt], "Seen districts (random new houses)", BLUE, 11, hovertemplate="%{y}<br>seen AUC %{x:.3f}<extra></extra>"),
         dots([S[k]["unseen"]["auc"] for k in srt], [L[k] for k in srt], "Held-out districts", ORANGE, 11, hovertemplate="%{y}<br>held-out AUC %{x:.3f}<extra></extra>")],
        lay("AUC", "", ya={"categoryorder": "array", "categoryarray": [L[k] for k in srt]}), 460)

    bt = [r for r in rep["bootstrap"]]
    bu = [r for r in bt if r["set"] == "unseen"]
    bse = [r for r in bt if r["set"] == "seen"]
    sig = [r for r in bu if r["low"] > 0]
    add("bootstrap", "Comparison", "Which gains are real? 95% bootstrap intervals",
        f"{len(sig)} of {len(bu)} comparisons on the held-out districts have an interval entirely above zero.",
        "The difference in AUC between two models, with a 95% interval from resampling houses.",
        f"Paired bootstrap: draw {config.N_BOOT_ROWS:,} houses with replacement {config.N_BOOT} times, score both models on the same draw, and take the 2.5th and 97.5th percentiles of the AUC difference.",
        "Each line is one comparison. If the whole line is right of zero, model A is reliably better than model B; if it crosses zero, the difference could be chance.",
        "Turns 'the ensemble scored higher' into 'the ensemble is better, and here is how sure we are'.",
        [{"type": "scatter", "mode": "markers", "x": [r["mean"] for r in rs], "y": [r["label"] for r in rs], "name": nm,
          "marker": {"color": c, "size": 10, "line": {"color": SURFACE, "width": 1}},
          "error_x": {"type": "data", "symmetric": False, "array": [r["high"] - r["mean"] for r in rs], "arrayminus": [r["mean"] - r["low"] for r in rs], "color": c, "thickness": 2, "width": 4},
          "hovertemplate": "%{y}<br>ΔAUC %{x:+.4f}<extra></extra>"} for rs, nm, c in ((bse, "Seen districts", BLUE), (bu, "Held-out districts", ORANGE))],
        lay("Difference in AUC (A − B)", "", shapes=[vline(0, AXIS, "solid")]), 480)

    R = rep["roc"]
    traces = [mline(k, R[k]["fpr"], R[k]["tpr"], f"{L[k]} ({S[k]['unseen']['auc']:.3f})" if k in S else f"{L[k]} ({rep['baselines'][k]['unseen']:.3f})",
                    hovertemplate=f"{L[k]}<br>false alarms %{{x:.0%}}<br>collapses caught %{{y:.0%}}<extra></extra>") for k in ("crowd", "stack", "gboost", "forest", "gam", "cart", "shaking", "rule") if k in R]
    traces.append(line([0, 1], [0, 1], "Coin toss", AXIS, dash="dash", hoverinfo="skip"))
    add("roc", "Comparison", "ROC curves on the held-out districts",
        "The combined models sit on top, but a simple 'mud walls first, then shaking' rule is closer than you might expect.",
        "For every possible risk cut-off: the share of destroyed houses caught against the share of standing houses wrongly flagged.",
        "Sort the held-out houses by each model's risk. Walking down the list, plot the share of real collapses found (up) against the share of false alarms (right). Area under the curve = AUC (in the legend).",
        "A curve closer to the top-left corner is better at every budget. The diagonal is a random order.",
        "Used to compare models independently of any one cut-off.",
        traces, lay("False-alarm rate", "Share of collapses caught", xa={"tickformat": ".0%"}, ya={"tickformat": ".0%"}), 440)

    cal = rep["calibration"]["unseen"]
    traces = [line([0, 1], [0, 1], "Perfect", AXIS, dash="dash", hoverinfo="skip")]
    for k in ("cart", "forest", "gboost", "gam", "crowd", "stack"):
        if k in cal:
            traces.append(mline(k, [c["predicted"] for c in cal[k]], [c["observed"] for c in cal[k]], L[k], mode="lines+markers", marker={"size": 7, "color": MC[k]},
                                customdata=[c["n"] for c in cal[k]], hovertemplate=f"{L[k]}<br>said %{{x:.0%}}<br>happened %{{y:.0%}}<br>%{{customdata:,}} houses<extra></extra>"))
    add("calibration", "Comparison", "When a model says 60%, is it 60%?",
        f"In the held-out districts every model over-predicts a little: the true collapse rate there ({pct(D['rate_unseen'])}) is lower than in training ({pct(D['rate_seen'])}).",
        "Reliability diagram: predicted probability against the share that really collapsed, on the held-out districts.",
        "Group houses into ten bins by predicted probability and plot the average prediction against the observed collapse share in each bin (bins with at least 5 houses).",
        "Points on the dashed diagonal are honest probabilities. Below it = the model is too pessimistic; above it = too optimistic.",
        "Calibration matters because the site adds up probabilities to forecast how many houses a district will lose.",
        traces, lay("Predicted probability", "Observed share destroyed", xa={"tickformat": ".0%", "range": [0, 1]}, ya={"tickformat": ".0%", "range": [0, 1]}), 440)

    srt_ll = sorted(order, key=lambda k: -S[k]["unseen"]["log_loss"])
    add("logloss", "Comparison", "Log loss: rewarding honest probabilities",
        f"Lowest log loss on the held-out districts: {L[srt_ll[-1]]} ({S[srt_ll[-1]]['unseen']['log_loss']:.3f}).",
        "Average log loss of every model on the held-out districts.",
        "Log loss = −mean(y·log p + (1−y)·log(1−p)). It punishes confident mistakes very hard, so it rewards both good ranking and honest probabilities.",
        "Shorter bar is better. A model that always says the average rate scores about 0.64 here.",
        "A second view of the comparison that also cares about calibration, not just ranking.",
        [hbars([S[k]["unseen"]["log_loss"] for k in srt_ll], [L[k] for k in srt_ll], color=[MC.get(k, REF) for k in srt_ll], showlegend=False, hovertemplate="%{y}<br>log loss %{x:.4f}<extra></extra>")],
        lay("Log loss on held-out districts", ""), 440)

    ns_ = rep["no_shaking"]
    keys = [k for k in ("logit", "gboost") if k in ns_]
    with_ = {k: S[k]["unseen"]["auc"] for k in keys}
    add("no_shaking", "Comparison", "What the shaking map adds",
        f"Without the ward shaking estimate, gradient boosting falls from {with_['gboost']:.3f} to {ns_['gboost']['unseen']:.3f} AUC on the held-out districts.",
        "AUC of two models trained with and without the estimated shaking of each house's ward.",
        "Train the same model twice: once on house facts only, once with the ward's shaking intensity added. Score both on the held-out districts.",
        "The gap between each pair of bars is the value of knowing how hard the ground shook. House facts alone still carry most of the signal.",
        "Shows the tool works best when combined with a shaking scenario, which is exactly what the map's 'stronger quake' slider does.",
        [bars([L[k] for k in keys], [ns_[k]["unseen"] for k in keys], "House facts only", REF, hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>"),
         bars([L[k] for k in keys], [with_[k] for k in keys], "House facts + shaking", ORANGE, hovertemplate="%{x}<br>AUC %{y:.3f}<extra></extra>")],
        lay("", "AUC on held-out districts", ya={"range": [0.5, 0.8]}, barmode="group"))

    # ================================================================== REAL WORLD
    cp = rep["capacity"]
    sh = cp["shares"]
    a20 = cp["at20"]
    traces = []
    for k in ("crowd", "stack", "gboost", "cart", "checklist", "rule", "shaking", "random"):
        if k in cp["curves"]:
            traces.append(mline(k, sh, cp["curves"][k], L[k], mode="lines+markers", marker={"size": 6, "color": style(k)[0]},
                                hovertemplate=f"{L[k]}<br>reach %{{x:.0%}} of houses<br>cover %{{y:.0%}} of collapses<extra></extra>"))
    add("capacity", "Real world", "If you can only strengthen 1 house in 5",
        f"Ranked by the {HL.lower()}, the riskiest 20% of houses in the held-out districts held {pct(a20[rep['headline']]['recall'])} of all collapses ({a20[rep['headline']]['real']:,} homes). A random 20% holds {pct(a20['random']['recall'])}.",
        "The share of all destroyed houses covered when a programme can only reach a limited share of houses, ranked by each method.",
        f"On the {cp['unseen_houses']:,} houses of the held-out districts ({cp['unseen_collapsed']:,} destroyed): sort by each method's risk, take the top X%, count the real collapses inside.",
        "Higher is better. The vertical gap at 20% is how many more doomed houses a model-guided programme would have reached than a random or rule-of-thumb one.",
        "This is the headline use: deciding where scarce retrofit money, inspections or tarpaulins go first, before the next earthquake.",
        traces, lay("Share of houses reached", "Share of collapses covered", xa={"tickformat": ".0%"}, ya={"tickformat": ".0%"}, shapes=[vline(0.2, AXIS, "dot")]), 440)

    types = list(fg["types"])
    add("fragility_observed", "Real world", "Fragility curves, straight from the survey",
        f"At the strongest shaking, {types[0].lower()} houses were destroyed {pct(max(v for v in fg['types'][types[0]]['observed'] if v is not None))} of the time.",
        "Observed collapse rate against ward shaking intensity, separately for six construction types.",
        "Group houses by main construction type and by shaking band (0.25 wide). Plot the destroyed share where a cell has at least 150 houses.",
        "Each line is a construction type. Steeper and higher = more fragile. The spread between lines at the same shaking is what better building buys.",
        "Fragility curves are what earthquake engineers use to plan; here they come straight out of 760,000 real outcomes.",
        [line(fg["bands"], fg["types"][t]["observed"], t, PAL[i], mode="lines+markers", marker={"size": 7, "color": PAL[i], "line": {"color": SURFACE, "width": 1}},
              customdata=fg["types"][t]["observed_n"], hovertemplate=f"{t}<br>shaking %{{x}}<br>%{{y:.1%}} destroyed<br>%{{customdata:,}} houses<extra></extra>", connectgaps=False)
         for i, t in enumerate(types)],
        lay("Estimated shaking intensity (ward)", "Share destroyed", ya={"tickformat": ".0%", "range": [0, 1]}), 440)

    add("fragility_model", "Real world", "Fragility curves learned by the combined model",
        "The model reproduces the survey's curves and extends them smoothly to shaking levels where few houses of a type were seen.",
        f"The {HL.lower()}'s average predicted collapse probability for each construction type as shaking is dialled from 6.25 to 8.5.",
        "Partial dependence: take up to 1,500 real houses of each type, set every one's shaking to the same value, predict, and average. Repeat across the shaking scale.",
        "Compare with the previous chart. Lines that stay apart show the model has learned that construction, not just location, decides who falls.",
        "These curves drive the 'stronger quake' slider on the map and the house builder's shaking control.",
        [line(fg["grid"], fg["types"][t]["model"], t, PAL[i], hovertemplate=f"{t}<br>shaking %{{x}}<br>predicted %{{y:.1%}}<extra></extra>") for i, t in enumerate(types)],
        lay("Shaking intensity (set for every house)", "Predicted share destroyed", ya={"tickformat": ".0%", "range": [0, 1]}), 440)

    dd = rep["districts"]
    mx = max(max(d["collapsed"], d["predicted"]) for d in dd) * 1.08
    worst_d = max((d for d in dd if d["unseen"]), key=lambda d: abs(d["predicted"] - d["collapsed"]))
    add("district_totals", "Real world", "Forecasting each district's losses",
        f"For held-out {worst_d['name']} the model forecast {worst_d['predicted']:,.0f} destroyed houses; {worst_d['collapsed']:,} were.",
        "Predicted number of destroyed houses (sum of probabilities) against the real number, per district.",
        f"Add up the {HL.lower()}'s collapse probabilities over every house in a district. For training districts, about 1 house in 9 was used for training, so their points are slightly optimistic; the orange held-out districts are fully honest.",
        "Points on the dashed line are perfect forecasts. Above the line = the model expected more damage than happened.",
        "A district-level loss forecast is what relief agencies need in the first hours, before any survey exists.",
        [line([0, mx], [0, mx], "Perfect forecast", AXIS, dash="dash", hoverinfo="skip"),
         dots([d["collapsed"] for d in dd if not d["unseen"]], [d["predicted"] for d in dd if not d["unseen"]], "Training districts", BLUE, 12, text=[d["name"] for d in dd if not d["unseen"]],
              hovertemplate="%{text}<br>real %{x:,}<br>forecast %{y:,.0f}<extra></extra>"),
         dots([d["collapsed"] for d in dd if d["unseen"]], [d["predicted"] for d in dd if d["unseen"]], "Held-out districts", ORANGE, 14, text=[d["name"] for d in dd if d["unseen"]],
              hovertemplate="%{text}<br>real %{x:,}<br>forecast %{y:,.0f}<extra></extra>")],
        lay("Real destroyed houses", "Forecast destroyed houses", xa={"range": [0, mx]}, ya={"range": [0, mx]}), 440)

    W = rep["wards"]
    sz = [max(5, min(22, math.sqrt(w["n"]) / 3)) for w in W]
    map_lay = lambda: lay("Longitude", "Latitude", xa={"showgrid": False}, ya={"showgrid": False, "scaleanchor": "x", "scaleratio": 1.13})
    hov = "%{text}<br>%{customdata[0]:,} houses · shaking %{customdata[1]:.2f}<br>real %{customdata[2]:.0%} · model %{customdata[3]:.0%}<extra></extra>"
    cdw = [[w["n"], w["shaking"], w["actual"], w["model"]] for w in W]
    txt = [f"{w['place']} ({w['district']}{', held out' if w['unseen'] else ''})" for w in W]
    add("map_actual", "Real world", "Where houses really fell",
        "The damage band runs east–west, worst in the north-central hill districts, not simply around the epicentre.",
        "Every surveyed ward placed by its latitude and longitude, coloured by the share of its houses destroyed.",
        "One dot per ward at its centre point. Dot size grows with the number of surveyed houses. Colour = destroyed / surveyed.",
        "Brighter = worse. Hover any ward for its name, size, shaking and both the real and predicted rates.",
        "The real-outcome half of the map on the main page.",
        [{"type": "scatter", "mode": "markers", "x": [w["lon"] for w in W], "y": [w["lat"] for w in W], "text": txt, "customdata": cdw, "hovertemplate": hov, "showlegend": False,
          "marker": {"color": [w["actual"] for w in W], "colorscale": RISK, "cmin": 0, "cmax": 1, "size": sz, "line": {"color": SURFACE, "width": 0.5},
                     "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": ".0%", "tickfont": {"color": MUTED}}}}],
        map_lay(), 480)
    add("map_model", "Real world", "Where the model expected houses to fall",
        "The model rebuilds the same pattern from house facts and shaking alone, including in the three districts it never saw.",
        f"The same wards coloured by the {HL.lower()}'s average predicted collapse probability.",
        "Average of the model's probabilities over every surveyed house in the ward.",
        "Compare with the previous map. Where the colours agree, the model has learned what really happened.",
        "The prediction half of the map on the main page, and the starting point of the 'stronger quake' slider.",
        [{"type": "scatter", "mode": "markers", "x": [w["lon"] for w in W], "y": [w["lat"] for w in W], "text": txt, "customdata": cdw, "hovertemplate": hov, "showlegend": False,
          "marker": {"color": [w["model"] for w in W], "colorscale": RISK, "cmin": 0, "cmax": 1, "size": sz, "line": {"color": SURFACE, "width": 0.5},
                     "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": ".0%", "tickfont": {"color": MUTED}}}}],
        map_lay(), 480)
    err = [w["actual"] - w["model"] for w in W]
    add("map_error", "Real world", "Where reality beat or missed the forecast",
        "Orange wards lost more houses than their houses and shaking suggest: likely landslides, soft soil or aftershocks the ward estimate misses.",
        "The difference between the real and predicted destroyed share in every ward.",
        "Real share − predicted share per ward. Blue = fewer collapses than expected; orange = more.",
        "Grey dots are wards the model got about right. Clusters of one colour point to something local the model does not know about.",
        "Errors are where the next survey or the next model input (soil, slope maps, aftershock shaking) should focus.",
        [{"type": "scatter", "mode": "markers", "x": [w["lon"] for w in W], "y": [w["lat"] for w in W], "text": txt, "customdata": cdw, "hovertemplate": hov, "showlegend": False,
          "marker": {"color": err, "colorscale": DIV, "cmin": -0.5, "cmax": 0.5, "size": sz, "line": {"color": SURFACE, "width": 0.5},
                     "colorbar": {"thickness": 10, "outlinewidth": 0, "tickformat": "+.0%", "tickfont": {"color": MUTED}}}}],
        map_lay(), 480)

    shf, stt = rep["shifts"], rep["shift_totals"]
    i0 = shf.index(0.0)
    add("shift_totals", "Real world", "What if the earthquake had been stronger or weaker?",
        f"Half a point less shaking everywhere: about {stt[shf.index(-0.5)]:,.0f} destroyed homes instead of {stt[i0]:,.0f}. Half a point more: {stt[shf.index(0.5)]:,.0f}.",
        "The model's forecast of destroyed houses across all surveyed wards if every ward's shaking were shifted up or down.",
        f"Re-score up to {config.WARD_SAMPLE} sampled houses per ward with the shaking changed by the same amount everywhere, average per ward, and scale by each ward's house count.",
        "The centre point is the real earthquake. The steepness shows how quickly losses grow with intensity: a few tenths of a point make a huge difference.",
        "Powers the scenario slider on the map. It turns the model into a planning tool for the next earthquake, not a story about the last one.",
        [line(shf, stt, "Forecast destroyed homes", ORANGE, mode="lines+markers", marker={"size": 8, "color": ORANGE, "line": {"color": SURFACE, "width": 1}},
              hovertemplate="shaking shift %{x:+.2f}<br>%{y:,.0f} homes destroyed<extra></extra>", showlegend=False)],
        lay("Shift in shaking intensity", "Forecast destroyed homes", xa={"tickformat": "+.2f"}, shapes=[vline(0, AXIS, "dot")]))

    up_ = rep["upgrades"]
    best_up = max(up_, key=lambda u: u["drop"]) if up_ else None
    add("upgrades", "Real world", "What each upgrade would change",
        f"'{best_up['label']}' cuts the average predicted collapse risk from {pct(best_up['before'])} to {pct(best_up['after'])} for the houses it applies to." if best_up else "",
        "Average predicted collapse probability for houses in the held-out districts, before and after one change to the house.",
        "Take real houses the change applies to (up to 6,000), change that one fact in their records, and re-predict with the combined model. These are associations learned from the survey, not tested engineering results.",
        "Each row joins 'as built' (grey) to 'after the change' (orange). A longer line = a bigger expected benefit.",
        "Helps compare cheap fixes (lighter roof) with expensive ones (rebuilding) when deciding retrofit advice.",
        [{"type": "scatter", "mode": "lines", "x": sum([[u["before"], u["after"], None] for u in up_], []), "y": sum([[u["label"], u["label"], None] for u in up_], []),
          "line": {"color": AXIS, "width": 3}, "hoverinfo": "skip", "showlegend": False},
         dots([u["before"] for u in up_], [u["label"] for u in up_], "As built", REF, 12, customdata=[u["n"] for u in up_], hovertemplate="%{y}<br>as built %{x:.1%}<br>%{customdata:,} houses<extra></extra>"),
         dots([u["after"] for u in up_], [u["label"] for u in up_], "After the change", ORANGE, 12, hovertemplate="%{y}<br>after %{x:.1%}<extra></extra>")],
        lay("Average predicted collapse probability", "", xa={"tickformat": ".0%", "range": [0, max([u["before"] for u in up_] + [0.1]) * 1.15]}), 360)

    # ================================================================== DAMAGE GRADES (CART for a five-way categorisation)
    gd = rep["grades"]
    gu, gs = gd["unseen"], gd["seen"]
    cats = ["CART", "Random forest", "Majority guess"]
    add("grade_scores", "Damage grades", "Sorting houses into all five damage grades",
        f"On the held-out districts CART puts {pct(gu['cart']['exact'])} of houses in exactly the right grade and {pct(gu['cart']['within_one'])} within one grade.",
        "Accuracy of the five-grade CART tree and five-grade random forest, against always guessing the most common grade.",
        f"CART here grows a multi-class tree on Gini impurity over the five grades, pruned by validation log loss ({gd['cart_leaves']} leaves). The forest averages {gd['forest_trees']} such trees. 'Within one' counts a grade-4 house called grade 3 or 5 as close.",
        "Taller is better. Exact grades are hard because neighbouring grades blur into each other on the ground; within-one accuracy is the practical measure.",
        "CART used for categorisation rather than a yes/no answer, on a real five-class problem.",
        [bars(cats, [gu["cart"]["exact"], gu["forest"]["exact"], gu["cart"]["majority_exact"]], "Exact grade", BLUE, hovertemplate="%{x}<br>exact %{y:.1%}<extra></extra>"),
         bars(cats, [gu["cart"]["within_one"], gu["forest"]["within_one"], gu["cart"]["majority_within_one"]], "Within one grade", AQUA, hovertemplate="%{x}<br>within one %{y:.1%}<extra></extra>")],
        lay("", "Accuracy on held-out districts", ya={"tickformat": ".0%", "range": [0, 1]}, barmode="group"))

    for key, nm in (("cart", "CART"), ("forest", "random forest")):
        M = gu[key]["confusion_rows"]
        add(f"grade_confusion_{key}", "Damage grades", f"Where the five-grade {nm} goes wrong",
            f"Of houses truly destroyed, the {nm} calls {pct(M[4][4])} grade 5. Most mistakes land on a neighbouring grade.",
            f"Confusion matrix of the five-grade {nm} on the held-out districts, as row percentages.",
            "Rows are the real grade, columns the predicted grade. Each row is divided by its total, so a row sums to 100%.",
            "A bright diagonal is good. Brightness one step off the diagonal is a near miss; far corners would be serious errors.",
            "Shows which grades the models confuse, and why within-one accuracy is the fairer score.",
            [heat(M, short, short, "real %{y}<br>predicted %{x}<br>%{z:.0%} of the row", zmin=0, zmax=1, text=[[pct(v) for v in r] for r in M])],
            lay("Predicted grade", "Real grade", ya={"autorange": "reversed"}), 420)
    return out


def write_graphs(rep=None):
    if rep is None:
        rep = json.loads((config.ARTIFACTS / "report.json").read_text(encoding="utf-8"))
    graphs = build_graphs(rep)
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(graphs, separators=(",", ":")), encoding="utf-8")
    print(f"{len(graphs)} graphs written")
    return graphs


if __name__ == "__main__":
    write_graphs()
