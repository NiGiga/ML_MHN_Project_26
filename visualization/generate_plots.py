import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

BASE = "/home/gugutca/PycharmProjects/ML_Course_Project"
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

# Colori fissi per tutti i grafici
color_map = {
    "classic": "#636EFA",  # blu
    "mhn": "#EF553B"       # rosso
}

# Layout comune
common_layout = dict(
    template="plotly_white",
    title_font=dict(size=22),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.28,
        xanchor="center",
        x=0.5,
        title_text=""
    ),
    margin=dict(t=240, l=70, r=30, b=70),
)

def load_json(name):
    with open(os.path.join(RES, name), "r") as f:
        return json.load(f)

# Caricamento risultati
exp1 = load_json("exp1_correction.json")
exp2 = load_json("exp2_capacity.json")
exp4 = load_json("exp4_complexity.json")
exp3 = load_json("exp3_comparison.json")

# ============================================================================
# CSV 1 — Esperimento 1
# ============================================================================
rows = []
for level, models in exp1.items():
    for model, rates in models.items():
        for rate, val in rates.items():
            rows.append({
                "level": level,
                "model": model,
                "corruption": float(rate),
                "recovery": val
            })
df1 = pd.DataFrame(rows)
df1.to_csv(os.path.join(RES, "../results/exp1_correction.csv"), index=False)

# ============================================================================
# CSV 2 — Esperimento 2
# ============================================================================
rows = []
for model in ["classic", "mhn"]:
    for k, val in exp2[model].items():
        rows.append({
            "model": model,
            "n_patterns": int(k),
            "recovery": val
        })
df2 = pd.DataFrame(rows)
df2.to_csv(os.path.join(RES, "../results/exp2_capacity.csv"), index=False)

# ============================================================================
# CSV 3 — Esperimento 4
# ============================================================================
rows = []
for level, models in exp4.items():
    for model in ["classic", "mhn"]:
        for rate, stats in models[model].items():
            rows.append({
                "level": level,
                "model": model,
                "corruption": float(rate),
                "mean": stats["mean"],
                "std": stats["std"]
            })
df4 = pd.DataFrame(rows)
df4.to_csv(os.path.join(RES, "../results/exp4_complexity.csv"), index=False)

# ============================================================================
# FIGURA 1 — Recovery vs corruption
# ============================================================================
fig1 = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=["Simple", "Medium", "Complex"]
)

for i, lvl in enumerate(["simple", "medium", "complex"], start=1):
    sub = df1[df1["level"] == lvl]

    for model in ["classic", "mhn"]:
        d = sub[sub["model"] == model]

        fig1.add_trace(
            go.Scatter(
                x=d["corruption"] * 100,
                y=d["recovery"] * 100,
                mode="lines+markers",
                name=model,
                legendgroup=model,
                showlegend=(i == 1),
                line=dict(color=color_map[model], width=2),
                marker=dict(color=color_map[model], size=7),
            ),
            row=i,
            col=1
        )

fig1.update_layout(
    title="Recovery vs corruption <br><span style='font-size:16px;font-weight:normal;'>Source: exp1 | MHN scales better with many memories</span>",
    **common_layout
)

fig1.update_xaxes(title_text="Corrup.")
fig1.update_yaxes(title_text="Recov.")

fig1.write_image(os.path.join(RES, "../results/exp1_correction.png"))

with open(os.path.join(RES, "../results/exp1_correction.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Recovery vs corruption ",
        "description": "Classical Hopfield and MHN recovery rates across corruption levels for three LTLf complexity levels."
    }, f)

# ============================================================================
# FIGURA 2 — Recovery vs patterns
# ============================================================================
fig2 = px.line(
    df2,
    x="n_patterns",
    y="recovery",
    color="model",
    markers=True,
    color_discrete_map=color_map
)

fig2.update_traces(line=dict(width=2), marker=dict(size=7))

fig2.update_layout(
    title="Recovery vs patterns <br><span style='font-size:16px;font-weight:normal;'>Source: exp2 | classical Hopfield drops near its capacity</span>",
    **common_layout
)

fig2.update_xaxes(title_text="N pat")
fig2.update_yaxes(title_text="Recov.")

fig2.write_image(os.path.join(RES, "../results/exp2_capacity.png"))

with open(os.path.join(RES, "../results/exp2_capacity.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Recovery vs patterns ",
        "description": "Classical capacity threshold and gradual MHN decay as stored pattern count increases."
    }, f)

# ============================================================================
# FIGURA 3 — Recovery by complexity
# ============================================================================
sub = df4[df4["corruption"] == 0.15]

fig3 = px.line(
    sub,
    x="level",
    y="mean",
    color="model",
    markers=True,
    error_y="std",
    color_discrete_map=color_map
)

fig3.update_traces(line=dict(width=2), marker=dict(size=7))

fig3.update_layout(
    title="Recovery by complexity <br><span style='font-size:16px;font-weight:normal;'>Source: exp4 | both models improve on complex traces</span>",
    **common_layout
)

fig3.update_xaxes(title_text="LTLf lvl")
fig3.update_yaxes(title_text="Recov.")

fig3.write_image(os.path.join(RES, "../results/exp4_complexity.png"))

with open(os.path.join(RES, "../results/exp4_complexity.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Recovery by complexity",
        "description": "Average recovery rates at 15% corruption across simple, medium, and complex LTLf levels."
    }, f)

# ============================================================================
# FIGURA 4 — Exp3 comparison
# ============================================================================
df3 = pd.read_csv(os.path.join(RES, "exp3_comparison.csv"))

# preparo etichette modello
def model_label(row):
    if row["model"] == "classic":
        return "classic"
    return f"mhn β={row['beta']}"

df3["model_label"] = df3.apply(model_label, axis=1)

# Grafico accuracy
fig4 = px.bar(
    df3,
    x="level",
    y="accuracy",
    color="model_label",
    barmode="group",
    category_orders={"level": ["simple", "medium", "complex"]},
    title="Accuracy by model <br><span style='font-size:16px;font-weight:normal;'>Source: exp3 | direct comparison at 15% corruption</span>"
)

fig4.update_layout(**common_layout)
fig4.update_xaxes(title_text="Level")
fig4.update_yaxes(title_text="Accur.")
fig4.write_image(os.path.join(RES, "exp3_accuracy.png"))

with open(os.path.join(RES, "exp3_accuracy.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Accuracy by model",
        "description": "Grouped bar chart comparing classical Hopfield and MHN variants across LTLf complexity levels."
    }, f)

# Grafico steps
fig5 = px.bar(
    df3,
    x="level",
    y="steps_mean",
    color="model_label",
    barmode="group",
    category_orders={"level": ["simple", "medium", "complex"]},
    title="Steps to convergence <br><span style='font-size:16px;font-weight:normal;'>Source: exp3 | MHN typically converges in fewer steps</span>"
)

fig5.update_layout(**common_layout)
fig5.update_xaxes(title_text="Level")
fig5.update_yaxes(title_text="Steps")
fig5.write_image(os.path.join(RES, "exp3_steps.png"))

with open(os.path.join(RES, "exp3_steps.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Steps to convergence",
        "description": "Grouped bar chart comparing mean convergence steps for classical Hopfield and MHN variants."
    }, f)

# ============================================================================
# FIGURA 6 — Energy profile (Exp3)
# ============================================================================
fig6 = go.Figure()

energy_colors = {
    "classic": "#636EFA",
    "mhn_1.0": "#EF553B",
    "mhn_2.0": "#00CC96",
    "mhn_5.0": "#AB63FA",
    "mhn_10.0": "#FFA15A",
}

level = "complex"  # puoi cambiare in "simple" o "medium"

# Classic
classic = exp3[level]["classic"]
x_classic = list(range(len(classic["energy_profile_mean"])))
y_classic = np.array(classic["energy_profile_mean"])
s_classic = np.array(classic["energy_profile_std"])

fig6.add_trace(go.Scatter(
    x=x_classic,
    y=y_classic + s_classic,
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))
fig6.add_trace(go.Scatter(
    x=x_classic,
    y=y_classic - s_classic,
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(99,110,250,0.18)",
    showlegend=False,
    hoverinfo="skip"
))
fig6.add_trace(go.Scatter(
    x=x_classic,
    y=y_classic,
    mode="lines+markers",
    name="classic",
    line=dict(color=energy_colors["classic"], width=2),
    marker=dict(color=energy_colors["classic"], size=6)
))

# MHN beta sweep
for beta_key, label, rgba_fill in [
    ("beta_1.0", "mhn β=1.0", "rgba(239,85,59,0.16)"),
    ("beta_2.0", "mhn β=2.0", "rgba(0,204,150,0.16)"),
    ("beta_5.0", "mhn β=5.0", "rgba(171,99,250,0.16)"),
    ("beta_10.0", "mhn β=10.0", "rgba(255,161,90,0.16)")
]:
    r = exp3[level]["mhn"][beta_key]
    x = list(range(len(r["energy_profile_mean"])))
    y = np.array(r["energy_profile_mean"])
    s = np.array(r["energy_profile_std"])

    color_key = f"mhn_{r['beta']}"

    fig6.add_trace(go.Scatter(
        x=x,
        y=y + s,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig6.add_trace(go.Scatter(
        x=x,
        y=y - s,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=rgba_fill,
        showlegend=False,
        hoverinfo="skip"
    ))
    fig6.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=label,
        line=dict(color=energy_colors[color_key], width=2),
        marker=dict(color=energy_colors[color_key], size=6)
    ))

fig6.update_layout(
    title="Energy profile vs step <br><span style='font-size:16px;font-weight:normal;'>Source: exp3 | mean energy trajectory on complex traces</span>",
    **common_layout
)
fig6.update_xaxes(title_text="Step")
fig6.update_yaxes(title_text="Energy")
fig6.write_image(os.path.join(RES, "exp3_energy_profile.png"))

with open(os.path.join(RES, "exp3_energy_profile.png.meta.json"), "w") as f:
    json.dump({
        "caption": "Energy profile vs step",
        "description": "Mean energy trajectory with standard deviation bands for classical Hopfield and MHN across beta values on complex traces."
    }, f)

print("Saved PNG and CSV files in results/")