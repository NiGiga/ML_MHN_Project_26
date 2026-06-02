import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classical.hopfield_bindings import ClassicalHopfield
from mhn.modern_hopfield import ModernHopfieldNetwork

BASE = "/home/gugutca/PycharmProjects/ML_Course_Project"
DATA_DIR = os.path.join(BASE, "dataset", "data")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
LEVEL = "complex"
N_STORE = 12
N_SAMPLES = 12
CORRUPTION = 0.20
MHN_BETA = 5.0
CLASSIC_STEPS = 30
MHN_STEPS = 20


def select_orthogonal(traces, k):
    all_t = traces.astype(np.float64)
    corr = np.abs(all_t @ all_t.T) / all_t.shape[1]
    np.fill_diagonal(corr, 1.0)
    selected = [0]
    for _ in range(k - 1):
        avg_corr = corr[selected].mean(axis=0)
        avg_corr[selected] = 1.0
        selected.append(int(np.argmin(avg_corr)))
    return all_t[selected]


def corrupt_pattern(x, rate, rng):
    y = x.copy()
    mask = rng.random(len(y)) < rate
    y[mask] *= -1.0
    return y


def build_shared_tsne(memories, corrupted_arr, recovered_classic_arr, recovered_mhn_arr):
    X = np.vstack([
        memories,
        corrupted_arr,
        recovered_classic_arr,
        recovered_mhn_arr
    ])

    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=min(10, len(X) - 1),
        learning_rate="auto",
        random_state=SEED
    )
    coords = tsne.fit_transform(X)

    n_mem = len(memories)
    n_cor = len(corrupted_arr)
    n_cla = len(recovered_classic_arr)

    mem_xy = coords[:n_mem]
    cor_xy = coords[n_mem:n_mem + n_cor]
    cla_xy = coords[n_mem + n_cor:n_mem + n_cor + n_cla]
    mhn_xy = coords[n_mem + n_cor + n_cla:]

    return mem_xy, cor_xy, cla_xy, mhn_xy


def main():
    rng = np.random.default_rng(SEED)

    traces = np.load(os.path.join(DATA_DIR, f"valid_{LEVEL}.npy"))
    memories = select_orthogonal(traces, N_STORE)

    classic = ClassicalHopfield()
    classic.store(memories)

    mhn = ModernHopfieldNetwork(beta=MHN_BETA)
    mhn.store(memories)

    corrupted_list = []
    recovered_classic = []
    recovered_mhn = []
    source_idx = []

    for _ in range(N_SAMPLES):
        idx = rng.integers(len(memories))
        x = memories[idx].copy()
        xc = corrupt_pattern(x, CORRUPTION, rng)

        xr_classic, _, _ = classic.retrieve(xc.copy(), max_steps=CLASSIC_STEPS)
        xr_mhn = mhn.retrieve_binary(xc.copy(), n_steps=MHN_STEPS)

        corrupted_list.append(xc)
        recovered_classic.append(xr_classic)
        recovered_mhn.append(xr_mhn)
        source_idx.append(int(idx))

    corrupted_arr = np.array(corrupted_list)
    recovered_classic_arr = np.array(recovered_classic)
    recovered_mhn_arr = np.array(recovered_mhn)

    mem_xy, cor_xy, cla_xy, mhn_xy = build_shared_tsne(
        memories,
        corrupted_arr,
        recovered_classic_arr,
        recovered_mhn_arr
    )

    color_map = {
        "memory": "#1f77b4",
        "corrupted": "#7f7f7f",
        "classic": "#2ca02c",
        "mhn": "#d62728"
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        subplot_titles=("Classical Hopfield", "Modern Hopfield Network")
    )

    # LEFT
    fig.add_trace(go.Scatter(
        x=mem_xy[:, 0], y=mem_xy[:, 1],
        mode="markers+text",
        name="memories",
        legendgroup="memories",
        showlegend=True,
        text=[f"M{i}" for i in range(len(mem_xy))],
        textposition="top center",
        marker=dict(size=11, color=color_map["memory"], symbol="circle")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cor_xy[:, 0], y=cor_xy[:, 1],
        mode="markers",
        name="corrupted",
        legendgroup="corrupted",
        showlegend=True,
        marker=dict(size=10, color=color_map["corrupted"], symbol="x")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=cla_xy[:, 0], y=cla_xy[:, 1],
        mode="markers",
        name="classic rec.",
        legendgroup="classic",
        showlegend=True,
        marker=dict(size=10, color=color_map["classic"], symbol="diamond")
    ), row=1, col=1)

    # RIGHT
    fig.add_trace(go.Scatter(
        x=mem_xy[:, 0], y=mem_xy[:, 1],
        mode="markers+text",
        name="memories",
        legendgroup="memories",
        showlegend=False,
        text=[f"M{i}" for i in range(len(mem_xy))],
        textposition="top center",
        marker=dict(size=11, color=color_map["memory"], symbol="circle")
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=cor_xy[:, 0], y=cor_xy[:, 1],
        mode="markers",
        name="corrupted",
        legendgroup="corrupted",
        showlegend=False,
        marker=dict(size=10, color=color_map["corrupted"], symbol="x")
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=mhn_xy[:, 0], y=mhn_xy[:, 1],
        mode="markers",
        name="mhn rec.",
        legendgroup="mhn",
        showlegend=True,
        marker=dict(size=10, color=color_map["mhn"], symbol="square")
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        title=(
            f"t-SNE attractor basins ({LEVEL})"
            f"<br><span style='font-size:16px;font-weight:normal;'>"
            f"Source: dataset + recovery runs | local neighborhood view of memories and recovered states"
            f"</span>"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.18,
            xanchor="center",
            x=0.5,
            title_text=""
        ),
        margin=dict(t=210, l=70, r=30, b=70)
    )

    fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
    fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
    fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)

    out_png = os.path.join(RESULTS_DIR, "tsne_comparison_subplot.png")
    out_csv = os.path.join(RESULTS_DIR, "tsne_comparison_points.csv")
    out_meta = os.path.join(RESULTS_DIR, "tsne_comparison_subplot.png.meta.json")

    rows = []
    for i in range(len(mem_xy)):
        rows.append({
            "kind": "memory",
            "id": f"M{i}",
            "x": mem_xy[i, 0],
            "y": mem_xy[i, 1]
        })
    for i in range(len(cor_xy)):
        rows.append({
            "kind": "corrupted",
            "id": f"C{i}",
            "x": cor_xy[i, 0],
            "y": cor_xy[i, 1],
            "source_memory": source_idx[i]
        })
        rows.append({
            "kind": "classic_recovered",
            "id": f"CL{i}",
            "x": cla_xy[i, 0],
            "y": cla_xy[i, 1],
            "source_memory": source_idx[i]
        })
        rows.append({
            "kind": "mhn_recovered",
            "id": f"MHN{i}",
            "x": mhn_xy[i, 0],
            "y": mhn_xy[i, 1],
            "source_memory": source_idx[i]
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    fig.write_image(out_png)

    with open(out_meta, "w") as f:
        json.dump({
            "caption": "t-SNE attractor basins ",
            "description": "Side-by-side t-SNE visualization of memories, corrupted inputs, and recovered states for classical Hopfield and MHN."
        }, f)

    print("Saved:")
    print(out_png)
    print(out_csv)


if __name__ == "__main__":
    main()