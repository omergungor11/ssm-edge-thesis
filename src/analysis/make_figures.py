"""Tez şekilleri — ham veriden PDF + PNG (300 dpi) üretir.

Kaynaklar (salt okunur):
  results/raw/latency_matrix_*.jsonl   → gecikme kayıtları
  results/raw/energy_matrix.json       → enerji hücreleri
  results/raw/export_matrix.jsonl      → export/yükleme maliyet katmanları

Çıktılar: results/figures/fig-4.{1..4}-*.{pdf,png}

Not: Aynı (model, yığın, çözünürlük) için birden fazla gecikme kaydı varsa
EN DÜŞÜK medyan alınır (en-iyi-koşu; termal/arka plan gürültüsü ayıklanır).
Eksik veri dosyası varsa ilgili şekil atlanır ve uyarı basılır.
Betik idempotenttir; tekrar çalıştırınca üzerine yazar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"

# mIoU (ADE20K val, doğrulanmış — Faz 1)
MIOU = {"vmamba": 48.33, "swin": 44.32, "convnext": 45.42}

MODEL_LABEL = {"vmamba": "VMamba-T", "swin": "Swin-T", "convnext": "ConvNeXt-T"}
# Renk körü dostu (tab10'dan seçim)
MODEL_COLOR = {"vmamba": "#d62728", "swin": "#1f77b4", "convnext": "#2ca02c"}

STACK_LABEL = {
    "torch_cpu": "PyTorch CPU",
    "torch_mps": "PyTorch MPS",
    "ort_CPU": "ORT CPU",
    "coreml_all": "CoreML ALL",
    "coreml_cpu_and_gpu": "CoreML CPU+GPU",
    "coreml_cpu_only": "CoreML CPU",
}

# Model başına mevcut EN İYİ dağıtılabilir yığın (Şekil 4.1)
BEST_STACK = {
    "convnext": "coreml_cpu_and_gpu",
    "swin": "coreml_cpu_and_gpu",
    "vmamba": "torch_mps",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  yazıldı: {name}.pdf / .png")


def load_latency() -> dict[tuple[str, str, int], float]:
    """(model, stack, resolution) → en düşük medyan gecikme (ms)."""
    best: dict[tuple[str, str, int], float] = {}
    for model in ("convnext", "swin", "vmamba"):
        path = RAW / f"latency_matrix_{model}.jsonl"
        if not path.exists():
            print(f"UYARI: {path.name} yok, {model} gecikme verisi atlandı")
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (model, rec["stack"], int(rec["config"]["resolution"]))
            med = float(rec["latency"]["median_ms"])
            if key not in best or med < best[key]:
                best[key] = med
    return best


def fig_pareto(lat: dict[tuple[str, str, int], float]) -> None:
    pts_512 = {k: v for k, v in lat.items() if k[2] == 512}
    if not pts_512:
        print("UYARI: 512 gecikme verisi yok — fig-4.1 atlandı")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_best = n_other = 0
    for model in ("convnext", "swin", "vmamba"):
        color = MODEL_COLOR[model]
        best_stack = BEST_STACK[model]
        for (m, stack, _res), ms in sorted(pts_512.items()):
            if m != model:
                continue
            if stack == best_stack:
                ax.scatter(ms, MIOU[model], s=130, color=color, zorder=5,
                           edgecolors="black", linewidths=0.8,
                           label=MODEL_LABEL[model])
                ax.annotate(STACK_LABEL.get(stack, stack), (ms, MIOU[model]),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9, color=color, fontweight="bold")
                n_best += 1
            else:
                ax.scatter(ms, MIOU[model], s=28, color=color, alpha=0.35, zorder=3)
                ax.annotate(STACK_LABEL.get(stack, stack), (ms, MIOU[model]),
                            textcoords="offset points", xytext=(0, -12),
                            ha="center", fontsize=7.5, color=color, alpha=0.75)
                n_other += 1
    # VMamba notu
    key = ("vmamba", BEST_STACK["vmamba"], 512)
    if key in pts_512:
        ax.annotate("CoreML: dönüşemiyor", (pts_512[key], MIOU["vmamba"]),
                    textcoords="offset points", xytext=(12, -18), fontsize=9,
                    color=MODEL_COLOR["vmamba"], style="italic",
                    arrowprops=dict(arrowstyle="-", color=MODEL_COLOR["vmamba"],
                                    alpha=0.5, lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("Medyan gecikme (ms, log ölçek)")
    ax.set_ylabel("mIoU (ADE20K val, %)")
    ax.set_ylim(43.0, 49.5)
    ax.legend(loc="upper left", framealpha=0.9)
    save(fig, "fig-4.1-pareto")
    print(f"  fig-4.1: {n_best} büyük + {n_other} soluk nokta")


def fig_scaling(lat: dict[tuple[str, str, int], float]) -> None:
    stacks = ("torch_cpu", "torch_mps")
    resolutions = (256, 512, 768, 1024)
    series = []
    for model in ("convnext", "swin", "vmamba"):
        for stack in stacks:
            xs = [r for r in resolutions if (model, stack, r) in lat]
            ys = [lat[(model, stack, r)] for r in xs]
            if xs:
                series.append((model, stack, xs, ys))
            missing = [r for r in resolutions if r not in xs]
            if missing:
                print(f"  fig-4.2 eksik: {model}/{stack} @ {missing}")
    if not series:
        print("UYARI: ölçeklendirme verisi yok — fig-4.2 atlandı")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_pts = 0
    for model, stack, xs, ys in series:
        ls = "--" if stack == "torch_cpu" else "-"
        marker = "s" if stack == "torch_cpu" else "o"
        ax.plot(xs, ys, ls, marker=marker, color=MODEL_COLOR[model],
                markersize=6, linewidth=1.6,
                label=f"{MODEL_LABEL[model]} · {STACK_LABEL.get(stack, stack)}")
        n_pts += len(xs)
    ax.set_yscale("log")
    ax.set_xticks(list(resolutions))
    ax.set_xticklabels([f"{r}²" for r in resolutions])
    ax.set_xlabel("Girdi çözünürlüğü (piksel)")
    ax.set_ylabel("Medyan gecikme (ms, log ölçek)")
    ax.legend(ncols=2, framealpha=0.9)
    save(fig, "fig-4.2-scaling")
    print(f"  fig-4.2: {len(series)} seri, {n_pts} nokta")


def load_export() -> list[dict]:
    path = RAW / "export_matrix.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fig_cost_layers(rows: list[dict]) -> None:
    """4 katman (ONNX hattı, 512²): export süresi, graf boyutu, ORT yükleme, ilk çıkarım."""
    if not rows:
        print("UYARI: export_matrix.jsonl yok — fig-4.3 atlandı")
        return

    def pick(model: str, stage: str, field: str) -> float | None:
        vals = [r.get(field) for r in rows
                if r.get("model") == model and r.get("format") == "onnx"
                and r.get("stage") == stage and r.get(field) is not None
                and r.get("ok", True)]
        return vals[-1] if vals else None

    models = ("convnext", "swin", "vmamba")
    layers = [
        ("Export süresi (s)", lambda m: pick(m, "export", "s")),
        ("Graf boyutu (MB)", lambda m: pick(m, "export", "size_mb")),
        ("ORT yükleme (s)", lambda m: pick(m, "ort_load", "s")),
        ("İlk çıkarım (ms)", lambda m: (lambda v: v * 1000 if v is not None else None)(
            pick(m, "ort_first_run", "s"))),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    width = 0.26
    n_bars = 0
    import numpy as np

    x = np.arange(len(layers))
    for i, model in enumerate(models):
        vals, pos = [], []
        for j, (_lbl, fn) in enumerate(layers):
            v = fn(model)
            if v is None:
                print(f"  fig-4.3 eksik: {model} / {_lbl}")
                continue
            vals.append(v)
            pos.append(x[j] + (i - 1) * width)
        bars = ax.bar(pos, vals, width, color=MODEL_COLOR[model],
                      label=MODEL_LABEL[model], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:g}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=8)
        n_bars += len(vals)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in layers], fontsize=9.5)
    ax.set_ylabel("Değer (log ölçek, katman birimi x etiketinde)")
    ax.legend(framealpha=0.9)
    save(fig, "fig-4.3-cost-layers")
    print(f"  fig-4.3: {n_bars} çubuk")


def fig_energy() -> None:
    path = RAW / "energy_matrix.json"
    if not path.exists():
        print("UYARI: energy_matrix.json yok — fig-4.4 atlandı")
        return
    data = json.loads(path.read_text())
    cells = data.get("cells", [])
    if not cells:
        print("UYARI: enerji hücresi yok — fig-4.4 atlandı")
        return
    cells = sorted(cells, key=lambda c: c["mj_per_inf"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels, vals, colors, hatches = [], [], [], []
    ane_any = False
    for c in cells:
        model, stack = c["label"].split("/", 1)
        labels.append(f"{MODEL_LABEL.get(model, model)}\n{STACK_LABEL.get(stack, stack)}")
        vals.append(c["mj_per_inf"])
        colors.append(MODEL_COLOR.get(model, "#7f7f7f"))
        ane = c["power"].get("ane_mw", 0) > 0
        hatches.append("//" if ane else "")
        ane_any = ane_any or ane
    bars = ax.bar(range(len(vals)), vals, color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.6)
    for b, c in zip(bars, cells):
        ax.annotate(f"{c['net_power_mw'] / 1000:.1f} W",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Enerji (mJ / çıkarım)")
    ax.set_yscale("log")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLOR[m],
                             edgecolor="black", linewidth=0.6, label=MODEL_LABEL[m])
               for m in ("convnext", "swin", "vmamba")]
    if ane_any:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="//",
                                     edgecolor="black", linewidth=0.6,
                                     label="ANE aktif"))
    ax.legend(handles=handles, framealpha=0.9, loc="upper left")
    save(fig, "fig-4.4-energy")
    print(f"  fig-4.4: {len(vals)} çubuk (ANE aktif: {sum(1 for h in hatches if h)})")


# fp32 CoreML paket boyutları (MB) — export_matrix'ten doğrulanmış sabitler
QUANT_FP32_SIZE = {"convnext": 118.8, "swin": 127.5}

# ORT dynamic INT8 medyan gecikmeleri (ms) — nicemleme-sonuclari.md, 512²
ORT_INT8_LATENCY = {
    "convnext": (10863.0, 644.0),  # (int8_ms, fp32_ms)
    "swin": (7094.0, 714.0),
    "vmamba": (3752.0, 618.0),
}


def fig_quant() -> None:
    """(a) boyut vs mIoU (CoreML fp32/W8/W4), (b) ORT INT8 gecikme çarpanı."""
    quant_path = RAW / "quant_matrix.jsonl"
    if not quant_path.exists():
        print("UYARI: quant_matrix.jsonl yok — fig-4.5 atlandı")
        return
    size: dict[tuple[str, str], float] = {}
    for line in quant_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("backend") == "coreml" and rec.get("ok") and rec.get("size_mb"):
            size[(rec["model"], rec["mode"])] = float(rec["size_mb"])

    def miou(model: str, tag: str) -> float | None:
        path = RAW / f"quant_miou_{model}_{tag}.json"
        if not path.exists():
            print(f"  fig-4.5 eksik: {path.name}")
            return None
        return float(json.loads(path.read_text())["mIoU"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4.2))
    n_pts = 0
    variants = [
        ("fp32", "fp32export", lambda m: QUANT_FP32_SIZE.get(m)),
        ("W8", "w8", lambda m: size.get((m, "w8_linear"))),
        ("W4", "w4", lambda m: size.get((m, "w4_palette"))),
    ]
    for model in ("convnext", "swin"):
        xs, ys, lbls = [], [], []
        for lbl, tag, size_fn in variants:
            s, mi = size_fn(model), miou(model, tag)
            if s is None or mi is None:
                print(f"  fig-4.5 eksik: {model}/{lbl}")
                continue
            xs.append(s)
            ys.append(mi)
            lbls.append(lbl)
        ax_a.plot(xs, ys, "-o", color=MODEL_COLOR[model], markersize=7,
                  linewidth=1.6, label=MODEL_LABEL[model])
        for x_, y_, lbl in zip(xs, ys, lbls):
            ax_a.annotate(lbl, (x_, y_), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=8,
                          color=MODEL_COLOR[model])
        n_pts += len(xs)
    ax_a.set_xlabel("Paket boyutu (MB)")
    ax_a.set_ylabel("mIoU (ADE20K val, square-512, %)")
    ax_a.set_title("(a) Nicemleme: boyut ↔ doğruluk (CoreML)", fontsize=10)
    ax_a.legend(framealpha=0.9)

    models_b = [m for m in ("convnext", "swin", "vmamba") if m in ORT_INT8_LATENCY]
    factors = [ORT_INT8_LATENCY[m][0] / ORT_INT8_LATENCY[m][1] for m in models_b]
    bars = ax_b.bar(range(len(models_b)), factors,
                    color=[MODEL_COLOR[m] for m in models_b],
                    edgecolor="black", linewidth=0.6)
    for b, f in zip(bars, factors):
        ax_b.annotate(f"{f:.1f}×", (b.get_x() + b.get_width() / 2, b.get_height()),
                      textcoords="offset points", xytext=(0, 3),
                      ha="center", fontsize=9, fontweight="bold")
    ax_b.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_b.annotate("1× = fp32 ile eşit", (len(models_b) - 0.5, 1.0),
                  textcoords="offset points", xytext=(0, 4), ha="right",
                  fontsize=8, style="italic")
    ax_b.set_xticks(range(len(models_b)))
    ax_b.set_xticklabels([MODEL_LABEL[m] for m in models_b], fontsize=9)
    ax_b.set_ylabel("INT8 gecikme / fp32 gecikme (kat)")
    ax_b.set_title("(b) ORT dynamic INT8: 'pesimizasyon' (512²)", fontsize=10)
    n_pts += len(factors)
    fig.tight_layout()
    save(fig, "fig-4.5-quant")
    print(f"  fig-4.5: (a) {n_pts - len(factors)} nokta + (b) {len(factors)} çubuk")


def fig_outliers() -> None:
    """Çözünürlüğe karşı en kötü katman: kurtosis (sol) ve chmax/median (sağ)."""
    resolutions = (256, 512, 768)
    series: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for model in ("convnext", "swin", "vmamba"):
        path = RAW / f"activation_stats_{model}.json"
        if not path.exists():
            print(f"UYARI: {path.name} yok — {model} fig-4.6'dan atlandı")
            continue
        stats = json.loads(path.read_text())["stats"]
        xs, kurt, chm = [], [], []
        for res in resolutions:
            layers = stats.get(str(res))
            if not layers:
                print(f"  fig-4.6 eksik: {model} @ {res}")
                continue
            xs.append(res)
            kurt.append(max(v["kurtosis"] for v in layers.values()))
            chm.append(max(v["chmax_over_median"] for v in layers.values()))
        if xs:
            series[model] = (xs, kurt, chm)
    if not series:
        print("UYARI: aktivasyon istatistiği yok — fig-4.6 atlandı")
        return
    fig, (ax_k, ax_c) = plt.subplots(1, 2, figsize=(10, 4.2))
    n_pts = 0
    for model, (xs, kurt, chm) in series.items():
        ax_k.plot(xs, kurt, "-o", color=MODEL_COLOR[model], markersize=6,
                  linewidth=1.6, label=MODEL_LABEL[model])
        ax_c.plot(xs, chm, "-o", color=MODEL_COLOR[model], markersize=6,
                  linewidth=1.6, label=MODEL_LABEL[model])
        n_pts += 2 * len(xs)
    for ax, ylabel, title in (
        (ax_k, "En kötü katman kurtosis", "(a) Aykırılık şiddeti"),
        (ax_c, "En kötü katman chmax / medyan", "(b) Kanal dengesizliği"),
    ):
        ax.set_yscale("log")
        ax.set_xticks(list(resolutions))
        ax.set_xticklabels([f"{r}²" for r in resolutions])
        ax.set_xlabel("Girdi çözünürlüğü (piksel)")
        ax.set_ylabel(ylabel + " (log ölçek)")
        ax.set_title(title, fontsize=10)
        ax.legend(framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig-4.6-outliers")
    print(f"  fig-4.6: {len(series)} model, {n_pts} nokta")


def fig_reform() -> None:
    """Yeniden formülasyon: ONNX düğüm sayısı + CoreML ANE/GPU gecikmesi."""
    path = RAW / "reform_matrix.jsonl"
    if not path.exists():
        print("UYARI: reform_matrix.jsonl yok — fig-4.7 atlandı")
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    forms = ("seq", "blocked32", "blocked64", "blocked128", "ane32")
    FORM_LABEL = {
        "seq": "seq", "blocked32": "blok-32", "blocked64": "blok-64",
        "blocked128": "blok-128", "ane32": "ANE-32",
    }

    def nodes(form: str) -> float | None:
        vals = [r["num_nodes"] for r in rows
                if r.get("form") == form and r.get("stage") == "onnx_graph"
                and r.get("num_nodes") is not None]
        return vals[-1] if vals else None

    def latency(form: str, stage: str) -> float | None:
        vals = [float(r["median_ms"]) for r in rows
                if r.get("form") == form and r.get("stage") == stage
                and r.get("ok") and r.get("median_ms") is not None]
        return min(vals) if vals else None

    import numpy as np

    fig, (ax_n, ax_l) = plt.subplots(1, 2, figsize=(10, 4.2))
    n_pts = 0

    node_vals = []
    for form in forms:
        v = nodes(form)
        if v is None:
            print(f"  fig-4.7 eksik: {form} / onnx_graph")
        node_vals.append(v)
    xs = [i for i, v in enumerate(node_vals) if v is not None]
    ys = [v for v in node_vals if v is not None]
    bars = ax_n.bar(xs, ys, color="#7f7f7f", edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, ys):
        ax_n.annotate(f"{v:g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                      textcoords="offset points", xytext=(0, 3),
                      ha="center", fontsize=8)
    n_pts += len(ys)
    ax_n.set_yscale("log")
    ax_n.set_xticks(range(len(forms)))
    ax_n.set_xticklabels([FORM_LABEL[f] for f in forms], fontsize=9)
    ax_n.set_ylabel("ONNX düğüm sayısı (log ölçek)")
    ax_n.set_title("(a) Graf karmaşıklığı", fontsize=10)

    width = 0.38
    x = np.arange(len(forms))
    for i, (stage, lbl, color) in enumerate((
        ("coreml_all", "CoreML ALL (ANE)", "#9467bd"),
        ("coreml_cpu_and_gpu", "CoreML CPU+GPU", "#ff7f0e"),
    )):
        pos, vals = [], []
        for j, form in enumerate(forms):
            v = latency(form, stage)
            if v is None:
                print(f"  fig-4.7 eksik: {form} / {stage}")
                continue
            pos.append(x[j] + (i - 0.5) * width)
            vals.append(v)
        bars = ax_l.bar(pos, vals, width, color=color, label=lbl,
                        edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax_l.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                          textcoords="offset points", xytext=(0, 2),
                          ha="center", fontsize=7.5)
        n_pts += len(vals)
    ax_l.set_yscale("log")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([FORM_LABEL[f] for f in forms], fontsize=9)
    ax_l.set_ylabel("Medyan gecikme (ms, log ölçek)")
    ax_l.set_title("(b) CoreML gecikmesi", fontsize=10)
    ax_l.legend(framealpha=0.9)
    fig.tight_layout()
    save(fig, "fig-4.7-reform")
    print(f"  fig-4.7: {n_pts} çubuk")


def main() -> int:
    print(f"Şekiller → {FIG}")
    lat = load_latency()
    fig_pareto(lat)
    fig_scaling(lat)
    fig_cost_layers(load_export())
    fig_energy()
    fig_quant()
    fig_outliers()
    fig_reform()
    return 0


if __name__ == "__main__":
    sys.exit(main())
