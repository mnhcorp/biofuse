# ---------------------------------------------------------------
#  S6 Fig – peak-VRAM bar chart (sorted + 5-GB reference line)
# ---------------------------------------------------------------
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# --------------------  DATA  ------------------------------------
data = pd.DataFrame({
    "model": [
        "BioMedCLIP", "PubMedCLIP", "CONCH", "rad-dino",
        "UNI", "UNI2", "Prov-GigaPath", "Hibou-B", "CheXagent"
    ],
    "vram": [1.2, 1.1, 2.3, 1.0, 1.5, 3.2, 5.0, 8.8, 33.8]  # GB
})
data = data.sort_values("vram")        # ascending

# split to colour CheXagent
colors = ["#CC0000" if m == "CheXagent" else "#D55E00" for m in data["model"]]

# --------------------  PLOT  ------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.bar(data["model"], data["vram"], color=colors)

ax.set_ylabel("Peak VRAM (GB)", color="#CC0000")
ax.set_ylim(0, max(data["vram"]) * 1.15)
ax.set_yticks(range(0, int(max(data["vram"]) * 1.15) + 5, 5))
ax.tick_params(axis="x", rotation=45)
for lbl in ax.get_xticklabels():
    lbl.set_horizontalalignment("right")

# 5-GB dashed reference
ax.axhline(5, ls="--", lw=1, color="gray")
ax.text(len(data) - 0.5, 5.2, "5 GB threshold", ha="right", va="bottom",
        fontsize=8, color="gray")

# annotate bars
for bar, val in zip(bars, data["vram"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.4,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=8)

# -------------------  CAPTION  ----------------------------------
label   = "S8 Fig. Peak GPU VRAM usage (GB) for each model during embedding extraction"
legend  = ("Memory requirements are consistent across datasets "
           "at steady state; values were measured on RetinaMNIST. "
           "CheXagent exceeds 33 GB, whereas most other backbones operate "
           "within a 5 GB envelope.")

# bold label
fig.text(0.5, -0.10, label,  ha="center", va="top",
         fontsize=9, fontweight="bold")
# legend (normal weight)
fig.text(0.5, -0.14, legend, ha="center", va="top",
         fontsize=9, wrap=True)

# shrink the reserved bottom margin from 27 % to 18 %
plt.tight_layout(rect=[0, 0.18, 1, 1])
out_dir = Path("figs")
fig.savefig(out_dir / "S8_Fig.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

print("✓ S8_Fig.pdf generated in", out_dir.resolve())