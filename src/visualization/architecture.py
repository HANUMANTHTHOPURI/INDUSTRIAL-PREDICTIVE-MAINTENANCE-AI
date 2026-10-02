from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "system_architecture.png"
)


def add_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    subtitle,
    facecolor,
    edgecolor,
):
    """Add a colored rounded architecture component."""

    # Subtle shadow
    shadow = FancyBboxPatch(
        (x + 0.06, y - 0.06),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=0,
        facecolor="#D1D5DB",
        alpha=0.45,
        zorder=1,
    )

    # Main colored box
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )

    ax.add_patch(shadow)
    ax.add_patch(box)

    # Box heading
    ax.text(
        x + width / 2,
        y + height * 0.65,
        title,
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#000000",
        zorder=3,
    )

    # Box description
    ax.text(
        x + width / 2,
        y + height * 0.29,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color="#000000",
        zorder=3,
    )


def add_arrow(ax, start, end):
    """Connect architecture stages."""

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.8,
            "color": "#000000",
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=0,
    )


def create_architecture_diagram():
    """Generate the production ML architecture diagram."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Slightly larger figure for larger fonts
    fig, ax = plt.subplots(figsize=(13, 16))

    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    ax.set_xlim(0, 13)

    # Increased vertical coordinate space
    ax.set_ylim(0, 18.5)

    ax.axis("off")

    # Larger boxes
    width = 6.2
    height = 1.15
    x = 3.4

    stages = [
        (
            15.5,
            "AI4I 2020 Dataset",
            "10,000 industrial operating records",
            "#DBEAFE",
            "#2563EB",
        ),
        (
            13.8,
            "Data Preparation",
            "Feature selection + leakage prevention",
            "#E0F2FE",
            "#0284C7",
        ),
        (
            12.1,
            "Physics-Informed Feature Engineering",
            "Mechanical Power: P = torque x (2*pi*RPM/60)",
            "#FFEDD5",
            "#EA580C",
        ),
        (
            10.4,
            "Preprocessing",
            "One-hot encoding + numerical scaling",
            "#DCFCE7",
            "#16A34A",
        ),
        (
            8.7,
            "Random Forest Model",
            "700 trees | class_weight = balanced",
            "#D1FAE5",
            "#059669",
        ),
        (
            7.0,
            "Model Evaluation",
            "Precision | Recall | F1 | ROC-AUC | PR-AUC",
            "#EDE9FE",
            "#7C3AED",
        ),
        (
            5.3,
            "Explainability",
            "Feature importance + SHAP",
            "#F3E8FF",
            "#9333EA",
        ),
        (
            3.6,
            "Model Artifact + Metadata",
            "Joblib model + JSON model lineage",
            "#FEF3C7",
            "#D97706",
        ),
        (
            1.9,
            "FastAPI Inference Service",
            "Pydantic validation + logging",
            "#CFFAFE",
            "#0891B2",
        ),
        (
            0.2,
            "Docker Deployment",
            "Portable containerized prediction API",
            "#E0E7FF",
            "#4F46E5",
        ),
    ]

    # Draw boxes
    for (
        y,
        title,
        subtitle,
        facecolor,
        edgecolor,
    ) in stages:
        add_box(
            ax,
            x,
            y,
            width,
            height,
            title,
            subtitle,
            facecolor,
            edgecolor,
        )

    # Draw arrows
    for i in range(len(stages) - 1):
        current_y = stages[i][0]
        next_y = stages[i + 1][0]

        add_arrow(
            ax,
            (
                x + width / 2,
                current_y,
            ),
            (
                x + width / 2,
                next_y + height,
            ),
        )

    # Main title
    ax.text(
        6.5,
        18.0,
        "Industrial AI Predictive Maintenance System",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#000000",
    )

    # Main subtitle
    ax.text(
        6.5,
        17.55,
        "Physics-Informed Machine Learning to Production Deployment",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#000000",
    )

    # Technology stack
    ax.text(
        6.5,
        17.10,
        "Python  •  Scikit-learn  •  SHAP  •  FastAPI  •  Docker",
        ha="center",
        va="center",
        fontsize=11,
        color="#000000",
    )

    # Save high-resolution image
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    print(
        f"Architecture diagram saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    create_architecture_diagram()