import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import io
import base64

def _generate_chart(data: list[dict],
    x_key: str,
    y_key: str,
    chart_type: str = "bar",
    title: str = "",
) -> str:
    """
    Renders `data` as a chart and returns a base64-encoded PNG string
    (no 'data:image/png;base64,' prefix — caller/frontend adds that).
    """
    x_vals = [row[x_key] for row in data]
    y_vals = [row[y_key] for row in data]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if chart_type == "bar":
        ax.bar(x_vals, y_vals)
    elif chart_type == "line":
        ax.plot(x_vals, y_vals, marker="o")
    elif chart_type == "pie":
        ax.pie(y_vals, labels=x_vals, autopct="%1.1f%%")
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")
    ax.set_title(title)
    if chart_type != "pie":
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)  # important — prevents figure accumulation/memory leak across calls
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")

