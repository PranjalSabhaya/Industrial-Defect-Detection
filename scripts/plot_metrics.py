from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(run_dir: Path):
    metrics_path = run_dir / "metrics.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.csv found in {run_dir}")

    df = pd.read_csv(metrics_path)

    # ---- Accuracy plot ----
    plt.figure()
    plt.plot(df["epoch"], df["accuracy"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "accuracy.png")
    plt.close()

    # ---- Loss plot ----
    plt.figure()
    plt.plot(df["epoch"], df["loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / "loss.png")
    plt.close()

    print(f"📊 Plots saved in: {plots_dir}")


def main():
    experiments_dir = Path("experiments")
    runs = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir()],
        reverse=True
    )

    if not runs:
        raise RuntimeError("No experiment runs found")

    latest_run = runs[0]
    print(f"Using latest run: {latest_run.name}")

    plot_metrics(latest_run)


if __name__ == "__main__":
    main()
