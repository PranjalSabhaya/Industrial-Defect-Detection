from pathlib import Path
from datetime import datetime
import shutil
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_run(config_path: str) -> Path:
    """
    Creates a unique experiment run directory and saves config.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("experiments") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config used for this run
    shutil.copy(config_path, run_dir / "config.yaml")

    return run_dir