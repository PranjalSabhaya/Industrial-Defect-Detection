from src.utils import load_config
from src.train import train_model

def run_training(config_path: str):
    config = load_config(config_path)
    train_model(config)

if __name__ == "__main__":
    run_training("config/local.yaml")
