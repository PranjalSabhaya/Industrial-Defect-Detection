import json
import tensorflow as tf
from src.utils import load_config


class ModelLoader:

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.model = None
        self.class_names = None

    def load(self):
        model_path = self.config["model"]["model_path"]

        self.model = tf.keras.models.load_model(model_path)

        with open("class_names.json", "r") as f:
            self.class_names = json.load(f)

    def get_model(self):
        return self.model

    def get_config(self):
        return self.config

    def get_class_names(self):
        return self.class_names
