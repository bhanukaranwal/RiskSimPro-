import numpy as np
import json
import logging

class RNG:
    @staticmethod
    def seed(seed_value):
        np.random.seed(seed_value)

class Config:
    def __init__(self, config_dict=None):
        self.config = config_dict or {}

    def update(self, key, value):
        self.config[key] = value

    def get(self, key, default=None):
        return self.config.get(key, default)

class Logger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

class DataPreprocessor:
    @staticmethod
    def normalize(data):
        return (data - np.mean(data)) / np.std(data)

    @staticmethod
    def standardize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

class Serializer:
    @staticmethod
    def to_json(obj, filepath):
        with open(filepath, 'w') as f:
            json.dump(obj, f)

    @staticmethod
    def from_json(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
