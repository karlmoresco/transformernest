import json
import regex as re
import numpy as np


def relu(x):
    return np.maximum(0, x)


def normal_init(m, n, std):
    """Normal distribution initialization (0-centered)"""
    return np.random.randn(m, n) * std


def he_init(fan_in, fan_out):
    """He initialization"""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)


def clean_dataset(input_path, output_path):
    """Clean raw data corpus"""
    with open(input_path, "r", encoding="utf-8") as f_in:
        data = f_in.read()

    # Alternative data cleaning method
    # import unicodedata
    # data = unicodedata.normalize("NFKC", data)
    # data = ''.join(c for c in data if not unicodedata.category(c).startswith('C'))

    data = re.sub(r'\s+', ' ', data)
    data = data.strip()

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(data)

    return data
