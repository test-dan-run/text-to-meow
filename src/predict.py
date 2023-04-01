import os
import json
import random
import librosa
import numpy as np
import gradio as gr
from typing import Any, List, Dict, Tuple

from utils import meow_stretch, get_word_lengths
from config import config, BaseConfig

''' Gradio Input/Output Configurations '''
inputs: str = 'text'
outputs: gr.Audio = gr.Audio()

def load_meows(cfg: BaseConfig) -> List[Dict[str, Any]]:

    meow_dir = os.path.dirname(cfg.manifest_path)

    with open(cfg.manifest_path, mode='r') as fr:
        lines = fr.readlines()

    items = []
    for line in lines:
        item = json.loads(line)
        item['audio'], item['rate'] = librosa.load(os.path.join(meow_dir, item['audio_filepath']), sr=None)
        items.append(item)

    return items

def extract_meows_weights(items: List[Dict[str, Any]]) -> Tuple[List[np.ndarray], List[float]]:
    meows = [item['audio'] for item in items]
    weights = [item['weight'] for item in items]
    return meows, weights

''' Load meows '''
meow_items = load_meows(config)
meows, weights = extract_meows_weights(meow_items)

def predict(text: str) -> str:

    word_lengths = get_word_lengths(text)
    selected_meows = random.choices(meows, weights=weights, k=len(word_lengths))
    transformed_meows = [
        meow_stretch(
            meow, wl,
            init_factor=config.init_factor,
            add_factor=config.add_factor,
            power_factor=config.power_factor
        ) for meow, wl in zip(selected_meows, word_lengths)
    ]

    result_meows = np.concatenate(transformed_meows, axis=0)

    return (config.sample_rate, result_meows)


