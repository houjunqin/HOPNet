import argparse
from ruamel.yaml import YAML
import os
import pickle
from train import Trainer
import time
import sys
import types


    
parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='config.yaml', type=str,
                    help='Configuration filename for all models.')
parser.add_argument('--data', default='data', type=str, help='Path to store original data and processed zarr files.')
parser.add_argument('--dataset', choices=['KATRINA', 'IDA', 'HARVEY'], default='HARVEY', type=str, help='Which storm dataset to use.')
args = parser.parse_args(args=[])

yaml = YAML()
with open(args.config_filename, "r") as f:
    config = yaml.load(f)

processed_cache_path = f"HARVEY.pkl"


if os.path.exists(processed_cache_path):
    with open(processed_cache_path, "rb") as f:
        datasets = pickle.load(f)
    print(f"Loaded cached datasets from {processed_cache_path}")


solver = Trainer(datasets = datasets, **config)
solver.train()
#outputs = solver.test(time.time())
