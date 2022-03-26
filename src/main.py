import json
from os import getcwd
from os.path import join
from mask import Mask

PROJECT_ROOT = join(getcwd(), "..", "images")

if __name__ == '__main__':
    with open(join(PROJECT_ROOT, "config.json"), 'r') as cfg:
        configuration = json.load(cfg)
    mask = Mask()