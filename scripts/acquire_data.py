"""Acquire only the small public archive and annotations used in this study."""
import json
from scripts.resources import ROOT, download
from scripts.prepare_data import main as prepare

REVISION='230a2d4fac8900333c61754641c7a13e069ac9c6'

def main():
    files={'json/action_antonym.json':('data/action_antonym.json',53338),
           'video/ssv2_video.zip':('data/ssv2_video.zip',16819261)}
    for source,(target,size) in files.items():
        download(f'https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/{REVISION}/{source}',
                 target,size)
    prepare()

if __name__=='__main__':
    main()
