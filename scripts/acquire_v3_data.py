"""Pinned public subset acquisition, all within the 15 GB workspace budget."""
from scripts.resources import download
from scripts.prepare_v2_data import main as prepare_two
from scripts.prepare_v3_data import main as prepare_three

def main():
    revision='230a2d4fac8900333c61754641c7a13e069ac9c6'
    root=f'https://huggingface.co/datasets/OpenGVLab/MVBench/resolve/{revision}'
    download(root+'/json/fine_grained_action.json','data/fine_grained_action.json',48694)
    download(root+'/video/Moments_in_Time_Raw.zip','data/moments.zip',67108520)
    prepare_two();prepare_three()

if __name__=='__main__':main()
