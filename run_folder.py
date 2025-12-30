import os
import subprocess
from our_utils import compare
import time

def func(dataset_name):
    datapath = rf"../data/{dataset_name}"

    resolutions = {
        'bicycle': 4,
        'flowers': 4,
        'garden': 4,
        'stump': 4,
        'treehill': 4,
        'bonsai': 2,
        'counter': 2,
        'kitchen': 2,
        'room': 2,

        'drjohnson': 1,
        'playroom': 1,

        'train': 1,
        'truck': 1,

    }
    for model in os.listdir(datapath):

        colmap_path = rf"{datapath}/{model}"
        outputdir = rf"./output"

        resolution = resolutions[model]
        if resolution != 1:
            image_dir = f"images_{resolution}"
        else:
            image_dir = "images"

        command1 = f'python train.py -s {colmap_path} -m {outputdir}/{model} -i {image_dir} --eval '

        start = time.time()
        result = subprocess.run(command1, shell=True)
        if result.returncode != 0:
            exit(1)
        end = time.time()
        cost = end - start

        command2 = f'python render.py -m {outputdir}/{model} --skip_train --quiet'
        # subprocess.run(command2, shell=True)

        command3 = f'python metrics.py -m {outputdir}/{model}'
        result = subprocess.run(command3, shell=True, capture_output=True, text=True, encoding='utf-8')
        with open(rf"./{outputdir}/{model}/0_metric.txt", "a", encoding="UTF-8") as rec:
            rec.write(f"{model}\n")
            rec.write(f"{cost}\n")
            rec.write(result.stdout)

        compare(model, outputdir)
        time.sleep(300)


func("tat")
func("db")
func("m360")







