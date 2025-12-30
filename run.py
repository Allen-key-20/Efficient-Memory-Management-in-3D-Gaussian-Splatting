import subprocess
import time
from our_utils import compare

# dateset = 'm360'
model = ""

# colmap_path = rf"../data/{dateset}/{model}"
colmap_path = rf"../data/{model}"
outputdir = rf"./output"


command1 = f'python train.py -s {colmap_path} -m {outputdir}/{model} --eval '
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


