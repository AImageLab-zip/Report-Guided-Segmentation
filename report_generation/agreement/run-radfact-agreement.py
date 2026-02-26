import agreement.radfact
from agreement.radfact import compute_radfact
from tqdm import tqdm
from pathlib import Path
import csv

old_path = Path('/home/ocarpentiero/PycharmProjects/Brain2Text/agreement/old')
new_path = Path('/home/ocarpentiero/PycharmProjects/Brain2Text/agreement/new_translated')
references = []
predictions = []

old_files = [file for folder in sorted(old_path.iterdir()) for file in folder.iterdir() if 'eng' in file.name]
new_files = [file for folder in sorted(new_path.iterdir()) for file in folder.iterdir() if 'eng' in file.name]
assert len(old_files) == len(new_files) == 10

for old,new in zip(old_files,new_files):
    assert old.parent.name == new.parent.name
for old,new in zip(old_files,new_files):
    with open(old,'r') as f:
        references.append(f.read())
    with open(new,'r') as f:
        predictions.append(f.read())

assert len(references) == len(predictions) == 10

models = ['radfact-new-27b:latest']
temperatures = [0.0,0.0,0.0]

results_file = '/home/ocarpentiero/PycharmProjects/Brain2Text/agreement/results.txt'
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "temp", "precision",'recall','F1'])

    for model in models:
        for temperature in temperatures:
            results = compute_radfact(predictions=predictions,
                            references=references,
                            ollama_url='http://localhost:11434',
                            radfact_model=model,
                            max_workers=10,
                            temperature=temperature)
            writer.writerow([model, temperature, results['radfact-precision'],results['radfact-recall'],results['radfact-f1']])
            print(f'num calls:{agreement.radfact.NUM_CALLS}, total tokens:{agreement.radfact.NUM_TOKENS}')
