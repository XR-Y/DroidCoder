import glob
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import trim_mean
from sklearn.feature_extraction.text import TfidfVectorizer
from prepare_dataset import read_code
from utils import Tools

local_dir = ""

def iterate_repository(base_dir, repo):
    pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.kt")
    files = glob.glob(pattern, recursive=True)
    files.extend(glob.glob(os.path.join(f'{base_dir}/{repo}', "**", "*.java"), recursive=True))

    skipped_files = []
    loaded_code_files = []

    for fpath in files:
        try:
            code = read_code(fpath)
            fpath_tuple = tuple(os.path.normpath(fpath).split(os.sep)[len(os.path.normpath(base_dir).split(os.sep)):])
            for i, c in enumerate(code):
                loaded_code_files.append({
                    "context": c,
                    "repo": repo.split('/')[0],
                    "fpath_tuple": fpath_tuple
                })
        except Exception as e:
            skipped_files.append((fpath, e))
            continue

    if len(skipped_files) > 0:
        print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
        for fpath, e in skipped_files:
            print(f"{fpath}: {e}")
    return loaded_code_files


def getFunctions(base_dirs):
    if not isinstance(base_dirs, list):
        base_dirs = [base_dirs]
    result = []
    for base_dir in base_dirs:
        items = os.listdir(base_dir)
        repos = [item for item in items if os.path.isdir(os.path.join(base_dir, item))]
        for repo in repos:
            result.extend(iterate_repository(base_dir, repo))
    return result

def extractWords(s):
    words = re.findall(r'[A-Za-z][a-z]*|[A-Z][a-z]*|R', s)
    words = [word.lower() for word in words if len(word) > 1 or word == 'R']
    return ' '.join(words)

def getIDFs(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_
    idf_smoothed_values = [val + 1 for val in idf_values]
    return dict(zip(feature_names, idf_smoothed_values))

def createIDFs():
    results = getFunctions([f"{local_dir}/top",f"{local_dir}/communication/train", f"{local_dir}/notes/train", f"{local_dir}/media/train", f"{local_dir}/life/train"])
    tokens = []
    for res in results:
        tokens.append(extractWords(res['context']))
    idfs = getIDFs(tokens)
    Tools.dump_json(idfs, 'idfs.json')
    idfs = sorted(idfs.items(), key=lambda x: x[1], reverse=False)[100:200]
    for k, v in idfs:
        print(f'{k}: {v}')

def checkConnections():
    tasks = Tools.load_json(os.path.join(os.getcwd(), "datasets", "all_methods.json"))
    methods = [method for task_methods in tasks.values() for method in task_methods]
    pattern = r'\b[A-Z]+\_[A-Z]+\b'
    records = {}
    for i, method in enumerate(methods):
        matches = re.findall(pattern, method['context'])
        if matches:
            target = ' '.join(Tools.getSplitWords(method['methodName']))
            if target not in records:
                records[target] = list()
            records[target].extend(matches)
            records[target] = list(set(records[target]))
    records = {k: ' '.join(v) for k, v in records.items()}
    Tools.dump_json(records, 'upper_tags.json')

def checkFreResources():
    tasks = Tools.load_json(os.path.join(os.getcwd(), "datasets", "all_methods.json"))
    methods = [method for task_methods in tasks.values() for method in task_methods]
    pattern = r'R\.\w+'
    records = defaultdict(set)
    for method in methods:
        matches = re.findall(pattern, method['context'])
        if matches:
            target = ' '.join(Tools.getSplitWords(method['methodName']))
            records[target].update(matches)
    records = {k: ' '.join(v) for k, v in records.items()}
    results = {}
    for v in records.values():
        for i in v.split(' '):
            results[i] = results.get(i, 0) + 1
    results = sorted(results.items(), key=lambda s: s[1], reverse=True)
    print(results[:10])

def dumpResourceConnections():
    tasks = Tools.load_json(os.path.join(os.getcwd(), "datasets", "all_methods.json"))
    methods = [method for task_methods in tasks.values() for method in task_methods]
    pattern = r'R\.\w+'
    flags = ["R.drawable", "R.layout", "R.anim"]
    records = {}
    for i, method in enumerate(methods):
        matches = re.findall(pattern, method['context'])
        matches = [m for m in matches if m in flags]
        if matches:
            target = ' '.join(Tools.getSplitWords(method['methodName']))
            if target not in records:
                records[target] = list()
            records[target].extend(matches)
            records[target] = list(set(records[target]))
    records = {k: ' '.join(v) for k, v in records.items()}
    Tools.dump_json(records, 'resources_connections.json')

def dumpResources():
    dumpResourceConnections()
    types = Tools.load_json(local_dir + os.path.sep + "repos.json")
    files = {}
    for t in types.keys():
        repos = types[t]["test"]
        for repo in repos:
            repo = str(Path(repo))
            fpath = os.path.join(local_dir, t, "test", repo)
            files[repo[:repo.find(os.sep)]] = getOneRepoResources(fpath)
    Tools.dump_json(files, 'resources.json')

def getOneRepoResources(fpath):
    files = []
    for root, dirs, files_list in os.walk(fpath):
        for dir_name in dirs:
            if dir_name in ['drawable', 'layout', 'anim']:
                sub_path = os.path.join(root, dir_name)
                sub_files = [dir_name + os.sep + file[:file.rfind(".")] for file in os.listdir(str(sub_path))]
                files.extend(sub_files)
    return files

def methodsStatics(fpath="./datasets/all_methods.json"):
    methods = Tools.load_json("./datasets/dataset_sim_train.json")
    print(f"Total dataset samples: {len(methods)}")
    methods = Tools.load_json(fpath)
    methods =  [value for values in methods.values() for value in values]
    lines = [len(m['context'].split('\n')) for m in methods]
    print(np.median(lines), np.mean(lines), trim_mean(lines, 0.1))

    fig, ax = plt.subplots()
    bins = np.linspace(0, 100, 20)
    ax.hist(lines, bins, alpha=0.5, label='Line Count')
    ax.legend()
    ax.set_title('Value Distribution of Arrays')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    plt.show()

# the data processing for FT2Ra
# def tokenize_methods_train(fpath="./datasets/all_methods.json"):
#     methods = Tools.load_json(fpath)
#     methods =  [value for values in methods.values() for value in values]
#     data = []
#     for idx, value in enumerate(methods):
#         tokens = parse_code(value['context'])
#         data.extend([t["code"] for t in tokens])
#         if idx % 1000 == 0:
#             print(f"{idx}/{len(methods)}")
#     test_file = os.path.join("datasets", "ft2ra_train.txt")
#     with open(test_file, "w", encoding="utf-8") as file:
#         for item in data:
#             file.write(json.dumps(item))
#             file.write("\n")
#
# def tokenize_methods_test(fpath="./datasets/dataset_sim_test.json", multi=True):
#     methods = Tools.load_json(fpath)
#     data = []
#     wrong = 0
#     for idx, value in enumerate(methods):
#         gt = [i.strip() for i in value['ground_truth'].splitlines()]
#         if not multi:
#             gt = gt[0]
#         else:
#             gt = '</n>'.join(gt)
#         monk = value['prompt'] + "@gt\n" + gt
#         tokens = parse_code(monk)
#         if len(tokens) == 0:
#             wrong += 1
#             continue
#         tokens = tokens[0]["code"]
#         monk = ' '.join(tokens)
#         gt = monk[monk.find('@ gt') + len('@ gt'): monk.rfind(tokens[-1])].replace('< / n >', '\n').strip()
#         if len(gt) < 5:
#             continue
#         data.append({
#             "input": monk[:monk.find('@ gt')].strip(),
#             "gt": gt,
#             "repo": value['fpath'][:value['fpath'].find(os.sep)],
#             "task_id": value['task_id']
#         })
#         if idx % 1000 == 0:
#             print(f"{idx} {wrong}/{len(methods)}")
#     fname = "ft2ra_test.txt" if not multi else "ft2ra_test_multi.txt"
#     test_file = os.path.join("datasets", fname)
#     with open(test_file, "w", encoding="utf-8") as file:
#         for item in data:
#             file.write(json.dumps(item))
#             file.write("\n")

if __name__ == '__main__':
    # createIDFs()
    # checkConnections()
    # checkFreResources()
    # dumpResources()
    methodsStatics()
    # tokenize_methods_train()
    # tokenize_methods_test()