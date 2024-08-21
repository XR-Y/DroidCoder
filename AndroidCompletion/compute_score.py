import os
import pickle
import editdistance
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from utils import Tools

compute_first_line = False
def compute_EM(target, predictions, passk):
    target_lines = [line.strip() for line in target.strip().splitlines() if line.strip()]
    if compute_first_line:
        target_lines = [target_lines[0]] if len(target_lines) > 1 else target_lines
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
        if compute_first_line:
            prediction_lines = prediction_lines[:len(target_lines)]
            score = 1 if prediction_lines == target_lines else 0
            EM_scores.append(score)
            continue
        score = 0
        len_pred = len(prediction_lines)
        if len_pred == 0:
            EM_scores.append(0)
            continue
        for line in target_lines:
            if line in prediction_lines:
                score += 1
        EM_scores.append(score / len(target_lines) * min(1.0, len(target_lines) / len_pred))
    return max(EM_scores)

def compute_ES(target, predictions, passk):
    target_lines = [line.strip() for line in target.strip().splitlines() if line.strip()]
    if compute_first_line:
        target_lines = [target_lines[0]] if len(target_lines) > 1 else target_lines
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
        if compute_first_line:
            prediction_lines = prediction_lines[:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)


def compute_score_by_repo_with_metadata(app_types, repos, fname, passk=1):
    print(f'Result: {fname[fname.rfind("/") + 1:fname.rfind(".")]}')
    lines = Tools.load_jsonl(fname)
    em_scores = defaultdict(list)
    es_scores = defaultdict(list)
    es_android = list()
    es_java = list()
    if "unknown" in fname:
        tasks = pickle.load(open("datasets/pkls_new/base_test.pkl", "rb"))
    else:
        tasks = pickle.load(open("datasets/pkls/base_test.pkl", "rb"))
    demo = {}
    for t in tasks:
        demo[t['task_id']] = t['type']
    for line in lines:
        line['metadata']['ground_truth'] = Tools.removeComments(line['metadata']['ground_truth'])
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos or line['metadata']['task_id'] not in demo:
            continue
        samples = [Tools.removeComments(line['choices'][i]['text']) for i in range(len(line['choices']))]
        if "prompt" in fname:
            samples = [Tools.removeNewMethods(sample) for sample in samples]
        em_tmp = compute_EM(line['metadata']['ground_truth'], samples, passk)
        es_tmp = compute_ES(line['metadata']['ground_truth'], samples, passk)
        # gpt and starcoder is decoder-only, calculate again
        if not compute_first_line or "gpt-prompt" in fname or "starcoderbase" in fname:
            # some samples like gpt have same copy, remove them
            samples = [Tools.removeDuplicate(s, line['prompt']) for s in samples]
            target = '# Determine if the information above is useful, Complete The Following Android Code:'
            completion_index = line['prompt'].find(target)
            if completion_index != -1:
                line['prompt'] = line['prompt'][completion_index + len(target) + 1:]
            # some samples like gpt often return full code, calculate again
            prompt = line['metadata']['ground_truth']
            em_tmp = max(em_tmp, compute_EM(prompt, samples, passk))
            es_tmp = max(es_tmp, compute_ES(prompt , samples, passk))
        em_scores[repo].append(em_tmp)
        es_scores[repo].append(es_tmp)
        if demo[line['metadata']['task_id']] == 'android':
            es_android.append(es_tmp)
        else:
            es_java.append(es_tmp)
    em_avg_scores = {repo: round(max(sum(em_scores[repo]), 1) / len(em_scores[repo]), 4) for repo in em_scores}
    es_avg_scores = {repo: round(sum(es_scores[repo]) / len(es_scores[repo]), 4) for repo in es_scores}
    repo_count = {repo: len(em_scores[repo]) for repo in em_scores}
    print(f'EM\tES\tCount\tType\tRepo')
    for repo in repos:
        repo_type = ""
        try:
            for k, v in app_types.items():
                if repo in v:
                    repo_type = k
            print(f'{em_avg_scores[repo]:.4f}\t{es_avg_scores[repo]:.4f}\t{repo_count[repo]}\t{repo_type}\t{repo}')
        except KeyError:
            print(f'0.0000\t0.0000\t0\t{repo_type}\t{repo}')
    # avg_em = round(sum(num for sub in em_scores.values() for num in sub) / sum(len(sub) for sub in em_scores.values()), 4)
    # avg_es = round(sum(num for sub in es_scores.values() for num in sub) / sum(len(sub) for sub in es_scores.values()), 4)

def viewResults(arr1, arr2):
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 20)
    ax.hist(arr1, bins, alpha=0.5, label='RAG SFT')
    ax.hist(arr2, bins, alpha=0.5, label='SFT')
    ax.legend()
    ax.set_title('Value Distribution of Arrays')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    plt.show()

def compare_score(repos, passk=1):
    y_lines = Tools.load_jsonl('../predictions/codeT-prompt.jsonl')
    x_lines = Tools.load_jsonl('../predictions/codet5p-770m-tfidf-type-all-one.jsonl')
    x_lines = {line['metadata']['task_id']:line for line in x_lines}
    y_lines = {line['metadata']['task_id']:line for line in y_lines}
    threshold = 0.2
    x_cnt = 0
    y_cnt = 0
    ans = []
    x_results, y_results = [], []
    for key, line in x_lines.items():
        repo = line['metadata']['task_id'].split('/')[0]
        line['metadata']['ground_truth'] = Tools.removeComments(line['metadata']['ground_truth'])
        if repo not in repos or key not in y_lines:
            continue
        x_em = compute_EM(line['metadata']['ground_truth'], [line['choices'][i]['text'] for i in range(len(line['choices']))], passk)
        x_es = compute_ES(line['metadata']['ground_truth'], [line['choices'][i]['text'] for i in range(len(line['choices']))], passk)
        y_em = compute_EM(line['metadata']['ground_truth'], [y_lines[key]['choices'][i]['text'] for i in range(len(y_lines[key]['choices']))], passk)
        y_es = compute_ES(line['metadata']['ground_truth'], [y_lines[key]['choices'][i]['text'] for i in range(len(y_lines[key]['choices']))], passk)
        x_results.append(x_es)
        y_results.append(y_es)
        if y_es < threshold:
            y_cnt += 1
        # if x_em == 1 and y_es < threshold and len(line['metadata']['ground_truth'].split('\n')) > 4:
        # if "log" in line['choices'][0]['text'].lower():
        # if y_es - x_es > threshold and repo == "Tusky":
        # if x_es != 0.1:
        if x_em == 1 and y_es < 0.8:
        # if x_es < 1 - threshold < y_es:
            # print(f'{x_em}\t{x_es}\t{y_em}\t{y_es}\t{line["metadata"]["task_id"]}')
            prompt = line['prompt'].replace('[SEP]', '\n')
            ground_truth = line['metadata']['ground_truth']
            x_prediction = line['choices'][0]['text']
            y_prediction = y_lines[key]['choices'][0]['text']
            x_cnt += 1
            print(f"Prompt: {prompt}\nGround truth: {ground_truth}\nMore: {y_prediction}\nOutput: {x_prediction}")
            print('---')
            ans.append(line)
    print(f'{x_cnt} {y_cnt} / {len(x_lines)}')

if __name__ == '__main__':
    app_types = {}
    repos = []
    base_dir = os.getcwd()[:os.getcwd().rfind(os.sep) + 1] + "datasets" # the path of the base apps dataset
    directories = ["notes", "media", "life", "communication"]
    for directory in directories:
        path = os.path.join(base_dir, directory, "test")
        app_types[directory] = os.listdir(path)
        repos.extend(app_types[directory])
    file_path = '../predictions/CodeGPT-small-java-adaptedGPT2-tfidf-type-all.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codegpt-prompt.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/starcoderbase-1b-tfidf-type-all.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/starcoderbase-prompt.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-770m-tfidf-type-all-one.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-without.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-noandroid.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-nometadata.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-notfidf.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    # file_path = './predictions/starcoder-prompt.jsonl'
    # compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    # print('---')
    file_path = '../predictions/codeT-prompt.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/gpt-prompt.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)

    print('\nBelow is the result of apps with different topk:\n')
    file_path = '../predictions/codet5p-220m-tfidf-type-all.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-topk5.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-topk20.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-topk50.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)

    path = os.path.join(base_dir, "new", "test")
    repos = os.listdir(path)
    print('\nBelow is the result of apps with unknown types:\n')
    file_path = '../predictions/codet5p-770m-tfidf-type-all-unknown.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/starcoderbase-1b-tfidf-type-all-unknown.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codet5p-220m-tfidf-type-all-unknown.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/CodeGPT-small-java-adaptedGPT2-tfidf-type-all-unknown.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/codeT-prompt-unknown.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)
    print('---')
    file_path = '../predictions/gpt-prompt-unknown.jsonl'
    compute_score_by_repo_with_metadata(app_types, repos, file_path, passk=1)