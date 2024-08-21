import argparse
import glob
import os
import pickle
import random
import re
import shutil
from collections import defaultdict
from math import exp
from sentence_transformers import SentenceTransformer
import numpy as np
import scipy
import tiktoken
import torch
from utils import Tools
import javalang
from torch.utils.data import Dataset
from fastbm25 import fastbm25
from loguru import logger

logger.add(f'logs/prepare_dataset.log', rotation="5 MB")
class SimilarityScore:
    @staticmethod
    def cosine_similarity(source, target):
        return 1 - scipy.spatial.distance.cosine(source, target)

    @staticmethod
    def sentence_base_similarity(nlp, sentence1, sentence2):
        doc1 = nlp(sentence1)
        doc2 = nlp(sentence2)
        vec1 = doc1.vector
        vec2 = doc2.vector
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        similarity = np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)
        return similarity


    @staticmethod
    def jaccard_similarity(snippet, context):
        list1 = np.array(CodexTokenizer().tokenize(SimilarityScore.removeImports(snippet)))
        list2 = np.array(CodexTokenizer().tokenize(SimilarityScore.removeImports(context)))
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return float(intersection) / union

    @staticmethod
    def sentence_cos_similarity(model, source, target):
        sentence1_vector = model.encode(source)
        sentence2_vector = model.encode(target)
        return np.dot(sentence1_vector, sentence2_vector) / (
                    np.linalg.norm(sentence1_vector) * np.linalg.norm(sentence2_vector))

    @staticmethod
    def removeImports(s):
        for i, line in enumerate(s.split('\n')):
            if not line.strip().startswith("import"):
                return '\n'.join(s.split('\n')[i:])
        return s

    def __init__(self, funcs):
        self.idfs = Tools.load_json('./resources/idfs.json')
        self.bm25 = fastbm25(funcs) if funcs else None
        self.fuzzy_map = {}



    def bm25_similarity(self, source, target):
        source = SimilarityScore.removeImports(source)
        target = SimilarityScore.removeImports(target)
        score = self.bm25.similarity_bm25(source.lower().split(), target.lower().split())
        return 1 / (1 + exp(-score))

    def calTfIdf(self, list1, list2, alpha = 0.5):
        # length_penalty = (max((len(list2) - len(list1)), len(list1)) / len(list2)) ** alpha
        length_penalty = 1
        tf1 = {}
        tf2 = {}
        list1 = Tools.getSplitWords(SimilarityScore.removeImports(list1))
        list2 = Tools.getSplitWords(SimilarityScore.removeImports(list2))
        for word in list1:
            if word not in self.idfs: # find the closest word for unknown words in test apps
                if word not in self.fuzzy_map:
                    self.fuzzy_map[word] = Tools.find_closest_word(word, self.idfs.keys())
                word = self.fuzzy_map[word]
            tf1[word] = tf1.get(word, 0) + 1
        for word in list2:
            if word not in self.idfs:
                if word not in self.fuzzy_map:
                    self.fuzzy_map[word] = Tools.find_closest_word(word, self.idfs.keys())
                word = self.fuzzy_map[word]
            tf2[word] = tf2.get(word, 0) + 1
        tfidf1 = {word: tf * self.idfs[word] for word, tf in tf1.items() if word in self.idfs}
        tfidf2 = {word: tf * self.idfs[word] for word, tf in tf2.items() if word in self.idfs}

        vocab_keys = set(tfidf1.keys()).intersection(set(tfidf2.keys()))
        diversity = 0.0
        for word in set(list1):
            x = tfidf1[word] if word in tfidf1 else tfidf1[self.fuzzy_map[word]]
            if word not in vocab_keys:
                diversity += x
            else:
                diversity -= x
        for word in set(list2):
            x = tfidf2[word] if word in tfidf2 else tfidf2[self.fuzzy_map[word]]
            if word not in vocab_keys:
                diversity += x
            else:
                diversity -= x
        diversity /= (sum(tfidf1.values()) + sum(tfidf2.values()))
        return (diversity + 1) / 2 * length_penalty

class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")

    def tokenize(self, text):
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def read_code(fpath):
    with open(fpath, 'r', encoding='utf8') as f:
        lines = f.readlines()
    inside_method = False
    method_lines = []
    comment_lines = []
    result = []
    recent_comment = []
    brace_cnt = 0
    pattern = r'(?:public|private|protected|internal|override)?\s*fun\s+' if fpath.endswith(".kt") \
        else r'^\s*(public|private|protected|static|override)?\s+\w+\s+\w+\s*\([^)]*\)\s*\{'

    for i, line in enumerate(lines):
        # Check if the line is the start of a method
        if re.match(pattern, line):
            inside_method = True
            brace_cnt = 1 if "{" in line else 0
            # Store recent comments if any
            method_lines = [line]
        elif inside_method:
            if len(line.strip()) > 0:
                method_lines.append(line)
            if "{" in line:
                brace_cnt += 1
            # Check if we reached the end of the method
            if "}" in line:
                brace_cnt -= 1
                if brace_cnt == 0:
                    inside_method = False
                    # Check if the method body is more than one line and not just a single return statement
                    if len(method_lines) > 3 and not method_lines[1].strip().startswith("return"):
                        recent_comment.extend(method_lines)
                        result.extend(recent_comment)
                        result.append("$$$###")
                    method_lines = []
                    recent_comment = []
        else:
            # Handle multi-line comments
            if line.strip().startswith("/*"):
                comment_lines = [line]
            elif comment_lines:
                comment_lines.append(line)
                if line.strip().endswith("*/"):
                    recent_comment = comment_lines
                    comment_lines = []

    return ''.join(result).split("$$$###")[:-1]

def getFunctionName(java_code):
    # tokens = list(javalang.tokenizer.tokenize(java_code))
    java_code = f"public class VirtualClass {{ {java_code} }}"
    tree = javalang.parse.parse(java_code)
    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration) or isinstance(node, javalang.tree.ConstructorDeclaration):
            return node.name
    # print(f"Not found function in {java_code}!")
    return ''

def extractImports(s):
    components = []
    flags = ["android.widget", "android.view", "androidx", "com.google.android.material"]
    for i, line in enumerate(s.split('\n')):
        if line.strip().startswith("import") and any(flag in line for flag in flags):
            components.append(line[line.rfind('.') + 1:])
    return components

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TransDataSet:
    def __init__(self, device, tokenizer, model, base_dir, max_length=1228, threshold=0.5):
        super(TransDataSet, self).__init__()
        self.device = device
        self.base_dir = base_dir if base_dir else os.path.join(os.getcwd()[:os.getcwd().rfind(os.sep)], 'repositories', 'android')
        self.tokenizer = tokenizer if tokenizer else CodexTokenizer()
        self.model = model
        self.sim_scorer = ['bm25', 'jac']
        self.max_length = max_length # set max length of the input = 1.2 * token length
        self.threshold = threshold
        self.funcs = Tools.load_json(os.path.join(os.getcwd(), 'datasets', f'all_methods.json'))
        self.actions = Tools.load_json(os.path.join(os.getcwd(), 'datasets', f'component_action_map.json'))
        self.resources = Tools.load_json("./resources/resources.json")
        self.resources_connections = Tools.load_json("./resources/resources_connections.json")

    def tokenize_and_encode(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encoder(**tokens).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy()

    def iterate_repository(self, repo):
        base_dir = self.base_dir
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

    def getFunctions(self):
        results = []
        for k, v in self.funcs.items():
            if k == "communication":
                continue
            results.extend(v)
        return results

    def getTopFunctions(self):
        base_dir = r"" # set top apps path
        self.base_dir = base_dir
        items = os.listdir(base_dir)
        repos = [item for item in items if os.path.isdir(os.path.join(base_dir, item))]
        results = []
        for repo in repos:
            results.extend(self.iterate_repository(repo))
        return results

    def getTypeFunctions(self, app_type):
        return self.funcs[app_type]

    def calAndroidSim(self, task, method):
        sim_score = 0
        # check android lifecycles
        if task['methodName'] == method['methodName']:
            sim_score += 1  # same lifecycle method
        elif task['methodName'].startswith('on') and method['methodName'].startswith('on'):
            sim_score += 0.5  # both lifecycle method
        elif task['methodName'].startswith('on') or method['methodName'].startswith('on'):
            pass  # only one is lifecycle method
        else:
            sim_score += 0.25  # neither is lifecycle method

        # check library components
        source = set(extractImports(task['prompt']))
        target = set(extractImports(method['context']))
        if len(source) == 0 and len(target) == 0:
            sim_score += 1
        elif len(source) == 0 or len(target) == 0:
            pass
        else:
            sim_score += len(source.intersection(target)) / len(source)  # as long as source is subset of target

        # check activity/provider/service/broadcast receiver/fragment
        keywords = ["activity", "provider", "service", "receiver", "fragment"]
        source = Tools.getSplitWords(task['fpath'][:task['fpath'].rfind(".")])[-1]
        target = Tools.getSplitWords(method['fpath'][:method['fpath'].rfind(".")])[-1]
        if source == target:
            sim_score += 1
        elif source in keywords and target in keywords:
            sim_score += 0.5
        elif source in keywords or target in keywords:
            pass
        else:
            sim_score += 0.25

        # check intent-filter actions
        flags = self.actions.keys()
        source = set()
        target = set()
        for f in flags:
            if f in task['fpath']:
                source = set(self.actions[f])
            elif f in method['fpath']:
                target = set(self.actions[f])
        if source and target:
            sim_score += len(source.intersection(target)) / len(source)

        return sim_score / 4

    def getSimilarFunction(self, calculater, task, functions, repo, sim_scorer, topK=10, num_return=1):
        target = []
        snippet = task['prompt']
        for function in functions:
            if task['app_type'] == "new" or function['repo'] != repo:
                if task['type'] == 'android':
                    sim_score = self.calAndroidSim(task, function)
                else:
                    sim_score = 0
                    if 'cos' in sim_scorer:
                        sim_score += SimilarityScore.cosine_similarity(function['embedding'], self.tokenize_and_encode(snippet))
                    elif 'jac' in sim_scorer:
                        sim_score += SimilarityScore.jaccard_similarity(snippet, function['context'])
                    elif 'bm25' in sim_scorer:
                        sim_score += calculater.bm25_similarity(snippet, function['context'])
                    elif len(sim_scorer) == 0:
                        sim_score = 1
                    else:
                        raise ValueError(f'Unknown similarity scorer: {sim_scorer}')
                target.append((function, sim_score))
                target.sort(key=lambda x: x[1], reverse=True)
                target = target[:topK]
                # target = target[:num_return]
        target = [item[0] for item in target]
        res = []
        for i, function in enumerate(target):
            diversity_tfidf = calculater.calTfIdf(snippet, function['context'])
            res.append((function, diversity_tfidf))
            res.sort(key=lambda x: x[1], reverse=False)
            res = res[:num_return]
        res = [item[0] for item in res]
        return res if num_return != 1 else res[0]
        # return target if num_return != 1 else target[0]

    # For each complementary case, find the similar function
    def dumpSimilarFunction(self, tasks, fname, app_type, topK=10):
        if app_type == "new":
            funcs = [func for funcs in [self.getTypeFunctions(atype)
                                        for atype in ["notes", "media", "life", "communication"]] for func in funcs]
        else:
            funcs = self.getTypeFunctions(app_type)
        if self.sim_scorer == 'bm25':
            calculater = SimilarityScore(funcs)
        else:
            calculater = SimilarityScore(None)
        for idx, task in enumerate(tasks):
            repo = task['fpath'].split(os.sep)[0]
            res = self.getSimilarFunction(calculater, task, funcs, repo, sim_scorer=self.sim_scorer, num_return=1, topK=topK)
            max_res = res
            assert isinstance(max_res, dict)
            # if task['past'] != max_res['context'] and max_res['context'] not in task['past']:
            #     print(f"Prompt: {task['prompt']}\nOutput: {task['choice']}\nPast: {task['past']}\n"
            #           f"Now: {max_res['context']}\nTruth: {task['ground_truth']}")
            task['similar_function_context'] = max_res['context']
            task['similar_function_repo'] = max_res['repo']
            task['similar_function_file'] = max_res['fpath']
            task['similar_function_name'] = max_res['methodName']
            if idx % 20 == 0:
                logger.info(f'idx: {idx}/{len(tasks)}')
            if idx % 100 == 0 and idx > 0:
                Tools.dump_json(tasks, fname)
        return tasks

    def checkJavaCall(self, node, target):
        if isinstance(node, list):
            for child in node:
                if self.checkJavaCall(child, target):
                    return True
        elif isinstance(node, javalang.tree.MethodInvocation):
            if node.member == target:
                return True
        elif isinstance(node, javalang.tree.VariableDeclarator):
            if node.initializer and self.checkJavaCall(node.initializer, target):
                return True
        return False


    def findJavaCall(self, repo_name, target):
        base_dir = "../repositories/android"
        files = glob.glob(os.path.join(f'{base_dir}/{repo_name}', "**", "*.java"), recursive=True)
        for file in files:
            with open(file, 'r', encoding='utf8') as f:
                lines = f.readlines()
                f.close()
            if len(lines) < 1:
                continue
            java_code = ''.join(lines)
            calling_functions_code = []
            tree = javalang.parse.parse(java_code)
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                if node.body is None:
                    continue
                for method_node in node.body:
                        if self.checkJavaCall(method_node.children, target):
                            calling_functions_code.append(''.join(lines[node.position[0] - 1 : node.body[-1].position[0] + 1]))
            if len(calling_functions_code) > 0:
                return calling_functions_code
        # print(f"Not found function {target} invocation in {repo_name}!")
        return []

    def runBasePkl(self, test_flag=False, topK=10, mode=""):
        if not test_flag:
            if os.path.exists(f'./datasets/pkls{mode}/base_train.pkl'):
                new_tasks = pickle.load(open(f"./datasets/pkls{mode}/base_train.pkl", 'rb'))
                tasks_types = defaultdict(list)
                for task in new_tasks:
                    repo = task['fpath'].split(os.sep)[0]
                    tasks_types[repo].append(task) # same app together in training
                    # tasks_types[task['app_type']].append(task) # same app type together in training
                tasks_types = sorted(tasks_types.items(), key=lambda x: len(x[1]), reverse=True)
                train_tasks, val_tasks, test_tasks = [], [], []
                train_radio, val_radio, test_radio = 0.8, 0.1, 0.1
                for app_type, t in tasks_types:
                    random.shuffle(t)
                    train_cnt = int(train_radio * len(t))
                    val_cnt = int(val_radio * len(t))
                    train_tasks.extend(t[:train_cnt])
                    val_tasks.extend(t[train_cnt : train_cnt + val_cnt])
                    test_tasks.extend(t[train_cnt + val_cnt:])
                pickle.dump(train_tasks, open("train.pkl", 'wb'))
                pickle.dump(val_tasks, open("valid.pkl", 'wb'))
                pickle.dump(test_tasks, open("test.pkl", 'wb'))
                return
            tasks = Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_notes_train.json'))
            tasks.extend(Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_media_train.json')))
            tasks.extend(Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_life_train.json')))
            tasks.extend(Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_communication_train.json')))
        else:
            if mode == "_new":
                tasks = Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_new_test.json'))
            else:
                tasks = Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_notes_test.json'))
                tasks.extend(Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_media_test.json')))
                tasks.extend(Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_life_test.json')))
                tasks.extend(Tools.load_json(os.path.join(os.getcwd(), 'datasets', 'dataset_communication_test.json')))
            tasks = [t for t in tasks if len(t['ground_truth']) < 128]
        fname = "train" if not test_flag else "test"
        # java_cnt = 0
        # call_cnt = 0
        # task_ids = []
        new_tasks = []
        pre_prompt = "# You are an excellent Android developer, now you need to complete a method in Android APP, the APP type is "
        funcs = {
            "notes": self.getTypeFunctions("notes"),
            "media": self.getTypeFunctions("media"),
            "life": self.getTypeFunctions("life"),
            "communication": self.getTypeFunctions("communication"),
            # "top": self.getTopFunctions()
        }
        all_funcs = []
        for i in funcs.values():
            all_funcs.extend(i)
        calculater = SimilarityScore([SimilarityScore.removeImports(func['context']).lower().split() for func in all_funcs])
        for idx, task in enumerate(tasks):
            repo = task['fpath'].split(os.sep)[0]
            # task['embedding_target'] = self.tokenizer.encode(task['ground_truth'], return_tensors="pt", max_length=2048, truncation=True)
            cur_funcs = funcs[task['app_type']] if task['app_type'] != 'new' else all_funcs
            res = self.getSimilarFunction(calculater, task, cur_funcs, repo, sim_scorer=self.sim_scorer, num_return=3, topK=topK)
            sim_funcs = ""
            for sim_func in res:
                assert isinstance(sim_func, dict)
                sim_funcs = '\n'.join(['# ' + line for line in sim_func['context'].split('\n')]) + '\n' + '-' * 50 + '\n' + sim_funcs
            max_res = res[0]
            assert isinstance(max_res, dict)
            task['prompt'] = (f"{pre_prompt}[{task['app_type']}], please use the information contained in this type of data as much as possible.\n"
                              f" # Similar methods for the fragment to be completed, you need to determine if it is useful:\n {sim_funcs} [SEP]"
                                  f"# Significantly, the APP type is [{task['app_type']}], please use the information contained in this type of data as much as possible\n"
                              f"# Determine if the information above is useful, Complete The Following Android Code:\n {task['prompt']}")
            # if task['fpath'].endswith('.kt') or max_res['fpath_tuple'][-1].endswith('.kt'):
            #     # task['embedding_source'] = self.tokenizer.encode(task['prompt'], return_tensors="pt", max_length=2048, truncation=True)
            #     pass
            # else:
            #     java_cnt += 1
            #     try:
            #         func_name = getFunctionName(max_res['context'])
            #         if func_name == '':
            #             # task['embedding_source'] = self.tokenizer.encode(task['prompt'], return_tensors="pt", max_length=2048, truncation=True)
            #             new_tasks.append(task)
            #             continue
            #     except javalang.parser.JavaSyntaxError:
            #         # task['embedding_source'] = self.tokenizer.encode(task['prompt'], return_tensors="pt", max_length=2048, truncation=True)
            #         new_tasks.append(task)
            #         continue
            #     invoke_func = self.findJavaCall(max_res['repo'], func_name)
            #     if len(invoke_func) > 0:
            #         task_ids.append(task['task_id'])
            #         invoke_func = invoke_func[0]
            #         index_completion = task['prompt'].find('# Determine if the information above is useful, Complete The Following Android Code:')
            #         task['prompt'] = task['prompt'][:index_completion] + f'# Called Method Of Similar Method:\n {invoke_func} [SEP]' + task['prompt'][index_completion:]
            #         # task['embedding_source'] = self.tokenizer.encode(task['prompt'], return_tensors="pt", max_length=2048, truncation=True)
            #         call_cnt += 1
            #         new_tasks.append(task)
            new_tasks.append(task)
            if idx % 100 == 0:
                logger.info(f'idx: {idx}/{len(tasks)}')
                pickle.dump(new_tasks, open(f"./datasets/pkls{mode}/base_{fname}.pkl", 'wb'))
        # print(f'Java cnt: {java_cnt} Call cnt: {call_cnt}')
        # print(task_ids)
        pickle.dump(new_tasks, open(f"./datasets/pkls{mode}/base_{fname}.pkl", 'wb'))
        assert os.path.exists(f'./datasets/pkls{mode}/base_{fname}.pkl')
        if not test_flag:
            self.runBasePkl(mode=mode, topK=topK)
        else:
            if os.path.exists(f'./datasets/pkls{mode}/base_test_copy.pkl'):
                os.remove(f'./datasets/pkls{mode}/base_test_copy.pkl')
                logger.warning("Remove old base_test_copy.pkl")
            shutil.copyfile(f'./datasets/pkls{mode}/base_{fname}.pkl', f'./datasets/pkls{mode}/base_{fname}_copy.pkl')

    def getMatchTags(self, sim_scorer, sim_model, documents, methodName):
        target = []
        method_name = ' '.join(Tools.getSplitWords(methodName))
        for doc, tags in documents.items():
            sim_score = sim_scorer.bm25_similarity(doc, method_name)
            target.append((doc, tags, sim_score))
            target.sort(key=lambda x: x[2], reverse=True)
            target = target[:10]
        max_tar = []
        for one in target:
            sim_score = SimilarityScore.sentence_cos_similarity(sim_model, method_name, one[0])
            if len(max_tar) == 0 or max_tar[1] < sim_score:
                max_tar = [one[1], sim_score]
        prompt = ""
        if max_tar[1] >= self.threshold:
            prompt = "Probably, these configuration items appear in the Android code to be completed: "
            prompt += max_tar[0] + '\n'
        return prompt

    def dumpMatchTags(self):
        tasks = []
        for i in ["notes", "media", "life", "communication"]:
            for j in ["train", "test"]:
                tasks.extend(Tools.load_json(f'./datasets/dataset_{i}_{j}.json'))
        upperTags = Tools.load_json('./resources/upper_tags.json')
        sim_model = SentenceTransformer('roberta-base-nli-mean-tokens')
        upper_tags_data = {}
        bm25_scorer = SimilarityScore(upperTags.keys())
        logger.info("start dump upper tags data")
        for i, task in enumerate(tasks):
            upper_tags_data[task['task_id']] = self.getMatchTags(bm25_scorer, sim_model, upperTags, task['methodName'])
            if (i + 1) % 10 == 0:
                logger.info(f'Finished {i + 1}/{len(tasks)} tags')
                Tools.dump_json(upper_tags_data, './resources/upper_tags_data.json')
        Tools.dump_json(upper_tags_data, './resources/upper_tags_data.json')
        logger.success("finish dump upper tags data")

    def getMatchVariables(self, sim_model, variables, methodName, fileName, topK=3):
        lines = variables.split('\n')
        if len(lines) <= topK:
            return variables
        target = []
        method_name = ' '.join(Tools.getSplitWords(methodName))
        file_name = ' '.join(Tools.getSplitWords(fileName))
        for i, line in enumerate(lines):
            sline = line.strip()
            if line.startswith(("*", "//")) or "**" in line:
                continue
            words = sline.split()
            if len(words) <= 1:
                continue
            if "@override" in line or "{" in line:
                break
            if "=" in sline and "=" not in words:
                break
            if "=" in sline:
                index = words.index("=") - 1  # for assignment
            elif sline.endswith(';'):
                index = -1  # for java
            else:
                index = 1  # for kotlin
            source = ' '.join(Tools.getSplitWords(words[index]))
            sim_score = max(SimilarityScore.jaccard_similarity(method_name, source),
                            SimilarityScore.jaccard_similarity(file_name, source))
            target.append((line, sim_score, source))
            target.sort(key=lambda x: x[1], reverse=True)
        ans = []
        for item in target:
            sim_score = SimilarityScore.sentence_cos_similarity(sim_model, method_name, item[2])
            if sim_score < self.threshold:
                sim_score = max(sim_score, SimilarityScore.sentence_cos_similarity(sim_model, file_name, item[2]))
            ans.append((item[0], sim_score))
            ans.sort(key=lambda x: x[1], reverse=True)
            ans = ans[:topK]
        res = '\n'.join([item[0] for item in ans])
        return res

    def getMatchResources(self, repo, methodName, fileName):
        method_name = ' '.join(Tools.getSplitWords(methodName))
        if method_name not in self.resources_connections or repo not in self.resources:
            return ""
        dirs = self.resources_connections[method_name]
        ans = []
        resources = self.resources[repo]
        for one in dirs:
            for r in resources:
                source = set(r[r.find(os.sep) + 1:].split('_'))
                target = Tools.getSplitWords(methodName)
                target.extend(Tools.getSplitWords(fileName))
                if one in r and len(set(target) & source) > 2:
                    ans.append(r.replace(os.sep, '.'))
        if len(ans) == 0:
            return ""
        prompt = "Probably, these resources are needed in the code fragment to be completed: \n" + ' '.join(ans) + '\n'
        return prompt

    def getMatchMethodNames(self, sim_model, app_type, fpath, methodName, topK=3):
        if app_type == "new":
            return ""
        method_name = ' '.join(Tools.getSplitWords(methodName))
        targets = [func['methodName'] for func in self.getTypeFunctions(app_type) if func['fpath'] == fpath]
        ans = []
        for target in targets:
            sim_score = SimilarityScore.sentence_cos_similarity(sim_model, method_name, ' '.join(Tools.getSplitWords(target)))
            ans.append((target, sim_score))
            ans.sort(key=lambda x: x[1], reverse=True)
            ans = ans[:topK]
        ans = [one[0] for one in ans if one[1] >= self.threshold]
        if len(ans) == 0:
            return ""
        prompt = "Probably, these methods are needed in the code fragment to be completed: \n" + ' '.join(ans) + '\n'
        return prompt

    def runWithCall(self, test_flag=False, topK=10, mode=''):
        # load all data
        if not test_flag and not os.path.exists(f'./resources/call_tasks_train{mode}.json'):
            logger.error('Please run CallGraph first!')
            return
        fname = "train" if not test_flag else "test_copy"
        if not os.path.exists(f'./datasets/pkls{mode}/base_{fname}.pkl'):
            self.runBasePkl(test_flag, mode=mode, topK=topK)
        if not os.path.exists('./resources/upper_tags_data.json'):
            logger.error("Please run dumpMatchTags first!")
            return
        upper_tags_data = Tools.load_json('./resources/upper_tags_data.json')
        if test_flag:
            call_tasks = Tools.load_json(f'./resources/call_tasks_test{mode}.json')
            targets = [f'./datasets/pkls{mode}/base_test']
            if not os.path.exists(f"{targets[0]}.pkl"):
                shutil.copyfile(f'{targets[0]}_copy.pkl', f'{targets[0]}.pkl')
                logger.warning("Use base_test_copy.pkl!")
        else:
            call_tasks = Tools.load_json(f'./resources/call_tasks_train{mode}.json')
            targets = ['train', 'valid', 'test']
            if not os.path.exists("train.pkl"):
                self.runBasePkl(test_flag, topK=topK, mode=mode)
        sim_model = SentenceTransformer('roberta-base-nli-mean-tokens')
        for file in targets:
            tasks = pickle.load(open(f"{file}.pkl", 'rb'))
            if test_flag:
                tasks = [t for t in tasks if len(t['ground_truth']) < 128]
            for idx, task in enumerate(tasks):
                if (idx + 1) % 20 == 0:
                    logger.info(f'idx: {idx + 1}/{len(tasks)}')
                # ensure the length of the prompt is less than max_length
                if len(task['prompt']) > self.max_length:
                    completion_index = task['prompt'].find('# Called Example Of Similar Method:')
                    if completion_index == -1:
                        completion_index = task['prompt'].find('# Significantly, the APP type is ')
                    sim_index = task['prompt'].find('# Similar methods for the fragment to be completed, you need to determine if it is useful')
                    sim_content = task['prompt'][sim_index:completion_index].split('-' * 50 + '\n')
                    if len(sim_content) < 4:
                        continue
                    if task['task_id'] in call_tasks:
                        cur_method = call_tasks[task['task_id']][1]
                        sim_content[2] = '\n'.join([('# ' + line) for line in cur_method.split('\n')])
                    sim_content = sim_content[1] + '\n' + '-' * 50 + '\n' + sim_content[2]
                    task['prompt'] = (task['prompt'][:sim_index]
                                      + '# Similar methods for the fragment to be completed, you need to determine if it is useful\n'
                                      + sim_content + '[SEP]' + task['prompt'][completion_index:])

                # add caller method
                if task['task_id'] in call_tasks:
                    if call_tasks[task['task_id']][0] == '':
                        continue
                    completion_index = task['prompt'].find('# Similar methods for the fragment to be completed, you need to determine if it is useful:')
                    if completion_index == -1:
                        completion_index = task['prompt'].find('# Significantly, the APP type is ')
                    assert completion_index != -1
                    call_len = self.max_length - len(task['prompt']) - len("# Called Method Of Similar Method:\n ")
                    if call_len < 0:
                        continue
                    elif call_len < len(call_tasks[task['task_id']][0]):
                        call_tasks[task['task_id']][0] = call_tasks[task['task_id']][0][:call_len]
                        call_tasks[task['task_id']][0] = call_tasks[task['task_id']][0][:call_tasks[task['task_id']][0].rfind('-' * 50 + '\n')]
                    task['prompt'] = (task['prompt'][:completion_index]
                                      + f'# Called Method Of Similar Method:\n {call_tasks[task["task_id"]][0]} [SEP]'
                                      + task['prompt'][completion_index:])

                file_name = task['fpath'][task['fpath'].rfind(os.sep) + 1: task['fpath'].rfind(".")]
                # add global variables
                if len(task['prompt']) < self.max_length:
                    if task['variables'] != '':
                        completion_index = task['prompt'].find('# Determine if the information above is useful, Complete The Following Android Code:')
                        code = task['prompt'][completion_index:].split('\n')
                        prefix = code.pop(0)
                        add_index = 0
                        for i, line in enumerate(code):
                            if not line.strip().startswith('import'):
                                add_index = i
                                break
                        code = (prefix + '\n'.join(code[:add_index]) + '\n'
                                + self.getMatchVariables(sim_model, task['variables'], task['methodName'], file_name)
                                + '\n' + '\n'.join(code[add_index:]))
                        task['prompt'] = (task['prompt'][:completion_index] + code)

                # add upper tags
                if len(task['prompt']) < self.max_length:
                    if task['task_id'] in upper_tags_data and upper_tags_data[task['task_id']] != '':
                        completion_index = task['prompt'].find('# Significantly, the APP type is ')
                        task['prompt'] = (task['prompt'][:completion_index]
                                          + upper_tags_data[task['task_id']]
                                          + task['prompt'][completion_index:])

                # add relevant methods and resources
                if len(task['prompt']) < self.max_length and task['type'] == 'android':
                    completion_index = task['prompt'].find('# Significantly, the APP type is ')
                    cur_repo = task['task_id'][:task['task_id'].find('/')]
                    task['prompt'] = (task['prompt'][:completion_index]
                                      + self.getMatchResources(cur_repo, task['methodName'], file_name)
                                      + self.getMatchMethodNames(sim_model, task['app_type'], task['fpath'], task['methodName'])
                                      + task['prompt'][completion_index:])
            pickle.dump(tasks, open(f"./{file}.pkl", 'wb'))

def getDataset(device, tokenizer, model, base_dir, max_length=1024):
    if base_dir:
        wrong_tasks = Tools.load_json(base_dir)
        tasks = pickle.load(open("train.pkl", 'rb'))
        tasks.extend(pickle.load(open("valid.pkl", 'rb')))
        wrong_tasks = {wrong_task['task_id']: wrong_task['prompt'] for wrong_task in wrong_tasks}
        results = []
        for t in tasks:
            if t['task_id'] in wrong_tasks:
                t['prompt'] = wrong_tasks[t['task_id']]
                results.append(t)
        random.shuffle(results)
        train_cnt = int(len(results) * 0.8)
        valid_cnt = int(len(results) * 0.1)
        pickle.dump(results[:train_cnt], open("./train_wrong/train.pkl", 'wb'))
        pickle.dump(results[train_cnt:train_cnt + valid_cnt], open("./train_wrong/valid.pkl", 'wb'))
        pickle.dump(results[train_cnt + valid_cnt:], open("./train_wrong/test.pkl", 'wb'))
        return (CustomDataset(pickle.load(open("./train_wrong/train.pkl", 'rb'))), CustomDataset(pickle.load(open("./train_wrong/valid.pkl", 'rb'))),
                CustomDataset(pickle.load(open("./train_wrong/test.pkl", 'rb'))))
    if not os.path.exists('train.pkl'):
        TransDataSet(device, tokenizer, model, base_dir, max_length).runBasePkl()
    return (CustomDataset(pickle.load(open("train.pkl", 'rb'))), CustomDataset(pickle.load(open("valid.pkl", 'rb'))),
            CustomDataset(pickle.load(open("test.pkl", 'rb'))))

def checkBadResults():
    base_dir = "APPs Repos Path"
    bads = Tools.load_jsonl("../predictions/bad-results.jsonl")
    bad_ids = {}
    xstr = "# Similar methods for the fragment to be completed, you need to determine if it is useful:\n"
    for b in bads:
        bad_ids[b['metadata']['task_id']] = [b['prompt'][b['prompt'].index(xstr) + len(xstr):b['prompt'].index('[SEP]')], b['choices'][0]['text']]
    types = ["notes", "media", "life"]
    flags = ["train", "test"]
    for flag in flags:
        fname = os.path.join(os.getcwd(), 'datasets', f'dataset_sim_{flag}.json')
        tasks = []
        for t in types:
            tmp = Tools.load_json(os.path.join(os.getcwd(), 'datasets', f'dataset_{t}_{flag}.json'))
            origins = []
            for tx in tmp:
                if tx['task_id'] in bad_ids.keys():
                    tx['past'] = bad_ids[tx['task_id']][0]
                    tx['choice'] = bad_ids[tx['task_id']][1]
                    origins.append(tx)
            tasks.extend(TransDataSet(None, None, None, f"{base_dir}/{t}/train").dumpSimilarFunction(origins, fname, t))

def prepareBaseSim(topK=10):
    base_dir = "APPs Repos Path"
    types = ["communication", "notes", "media", "life"]
    flags = ["train", "test"]
    # types = ["new"]
    # flags = ["test"]
    for flag in flags:
        fname = os.path.join(os.getcwd(), 'datasets', f'dataset_sim_{flag}_topk{topK}.json')
        tasks = []
        for t in types:
            origins = Tools.load_json(os.path.join(os.getcwd(), 'datasets', f'dataset_{t}_{flag}.json'))
            if flag == "test":
                origins = [t for t in origins if len(t['ground_truth']) < 128]
            tasks.extend(TransDataSet(None, None, None, f"{base_dir}/{t}/train")
                         .dumpSimilarFunction(origins, fname, t, topK=topK))
        Tools.dump_json(tasks, fname)

def checkSimTasks():
    fname = os.path.join(os.getcwd(), 'datasets', f'dataset_sim_train.json')
    stats = {}
    for task in Tools.load_json(fname):
        if 'similar_function_file' not in task:
            break
        if task['similar_function_file'] in stats:
            stats[task['similar_function_file']] += 1
        else:
            stats[task['similar_function_file']] = 1
    stats = dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))
    for k, v in stats.items():
        print(k, v)

if __name__ == "__main__":
    # checkSimTasks()
    parser = argparse.ArgumentParser(description='Prepare Dataset')
    parser.add_argument("--sim", action="store_true", help="Dump Similar Samples For MetaEnhancement")
    parser.add_argument("--integrate", action="store_true", help="Integrate Data For Training")
    args = parser.parse_args()
    if args.sim:
        prepareBaseSim(topK=10)
    if args.integrate:
        TransDataSet(None, None, None, None).runWithCall(test_flag=False, mode="")
    # TransDataSet(None, None, None, None).runWithCall(test_flag=True, mode="_new")
    # TransDataSet(None, None, None, None).dumpMatchTags()