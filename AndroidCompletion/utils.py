# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import glob
import pickle
import json
import re
from fuzzywuzzy import fuzz
import tiktoken
from transformers import AutoTokenizer


class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")
    
    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

class CodeGenTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-6B-mono')

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

class Tools:
    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return f.read()
    
    @staticmethod
    def load_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def dump_json(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(obj, f)

    @staticmethod
    def dump_jsonl(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')
    
    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines

    @staticmethod
    def load_json(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return json.load(f)
    
    @staticmethod
    def iterate_repository(repo):
        base_dir = '../repositories/android'
        pattern = os.path.join(base_dir, repo, "**", "*")
        files = glob.glob(f"{pattern}.kt", recursive=True) + glob.glob(f"{pattern}.java", recursive=True)

        skipped_files = []
        loaded_code_files = dict()
        base_dir_list = os.path.normpath(base_dir).split(os.sep)
        for fname in files:
            try:
                code = Tools.read_code(fname)
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
                loaded_code_files[fpath_tuple]= code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")
        return loaded_code_files

    @staticmethod
    def tokenize(code):
        tokenizer = CodexTokenizer()
        return tokenizer.tokenize(code)

    @staticmethod
    def getSplitWords(s):
        words = re.findall(r'[A-Za-z][a-z]*|[A-Z][a-z]*', s)
        single = ''.join([word.lower() for word in words if len(word) == 1])
        words = [word.lower() for word in words if len(word) > 1]
        if len(single) > 1:
            words.append(single)
        return words

    @staticmethod
    def find_closest_word(query, word_list):
        closest_word = max(word_list, key=lambda word: fuzz.ratio(query, word))
        return closest_word

    @staticmethod
    def removeImportsAndSignature(s):
        s = '\n'.join([line for line in s.split('\n') if line.strip() != ''])
        for i, line in enumerate(s.split('\n')):
            if not line.strip().startswith("import") and i == 0:
                return s
            if not line.strip().startswith("import"):
                return '\n'.join(s.split('\n')[i + 1:]) if i < len(s.split('\n')) - 1 else '\n'.join(s.split('\n')[i:])
        return s

    @staticmethod
    def removeDuplicate(s, prompt):
        max_overlap = 0
        for i in range(1, min(len(s), len(prompt)) + 1):
            if prompt[-i:] == s[:i]:
                max_overlap = i
        return s[max_overlap:]

    @staticmethod
    def removeComments(code):
        lines = code.splitlines()
        lines = [line for line in lines if not line.strip().startswith("//")]
        return '\n'.join(lines)

    @staticmethod
    def removeNewMethods(code):
        targets = ["public", "private", "protected", "fun", "static", "class"]
        if len(code.splitlines()) > 2:
            x = '\n'.join(code.splitlines()[2:])
            for i in targets:
                if i in x:
                    return code[:code.rfind(i)]
        return code

    @staticmethod
    def truncateInput(code):
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 0:
                return '\n'.join(lines[:i])
        return '\n'.join(lines)
