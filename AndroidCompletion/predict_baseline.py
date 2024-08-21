import argparse
import os
import pickle
import random
import torch
from utils import Tools
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, BitsAndBytesConfig
from openai import OpenAI
from loguru import logger

client = OpenAI(
    api_key = "",
    base_url=""
)

def buildSamples():
    tasks = Tools.load_jsonl('../prompts/rg-one-gram-ws-20-ss-2.jsonl')
    random.shuffle(tasks)
    tasks = tasks[:160]
    # task_ids = [t['metadata']['task_id'] for t in tasks]
    Tools.dump_jsonl(tasks, '../prompts/rg-2k-one-gram-ws-20-ss-2-sample.jsonl')
    # tasks = Tools.load_jsonl('../prompts/rg-1k-one-gram-ws-20-ss-2.jsonl')
    # tasks_dict = {t['metadata']['task_id']: t for t in tasks}
    # tasks = [tasks_dict[tid] for tid in task_ids]
    # Tools.dump_jsonl(tasks, '../prompts/rg-1k-one-gram-ws-20-ss-2-sample.jsonl')

def updateSamples():
    tasks = Tools.load_jsonl('../prompts/rg-2k-one-gram-ws-20-ss-2-sample.jsonl')
    task_ids = [t['metadata']['task_id'] for t in tasks]
    tasks = Tools.load_jsonl('../prompts/random-rg-one-gram-ws-20-ss-2.jsonl')
    tasks_dict = {t['metadata']['task_id']: t for t in tasks}
    tasks = [tasks_dict[tid] for tid in task_ids]
    Tools.dump_jsonl(tasks, '../prompts/random-rg-2k-one-gram-ws-20-ss-2-sample.jsonl')

def getStats():
    tasks = Tools.load_jsonl('../prompts/rg-2k-one-gram-ws-20-ss-2-sample.jsonl')
    stats = {}
    for t in tasks:
        repo = t['metadata']['task_id'].split('/')[0]
        if repo not in stats:
            stats[repo] = 1
        else:
            stats[repo] += 1
    logger.info(stats)
    tasks = Tools.load_jsonl('../prompts/rg-one-gram-ws-20-ss-2.jsonl')
    line_cnt = 0
    for t in tasks:
        line_cnt += len(t['metadata']['prompt'].splitlines() + t['metadata']['ground_truth'].splitlines())
    logger.info(line_cnt / len(tasks))


def predict(model='gpt', device='cuda:0', flag=False):
    if flag:
        fname = f"../predictions/{model}-prompt-unknown.jsonl"
    else:
        fname = f"../predictions/{model}-prompt.jsonl"
    if model in ['codeT', 'starcoder', 'starcoderbase', 'codellama', 'codegpt']:
        if flag:
            tasks = pickle.load(open('./datasets/pkls_new/base_test.pkl', 'rb'))
        else:
            tasks = pickle.load(open('./datasets/pkls/base_test.pkl', 'rb'))
        try:
            results = Tools.load_jsonl(fname)
        except FileNotFoundError:
            results = []
        ids = [result['metadata']['task_id'] for result in results]
        tasks = [task for task in tasks if task['task_id'] not in ids]
        ids = [task['task_id'] for task in tasks]
        results = [result for result in results if result['metadata']['task_id'] not in ids]
        tokenizer, real_model = None, None
        if model == 'codeT':
            checkpoint = "Salesforce/codet5p-770m"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            real_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        # elif model == 'uniXCoder':
        #     real_model = UniXcoder("microsoft/unixcoder-base")
        elif 'starcoder' in model:
            checkpoint = "bigcode/starcoder2-3b" if "base" not in model else "bigcode/starcoderbase-1b"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            real_model = AutoModelForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=True, quantization_config=quantization_config)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            real_model.config.pad_token_id = tokenizer.eos_token_id
            real_model.config.eos_token_id = tokenizer.eos_token_id
        elif model == 'codellama':
            checkpoint = "codellama/CodeLlama-7b-hf"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            real_model = AutoModelForCausalLM.from_pretrained(checkpoint, low_cpu_mem_usage=True, quantization_config=quantization_config)
        elif model == 'codegpt':
            checkpoint = "microsoft/CodeGPT-small-java-adaptedGPT2"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            real_model = AutoModelForCausalLM.from_pretrained(checkpoint)
        else:
            checkpoint = "Salesforce/codegen-350M-multi"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            real_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16)
        logger.success('Model loaded')
        for i, t in enumerate(tasks):

            t['prompt'] = t['prompt'].replace('[SEP]', '\n')
            target = "# Determine if the information above is useful, Complete The Following Android Code:"
            t['prompt'] = t['prompt'][t['prompt'].find(target) + len(target):].strip()
            if "starcoder" in model:
                response = getStarCoder(t['prompt'], tokenizer, real_model, device)
            elif model == "codellama":
                response = getCodeLLaMa(t['prompt'], tokenizer, real_model, device)
            elif model == "codegpt":
                response = getCodeGPT(t['prompt'], tokenizer, real_model.to(device), device)
            else:
                response = getCodeT(t['prompt'], tokenizer, real_model.to(device), device)
            results.append({
                'prompt': t['prompt'],
                'choices': [{ "text": response }],
                'metadata': {
                    'task_id': t['task_id'],
                    'ground_truth': t['ground_truth'],
                    'fpath': t['fpath']
                } if 'metadata' not in t else t['metadata']
            })
            if (i + 1) % 10 == 0:
                logger.info(f'Finished {i + 1}/{len(tasks)} prompts')
                Tools.dump_jsonl(results, fname)
                # print(t['ground_truth'] if 'metadata' not in t else t['metadata']['ground_truth'])
                # print(response)
                # print('---')
    else:
        raise Exception('Invalid model')
    Tools.dump_jsonl(results, fname)
    removeDeprecated(fname, flag)

def getCodeGPT(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    generated_ids = model.generate(inputs.input_ids.to(device), max_new_tokens=128)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return output[output.find(prompt) + len(prompt):] if output.startswith(prompt) else output

def getStarCoder(prompt, tokenizer, model, device):
    inputs = tokenizer.encode(prompt, max_length=1024, return_tensors="pt", truncation=True, return_attention_mask=False).to(device)
    attention_mask = torch.ones_like(inputs)
    outputs = model.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id, num_beams=5, max_new_tokens=128)
    output = tokenizer.decode(outputs[0])
    output = output[:output.find('```')] if '```' in output else output
    return output

def getCodeLLaMa(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, max_length=1024, return_tensors="pt", truncation=True).to(device)
    output = model.generate(inputs["input_ids"], max_new_tokens=128, repetition_penalty=0.9, num_beams=5,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(output[0].to("cpu"))
    output = output[output.find('<s>') + len('<s>'):].strip()
    output = output[output.find(prompt) + len(prompt):] if output.startswith(prompt) else output
    return output.strip()

def getCodeT(prompt, tokenizer, model, device):
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(inputs, max_new_tokens=128, num_beams=5, num_return_sequences=1)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs

def getChatGPT(prompt):
    completion = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=[
        {"role": "system", "content": 'You are a helpful code assistant.Your language of preferences are Java and Kotlin.'},
        {"role": "user", "content": prompt}
    ])
    return completion.choices[0].message.content

def formatGPTResults(path='../predictions/gpt-prompt.jsonl'):
    res = Tools.load_jsonl(path)
    for r in res:
        if "```java" in r['choices'][0]['text']:
            r['choices'][0]['text'] = r['choices'][0]['text'][r['choices'][0]['text'].find("```java") + 7: r['choices'][0]['text'].rfind("```")]
        elif "```kotlin" in r['choices'][0]['text']:
            r['choices'][0]['text'] = r['choices'][0]['text'][r['choices'][0]['text'].find("```kotlin") + 9: r['choices'][0]['text'].rfind("```")]
        elif "```" in r['choices'][0]['text']:
            r['choices'][0]['text'] = r['choices'][0]['text'][r['choices'][0]['text'].find("```") + 3: r['choices'][0]['text'].rfind("```")]
        elif "llama" in path and "Sure" in r['choices'][0]['text']:
            r['choices'][0]['text'] = r['choices'][0]['text'][:r['choices'][0]['text'].find("\n\nIn this code,")].strip()
        elif r['choices'][0]['text'].startswith("\n"):
            r['choices'][0]['text'] = r['choices'][0]['text'][1:]
    Tools.dump_jsonl(res, path)

def removeDeprecated(fname, flag):
    if flag:
        targets = Tools.load_jsonl('../predictions/codet5p-770m-tfidf-type-all-unknown.jsonl')
    else:
        targets = Tools.load_jsonl('../predictions/codet5p-220m-tfidf-type-all-one.jsonl')
    targets = [t['metadata']['task_id'] for t in targets]
    tasks = Tools.load_jsonl(fname)
    tasks = [t for t in tasks if t['metadata']['task_id'] in targets]
    Tools.dump_jsonl(tasks, fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict By Prompt')
    parser.add_argument('--model', type=str, help='Model Select', default="codeT")
    parser.add_argument('--cuda', type=str, help='Cuda Set', default="0")
    parser.add_argument("--new", action="store_true", help="Use Unknown APP Type Predictions")
    args = parser.parse_args()
    predict(args.model, f"cuda:{args.cuda}", args.new)