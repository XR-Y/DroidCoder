import argparse
import pickle
from peft import PeftModel
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from utils import Tools
from loguru import logger

max_input_len = 1024
def generateOne(prompt, model):
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) > max_input_len:
        truncated_tokens = tokens[-max_input_len:]
    else:
        truncated_tokens = tokens
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
    inputs = tokenizer(truncated_text.replace('[SEP]', '\n'), return_tensors="pt", max_length=max_input_len, truncation=True)
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), max_new_tokens=args.token, num_beams=args.nb, num_return_sequences=1)
    return tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

def generateOneForGPT(prompt, model):
    prompt = prompt + '[TO_COMPLETE]'
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) > max_input_len:
        truncated_tokens = tokens[-max_input_len:]
    else:
        truncated_tokens = tokens
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens).strip()
    inputs = tokenizer(truncated_text, return_tensors="pt", max_length=max_input_len, truncation=True)
    generated_ids = model.generate(inputs["input_ids"].to(device), max_new_tokens=args.token, num_beams=args.nb,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    output = generated_ids[0]
    context_start = None
    for i, token_id in enumerate(output):
        if token_id == tokenizer.additional_special_tokens_ids[1]:
                context_start = i + 1
                break
    if context_start is not None:
        output = output[context_start:]
    output = tokenizer.decode(output, skip_special_tokens=True).strip()
    target = 'Complete The Following Android Code:'
    if truncated_text in output:
        output = output[output.find(truncated_text) + len(truncated_text):].strip()
    elif target in output:
        output = output[output.find(target) + len(target):].strip()
    target = '-' * 50
    return output if target not in output else output[:output.find(target)]

def generateAll():
    if args.all:
        if args.codegpt:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'additional_special_tokens': [code_sep_token, '[TO_COMPLETE]', '[SEP]']})
            model = AutoModelForCausalLM.from_pretrained(f"output_all{args.mode}")
            model.resize_token_embeddings(len(tokenizer))
        elif args.peft:
            tokenizer.pad_token = tokenizer.eos_token
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(checkpoint, use_cache=True,
                                                      quantization_config=quantization_config, low_cpu_mem_usage=True)
            tokenizer.add_special_tokens({'additional_special_tokens': [code_sep_token, '[TO_COMPLETE]', '[SEP]']})
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(model, "output_all_starcoder")
        else:
            model = T5ForConditionalGeneration.from_pretrained(f"output_all{args.mode}")
        model = model.to(device)
        model.decoder_only = True
        model.eval()
    else:
        model = T5ForConditionalGeneration.from_pretrained("output_java")
        # model = PeftModel.from_pretrained(model, "output_java")
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        model.decoder_only = True
        model.eval()

        model_android = T5ForConditionalGeneration.from_pretrained("output_android")
        # model_android = PeftModel.from_pretrained(model_android, "output_android")
        model_android = model_android.to(device)
        model_android.decoder_only = True
        model_android.eval()
        model_android.resize_token_embeddings(len(tokenizer))
        # model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float16, load_in_8bit=True)
        # model = PeftModel.from_pretrained(model, "output_all")
        # model = model.to(device)
        # model.eval()
    if args.new:
        mode = "_new"
    elif args.mode in ["_loss", "_best", "_770", "_220"] or args.codegpt or args.peft:
        mode = ""
    else:
        mode = args.mode
    tasks = pickle.load(open(f"./datasets/pkls{mode}/base_test.pkl", "rb"))
    # tasks = pickle.load(open("train.pkl", "rb"))
    # tasks.extend(pickle.load(open("valid.pkl", "rb")))
    # tasks.extend(pickle.load(open("test.pkl", "rb")))
    if args.reverse:
        tasks = tasks[::-1]
    try:
        results = Tools.load_jsonl(fname)
    except FileNotFoundError:
        results = []
    # results = []
    for i, task in enumerate(tasks):
        if i < len(results):
            continue
        if args.all:
            result = generateOne(task['prompt'], model) if not args.codegpt and not args.peft\
                else generateOneForGPT(task['prompt'], model)
        else:
            if task['type'] == 'android':
                result = generateOne(task['prompt'], model_android)
            else:
                result = generateOne(task['prompt'], model)
        results.append({
            'prompt': task['prompt'],
            'choices': [{"text": result}],
            'metadata': {
                'task_id': task['task_id'],
                'ground_truth': task['ground_truth'],
                'fpath': task['fpath']
            } if 'metadata' not in task else task['metadata']
        })
        if (i + 1) % 10 == 0:
            logger.info(f'Finished {i + 1}/{len(tasks)} prompts')
            Tools.dump_jsonl(results, fname)
    Tools.dump_jsonl(results, fname)
    formatResults(fname)
    logger.success("finish generate all")

def formatResults(fname):
    tasks = Tools.load_jsonl(fname)
    target = "# Determine if the information above is useful, Complete The Following Android Code:"
    ans = []
    for task in tasks:
        prompt = task['prompt']
        if target in prompt:
            prompt = prompt[prompt.find(target) + len(target):].strip()
            if len(prompt) == 0:
                continue
        if 'deprecate' in str(task['prompt'] + task['metadata']['ground_truth']).lower():
            continue
        ans.append(task)
    Tools.dump_jsonl(ans, fname)


def reGenerateWrong():
    model = T5ForConditionalGeneration.from_pretrained("output_wrong")
    # model = PeftModel.from_pretrained(model, "output_wrong")
    model = model.to(device)
    model.eval()
    wrong_tasks = Tools.load_json('wrong_tasks.json')
    results = Tools.load_jsonl(fname)
    tasks_map = {task['metadata']['task_id']: task for task in results}
    for i, task in enumerate(wrong_tasks):
        res = generateOne(task['prompt'], model)
        tasks_map[task['task_id']]['choices'][0]['text'] = res
        if (i + 1) % 10 == 0:
            logger.info(f'Finished {i + 1}/{len(wrong_tasks)} prompts')
            Tools.dump_jsonl(tasks_map.values(), fname.replace("-tfidf", "-tfidf-once"))
    Tools.dump_jsonl(tasks_map.values(), fname.replace("-tfidf", "-tfidf-once"))
    logger.success("finish regenerate wrong")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint', default="220m")
    parser.add_argument('--fname', type=str, default='test', help='File name to save predictions')
    parser.add_argument('--log', type=str, default='generate.log', help='File name to save logs')
    parser.add_argument('--wrong', action='store_true', help='Regenerate wrong predictions')
    parser.add_argument('--all', action='store_true', help='Use One Model')
    parser.add_argument('--reverse', action='store_true', help='Reverse Tasks Order')
    parser.add_argument('--new', action='store_true', help='Generate Unknown APP Type Predictions')
    parser.add_argument('--codegpt', action='store_true', help='Generate CodeGPT Model Predictions')
    parser.add_argument('--peft', action='store_true', help='Finetune Model Use PEFT')
    parser.add_argument('--cuda', type=str, help='Cuda Set', default="0")
    parser.add_argument('--mode', type=str, help='Set _best or _loss or _noandroid or _notfidf or _nometadata', default="")
    parser.add_argument('--nb', type=int, default=5, help='Num_Beams Set')
    parser.add_argument('--token', type=int, default=128, help='max_new_tokens Set')
    args = parser.parse_args()
    checkpoint = f'Salesforce/codet5p-{args.checkpoint}' if not args.codegpt else "microsoft/CodeGPT-small-java-adaptedGPT2"
    checkpoint = "bigcode/starcoderbase-1b" if args.peft else checkpoint
    device = torch.device(f"cuda:{args.cuda}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = max_input_len
    tokenizer.add_eos_token = True
    code_sep_token = '[CODE_SEP]'
    fname = f'../predictions/{checkpoint[checkpoint.rfind("/"):]}-tfidf-type-all-{args.fname}.jsonl'
    logger.add(f'logs/{args.log}', rotation="5 MB")
    if args.wrong:
        reGenerateWrong()
    else:
        generateAll()