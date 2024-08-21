import argparse
from custom_trainer import CustomTrainer
import random
import editdistance
from itertools import tee
from datetime import datetime
import torch
print(torch.cuda.is_available())
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import T5ForConditionalGeneration, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, \
    get_linear_schedule_with_warmup, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding
from prepare_dataset import getDataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

parser = argparse.ArgumentParser(description='Finetune model')
parser.add_argument('--checkpoint', type=str, help='Model checkpoint', default="Salesforce/codet5p-220m")
parser.add_argument('--mode', type=str, default='all', help='Set the mode to either java, android, all, or wrong')
parser.add_argument('--batchsize', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=6, help='Epoch Nums')
parser.add_argument("--resume", action="store_true", help="Resume or not")
parser.add_argument("--compare", type=str, default='', help='Set _notfidf or _noandroid or _nometadata or none')
args = parser.parse_args()

code_sep_token = '[CODE_SEP]'
token_config = {'additional_special_tokens': [code_sep_token]}
checkpoint = args.checkpoint
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
tokenizer.add_eos_token = True
def model_init():
    if args.checkpoint in ["Salesforce/codet5p-220m", "Salesforce/codet5p-770m"]:
        data_collator_ = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
        tokenizer.add_special_tokens(token_config)
        return T5ForConditionalGeneration.from_pretrained(checkpoint), data_collator_
    tokenizer.pad_token = tokenizer.eos_token
    token_config['additional_special_tokens'].append('[TO_COMPLETE]')
    token_config['additional_special_tokens'].append('[SEP]')
    tokenizer.add_special_tokens(token_config)
    data_collator_ = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    if args.checkpoint in ["bigcode/starcoderbase-1b"]:
        login(token="hf_token")
        tokenizer.pad_token = tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        model_ = AutoModelForCausalLM.from_pretrained(checkpoint, use_cache=True,
                                                      quantization_config=quantization_config, low_cpu_mem_usage=True)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_proj", "c_attn", "q_attn"]
        )
        model_ = get_peft_model(model_, lora_config)
        model_.print_trainable_parameters()
        return model_, data_collator_
    if args.checkpoint in ["microsoft/CodeGPT-small-java-adaptedGPT2"]:
        return AutoModelForCausalLM.from_pretrained(checkpoint), data_collator_
    return AutoModelForCausalLM.from_pretrained(checkpoint), data_collator_
model, data_collator = model_init()
# tokenizer.add_special_tokens(token_config)
model.resize_token_embeddings(len(tokenizer))
model.decoder_only = True
tokenizer.model_max_length = 1024

max_input_len = tokenizer.model_max_length
max_output_len = 128

def tokenize(task):
    target = '# Determine if the information above is useful, Complete The Following Android Code:'
    completion_index = task['prompt'].find(target)
    if completion_index != -1:
        code_context = task['prompt'][completion_index + len(target) + 1:]
        task['prompt'] = (task['prompt'][:completion_index + len(target) + 1]
                          + " " + code_sep_token + " " + code_context)
    res = { "flag": True }
    tokens = tokenizer.encode(task['prompt'])
    if len(tokens) > max_input_len:
        truncated_tokens = tokens[-max_input_len:]
        res['flag'] = (code_sep_token in truncated_tokens)
    else:
        truncated_tokens = tokens
    truncated_text = tokenizer.decode(truncated_tokens)
    if args.checkpoint in ["microsoft/CodeGPT-small-java-adaptedGPT2", "Salesforce/codegen-350M-multi", "bigcode/starcoderbase-1b"]:
        truncated_text = truncated_text + '[TO_COMPLETE]' + task['ground_truth']
        tokens = tokenizer(truncated_text, max_length=max_input_len + max_output_len, truncation=True, return_tensors=None,
                           padding="max_length")
    else:
        tokens = tokenizer(truncated_text, max_length=max_input_len, truncation=True, return_tensors=None,
                           padding=True)
    res['input_ids'] = tokens['input_ids'].copy()
    if args.checkpoint in ["microsoft/CodeGPT-small-java-adaptedGPT2", "Salesforce/codegen-350M-multi", "bigcode/starcoderbase-1b"]:
        labels = res['input_ids']
        if tokenizer.additional_special_tokens_ids[0] in labels:
            index = labels.index(tokenizer.additional_special_tokens_ids[0])
            labels = [-100 if 0 <= i <= index else x for i, x in enumerate(labels)]
        res['labels'] = labels
    else:
        tokens = tokenizer(task['ground_truth'], return_tensors=None, max_length=max_output_len, truncation=True,
                           padding="max_length")
        res['labels'] = tokens['input_ids'].copy()
    return res

def compute_ES(eval_pred):
    logits, labels = eval_pred
    logits = logits[0] if isinstance(logits, tuple) else logits
    pred_texts = tokenizer.batch_decode(logits.argmax(axis=-1), skip_special_tokens=True)
    scores = []
    val_set_iter, val_set_copy = tee(val_set, 2)
    for prediction, target in zip(pred_texts, val_set_iter):
        target_lines = (line.strip() for line in target.splitlines() if line.strip())
        target_str = ''.join(target_lines)
        prediction_lines = (line.strip() for line in prediction.splitlines() if line.strip())
        prediction_str = ''.join(prediction_lines)
        cur_score = 1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        scores.append(cur_score)
        del target_lines, prediction_lines, target_str, prediction_str
    del val_set_copy
    return {'es': sum(scores) / len(val_set)}

def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    # Pad input_ids and labels
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is the ignore index for cross-entropy loss
    # Create attention mask
    attention_mask = (input_ids != 0).long()
    # Create the batch
    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    return batch

data_type = args.mode # wrong/java/android/all
if data_type == 'wrong':
    train_set, val_set, test_set = getDataset(device, tokenizer, model, 'wrong_tasks_train.json', max_input_len)
else:
    train_set, val_set, test_set = getDataset(device, tokenizer, model, None, max_input_len)

train_set = [sample for sample in train_set if sample.get('type') == data_type or data_type in ['wrong', 'all']]
val_set = [sample for sample in val_set if sample.get('type') == data_type or data_type in ['wrong', 'all']][:200]

random.shuffle(train_set)
random.shuffle(val_set)

train_set_tokens = []
for sample in train_set:
    tmp = tokenize(sample)
    if tmp['flag']:
        train_set_tokens.append(tmp)
val_set_tokens = [tokenize(sample) for sample in val_set]
val_set = [sample['ground_truth'] for sample in val_set]

model.train() # put model back into training mode

batch_size = args.batchsize
output_dir = f"output_checkpoints_{data_type}"
model_name = args.checkpoint[args.checkpoint.find("/") + 1:]

training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-5,
    gradient_accumulation_steps=4,
    # fp16=True,
    dataloader_num_workers=0,
    bf16=True,
    weight_decay=1e-4,
    num_train_epochs=args.epochs,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    warmup_steps=100,
    output_dir=output_dir,
    save_total_limit=1,
    ignore_data_skip=False,
    load_best_model_at_end=True,
    group_by_length=True, # group sequences of roughly the same length together to speed up training
    run_name=f"{model_name}-{data_type}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    logging_dir='./logs',
    logging_steps=100,
    remove_unused_columns=True,
)

optimizer = AdamW(
    model.parameters(),
    lr=training_args.learning_rate,
    weight_decay=training_args.weight_decay,
)

num_training_steps = len(train_set_tokens) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps,
)

trainer = CustomTrainer(
    model=model,
    # model_init=model_init,
    tokenizer=tokenizer,
    train_dataset=train_set_tokens,
    eval_dataset=val_set_tokens,
    args=training_args,
    # data_collator=collate_fn,
    data_collator=data_collator,
    # compute_metrics=compute_ES,
)

trainer.optimizer = optimizer
trainer.lr_scheduler = scheduler
model.config.use_cache = False

trainer.train(resume_from_checkpoint=args.resume)
model.save_pretrained(f"output_{data_type}{args.compare}")