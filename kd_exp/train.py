import os

import wandb
import sys

sys.path.append("/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/")
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import MBartForConditionalGeneration
from datasets import load_metric
from datasets import Dataset
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--s_lang",
    type=str,
    required=True,
    help="Enter the source language (Hindi, Kannada, Sanskrit)",
)
parser.add_argument(
    "--t_lang",
    type=str,
    required=True,
    help="Enter the target language (Hindi, Kannada, Sanskrit",
)
parser.add_argument("--gpu", type=str, required=True, help="Enter GPU")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
source_lang = args.s_lang.capitalize()
target_lang = args.t_lang.capitalize()

wandb.init(project=f"IndicBART_Experiments", entity="divyanshuusingh")

model_name = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/models/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, do_lower_case=False, use_fast=False, keep_accents=True
)

model = MBartForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_name
)

filepath = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/data/parallel_mkb_hsk_train_data.csv"

df = pd.read_csv(filepath)


def convert_devanagri(sentence):
    return UnicodeIndicTransliterator.transliterate(sentence, "kn", "hi")


if source_lang == "Kannada" or target_lang == "Kannada":
    df["devanagari"] = df.Kannada.apply(lambda x: convert_devanagri(x))
    df.drop(["Kannada"], axis=1, inplace=True)
    df.rename(columns={"devanagari": "Kannada"}, inplace=True)

columns = [source_lang, target_lang]

df = pd.DataFrame(df, columns=columns)

dataset = Dataset.from_pandas(df, split="train")

dataset = dataset.train_test_split(test_size=0.10, shuffle=False)


print(dataset)


batch_size = 16
max_input_length = 128
max_target_length = 128

language_code = {"Hindi": "hi", "Sanskrit": "sa", "Kannada": "kn"}
s_lang = language_code[source_lang.capitalize()]
t_lang = language_code[target_lang.capitalize()]


def preprocess_function(examples):
    inputs = [example + " </s>" + f" <2{s_lang}>" for example in examples[source_lang]]
    targets = [f"<2{t_lang}> " + example + " </s>" for example in examples[target_lang]]

    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding=True, truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding=True, truncation=True
        )

    # Changes
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

print(tokenized_datasets)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=f"{source_lang}_{target_lang}",
    # evaluation_strategy='epoch',
    evaluation_strategy="no",
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.0001,
    save_total_limit=2,
    num_train_epochs=20,
    save_strategy="epoch",
    predict_with_generate=True,
    report_to="wandb",
)


# from datasets import load_metric
metric = load_metric("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    # print("compute_met called")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    # print("compute-met-end")
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

output_dir = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/models/IndicBART_{source_lang}_{target_lang}"
trainer.save_model(output_dir)
