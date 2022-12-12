import os
os.environ['CUDA_VISIBLE_DEVICES']="2"
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.append("/nlsasfs/home/ttbhashini/prathosh/divyanshu/NLTM/")

import torch
import numpy as np
from kd.model.modelling_mbart_kd import MBartForConditionalGeneration

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          #DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/IndicBART")

from datasets import load_from_disk
tokenized_datasets = load_from_disk("/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_layer5")

from kd.trainer.trainer_seq2seq_kd import Seq2SeqTrainer
from kd.trainer import trainer_kd
from kd.datacollater_kd import DataCollatorForSeq2Seq

batch_size = 16

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  

args = Seq2SeqTrainingArguments(
            'kd5',
            evaluation_strategy='no',
            learning_rate=0.001,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.0001,
            save_total_limit=2,
            num_train_epochs=20,
            # _setup_devices = torch.cuda.set_device(7),
            #label_names = ["pos"],
            predict_with_generate=True)

from datasets import load_metric
metric = load_metric('sacrebleu')

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

def compute_metrics(eval_preds):
        #print("compute_met called")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {'bleu': result['score']}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result['gen_len'] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        #print("compute-met-end")
        return result

trainer = Seq2SeqTrainer(
              model,
              args,
              train_dataset=tokenized_datasets['train'],
              eval_dataset=tokenized_datasets['test'],
              data_collator=data_collator,
              tokenizer=tokenizer,
              compute_metrics=compute_metrics)  

trainer.train()

output_dir = f'/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/model_weights/run_1'
trainer.save_model(output_dir)
