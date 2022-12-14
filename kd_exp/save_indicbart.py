import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.append("/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/")

import torch
import numpy as np
import pandas as pd
from transformers import MBartForConditionalGeneration
from datasets import load_dataset,load_metric
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path = model_name)

filepath = f'models/IndicBART'

model.save_pretrained(save_directory=filepath)
tokenizer.save_pretrained(save_directory=filepath)
