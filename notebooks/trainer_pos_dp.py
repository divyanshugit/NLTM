# %%
import sys
sys.path.append("/home/ece/utkarsh/NLTM/")

# %%
import torch
import os
from interlingua_model.model.modelling_mbart_modified_pos_dp import MBartForConditionalGeneration
from interlingua_model.data_handling.datacollater_pos_dp import DataCollatorForSeq2Seq

# %%
#import wandb
# wandb.init(project="NLTM_POS", entity="divyanshuusingh", mode = "disabled")

# %%
torch.cuda.is_available()
os.environ["WANDB_DISABLED"] = "true"

# %%
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          #DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)


# %%
model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
#

# %%
model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path = "/home/ece/utkarsh/old_source_interlingua-model/")

# %%
#model.__dict__

# %% [markdown]
# # seeing class MBartEncoder(MBartPreTrainedModel) 

# %%
#model # yeah

# %%
#model._modules['model']._modules['encoder']

# %%
#model._modules['model']._modules['encoder'].embed_positions

# %%
""" 
lets write mbartposembedding---

Q1. what input is model taking and then giving to function 

"""

# %% [markdown]
# # import dataset cdac

# %%
from datasets import load_from_disk

# %%
tok_data = load_from_disk('/home/ece/utkarsh/syn_data')

# %%
tok_data["train"]

# %% [markdown]
# # training

# %%
from interlingua_model.trainer.trainer_seq2seq_pos_dp import Seq2SeqTrainer
from interlingua_model.trainer import trainer_pos_dp
from interlingua_model.data_handling.datacollater_pos_dp import DataCollatorForSeq2Seq

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  

# %%
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          #DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments
                          )
import numpy as np

# %%
batch_size = 16

# %%
args = Seq2SeqTrainingArguments(
            'translation_pos_dp',
            evaluation_strategy='epoch',
            learning_rate=0.001,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.0001,
            save_total_limit=2,
            num_train_epochs=20,
            #label_names = ["pos"],
            predict_with_generate=True)
            #report_to="wandb"

# %%
#args.__dict__

# %%
tokenized_datasets = tok_data

# %%
from datasets import load_metric

# %%
metric = load_metric('sacrebleu')

# %%
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


# %%
trainer = Seq2SeqTrainer(
              model,
              args,
              train_dataset=tokenized_datasets['train'],
              eval_dataset=tokenized_datasets['test'],
              data_collator=data_collator,
              tokenizer=tokenizer,
              compute_metrics=compute_metrics)  



# %%
print(trainer.train())
"""
# %% [markdown]
# ## evaluate after training

# %%
max_target_length = 128


# %%
trainer.evaluate()

predict_dataset = tokenized_datasets['test']
predict_results = trainer.predict(predict_dataset,
                                    metric_key_prefix='predict',
                                    max_length=max_target_length)

predictions = tokenizer.batch_decode(predict_results.predictions,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)

predictions = [pred.strip() for pred in predictions]

# input_sentences = tokenizer.batch_decode(tokenized_datasets['test']['input_ids'], skip_special_tokens=True)
# output_sentences = tokenizer.batch_decode(tokenized_datasets['test']['labels'], skip_special_tokens=True)


# %%
predictions

# %%
import pandas as pd

# %%
source_lang = 'Hindi'
target_lang = 'Sanskrit'

# %%
language_code = {'Hindi':'hi', 'Sanskrit':'sa', 'Kannada':'kn'}
#s_lang = language_code[FLAGS.source_lang.capitalize()]
#t_lang = language_code[FLAGS.target_lang.capitalize()]
s_lang = 'hi'
t_lang = 'sa'

# %%

# if FLAGS.target_lang.capitalize() == 'Kannada':
#     kannada_sentences = []
#     for sentence in predictions:
#         sentence = UnicodeIndicTransliterator.transliterate(sentence, 'hi', 'kn')
#         kannada_sentences.append(sentence)      

predictions_data = pd.DataFrame({f'{source_lang}':tokenized_datasets['test'][source_lang.capitalize()], 
                                    f'{target_lang}':tokenized_datasets['test'][target_lang.capitalize()],
                                    'predictions':  predictions})

predictions_data.to_csv(f'{source_lang}_{target_lang}_predictions.csv')                                                                                                                                                                       

"""

