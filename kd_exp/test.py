from absl import app
from absl import flags
from absl import logging
import os
import transformers
import torch
import pandas as pd
import numpy as np
import indicnlp
from datasets import Dataset,load_dataset, load_metric

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          MBartForConditionalGeneration,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

# Usage:-python3 train.py --data_dir='./data_hindi_kannada.csv' --source_lang='hindi' --target_lang='kannada'

FLAGS = flags.FLAGS

filepath = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/data/parallel_mkb_hsk_test_data.csv"

flags.DEFINE_string('data_dir', default=filepath, help=('Data file.'))

flags.DEFINE_string('source_lang', default='Hindi', help=('The source language.'))

flags.DEFINE_string('target_lang', default='Sanskrit', help=('The target language.'))
 
def main(argv):
    if not FLAGS.data_dir:
       raise app.UsageError('Please provide the path to data file.')

    if not FLAGS.source_lang:
       raise app.UsageError('Please provide the source language.')

    if not FLAGS.target_lang:
       raise app.UsageError('Please provide the target language.')

    data_file = FLAGS.data_dir
    source_lang = FLAGS.source_lang.capitalize()
    target_lang = FLAGS.target_lang.capitalize() 
    df = pd.read_csv(data_file)

  
    def convert_devanagri(sentence):
        return UnicodeIndicTransliterator.transliterate(sentence, 'kn', 'hi')


    if FLAGS.source_lang == 'Kannada' or FLAGS.target_lang == "Kannada":
        df['devanagari'] = df.Kannada.apply(lambda x: convert_devanagri(x))
        df.drop(["Kannada"],axis=1,inplace = True)
        df.rename(columns = {'devanagari':'Kannada'}, inplace = True)

    columns = [FLAGS.source_lang,FLAGS.target_lang]

    df = pd.DataFrame(df,columns = columns)

    dataset = Dataset.from_pandas(df)
    # dataset = load_dataset('csv', data_files=data_file, header=0, split='train')
    print(dataset)


    if 'Unnamed: 0' in dataset.column_names:
        dataset = dataset.remove_columns(['Unnamed: 0'])

    # dataset = dataset.train_test_split(test_size=0.10, shuffle=False) 

    os.environ['CUDA_VISIBLE_DEVICES'] = '5' 
       
    model_name = f'/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/models/IndicBART_{FLAGS.source_lang.capitalize()}_{FLAGS.target_lang.capitalize()}'
    #print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
    model = MBartForConditionalGeneration.from_pretrained(model_name)


    max_input_length = 256
    max_target_length = 256

    language_code = {'Hindi':'hi', 'Sanskrit':'sa', 'Kannada':'kn'}
    s_lang = language_code[FLAGS.source_lang.capitalize()]
    t_lang = language_code[FLAGS.target_lang.capitalize()]
    metric = load_metric('sacrebleu')

    def preprocess_function(examples):
        inputs = [example + ' </s>' + f' <2{s_lang}>' for example in examples[source_lang]]
        targets = [f'<2{t_lang}> ' + example + ' </s>' for example in examples[target_lang]]

        model_inputs = tokenizer(inputs, max_length=max_input_length,padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length,padding=True, truncation=True)

        #Changes
        labels['input_ids'] = [[(l if l!=tokenizer.pad_token_id else -100) for l in label] for label in labels['input_ids']]    

        model_inputs['labels'] = labels['input_ids']
        return model_inputs


    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    batch_size = 16

    args = Seq2SeqTrainingArguments(
            'translation',
            evaluation_strategy='epoch',
            do_train = False,
            do_predict = True,
            learning_rate=0.001,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.0001,
            predict_with_generate=True)

    #Changes
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
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
        return result

    trainer = Seq2SeqTrainer(
              model,
              args,
              data_collator=data_collator,
              tokenizer=tokenizer,
              compute_metrics=compute_metrics)  

    predict_dataset = tokenized_datasets
    predict_results = trainer.predict(predict_dataset,
                                      metric_key_prefix='predict',
                                      max_length=max_target_length)

    print(f'The prediction metric:{predict_results}')                                    

    predictions = tokenizer.batch_decode(predict_results.predictions,
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)

    predictions = [pred.strip() for pred in predictions]
    
    # input_sentences = tokenizer.batch_decode(tokenized_datasets['test']['input_ids'], skip_special_tokens=True)
    # output_sentences = tokenizer.batch_decode(tokenized_datasets['test']['labels'], skip_special_tokens=True)

    if FLAGS.target_lang.capitalize() == 'Kannada':
       kannada_sentences = []
       for sentence in predictions:
          sentence = UnicodeIndicTransliterator.transliterate(sentence, 'hi', 'kn')
          kannada_sentences.append(sentence)      

    predictions_data = pd.DataFrame({f'{source_lang}':tokenized_datasets[FLAGS.source_lang.capitalize()],
                                     f'{target_lang}':tokenized_datasets[FLAGS.target_lang.capitalize()], 
                                     'predictions':predictions})

    predictions_data.to_csv(f'{FLAGS.source_lang}_{FLAGS.target_lang}_MKB_predictions.csv')
    #predictions_data.to_csv(f'back_translation_train_predictions.csv')                                                                                                                                                                   


if __name__ == '__main__':
    app.run(main)  



