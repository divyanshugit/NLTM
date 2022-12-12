import pandas as pd

hi_ka_df = pd.read_csv("/nlsasfs/home/ttbhashini/prathosh/divyanshu/NLTM/data_parallel/data_hindi_kannada.csv")
ka_sa_df = pd.read_csv("/nlsasfs/home/ttbhashini/prathosh/divyanshu/NLTM/data_parallel/data_kannada_sanskrit.csv")

hi_ka_df.drop(["devanagari_kannada","Kannada"],axis=1,inplace=True)

df1 = hi_ka_df.drop_duplicates(subset='Unnamed: 0', keep='first')
df2 = ka_sa_df.drop_duplicates(subset='Unnamed: 0', keep='first')

final_df = pd.merge(df1, df2, on='Unnamed: 0', how='inner')

final_df.drop(["Unnamed: 0","Kannada"],axis=1,inplace = True)
final_df.rename(columns = {'devanagari_kannada':'Kannada'}, inplace = True)

final_df.to_csv("final_test_data.csv",index=False)

from datasets import Dataset

dataset = Dataset.from_pandas(final_df,split='train')

if '__index_level_0__' in dataset.column_names:
        dataset = dataset.remove_columns(['__index_level_0__'])

dataset = dataset.train_test_split(test_size=0.10, shuffle=False)    

#Utils setup
import transformers

model_name = "ai4bharat/IndicBART"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

s_lang = 'hi'
t_lang = 'ka'
source_lang = 'Hindi'
target_lang = 'Kannada'

max_input_length = 128
max_target_length = 128
batch_size = 16

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

model_name = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/IndicBART"
tokenizer_hi = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_length = 128
num_beams = 4
bos_id = tokenizer._convert_token_to_id_with_added_voc('<s>')
eos_id = tokenizer._convert_token_to_id_with_added_voc('</s>')
pad_id = tokenizer._convert_token_to_id_with_added_voc('<pad>')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_name = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/IndicBART_sanskrit_kannada"
tokenizer_sa = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

model_sa = AutoModelForSeq2SeqLM.from_pretrained(model_name)

import torch

def get_hi_enc(sentence):
    #sentence += ' </s>' + f' <2hi>'
    #tokenized_sentence = tokenizer_sa(sentence, add_special_tokens=True, return_tensors='pt', padding=True).input_ids
    sentence = torch.tensor([sentence],dtype=torch.int32)
    model_sa._modules["model"]._modules["encoder"]._modules["layers"]._modules["5"]._modules["final_layer_norm"].register_forward_hook(get_activation('fc3'))
    output = model_sa(sentence)
    return  activation['fc3']

max_length = 128
num_beams = 4
bos_id = tokenizer._convert_token_to_id_with_added_voc('<s>')
eos_id = tokenizer._convert_token_to_id_with_added_voc('</s>')
pad_id = tokenizer._convert_token_to_id_with_added_voc('<pad>')

def get_ka_dec(sentence):
    #sentence += ' </s>' + f' <2sa>'
    #tokenized_sentence = tokenizer_hi(sentence, add_special_tokens=True, return_tensors='pt', padding=True).input_ids
    sentence = torch.tensor([sentence], dtype = torch.int32)
    model_hi._modules["model"]._modules["decoder"]._modules["layers"]._modules["5"]._modules["final_layer_norm"].register_forward_hook(get_activation('fc3'))
    output = model_hi(sentence) # tensor should go in
    return  activation['fc3']


from tqdm import tqdm

def preprocess_function(examples):

        sentences= examples[source_lang]
        sa_sentences = examples["Sanskrit"]
        inputs = [example + ' </s>' + f' <2{s_lang}>' for example in examples[source_lang]]
        targets = [f'<2{t_lang}> ' + example + ' </s>' for example in examples[target_lang]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        
        hi_enc_list = []
        ka_dec_list = []

        model_inputs['labels'] = labels['input_ids']
        model_inputs["input_ids_1"] = model_inputs['input_ids']
        model_inputs["input_ids_2"] = model_inputs['labels']
        #print(model_inputs['input_ids'][1])

        for i in tqdm(range(len(inputs))):
            # print(len(model_inputs['input_ids'][i]),model_inputs['input_ids'][i])
            # print
            #enc_val = get_hi_enc(sentences[i])
            #print("inside-loop")
            enc_val = get_hi_enc(model_inputs['input_ids'][i])
            hi_enc_list.append(enc_val)
            dec_val = get_ka_dec(model_inputs['labels'][i])
            ka_dec_list.append(dec_val)
        model_inputs['hi_enc'] = hi_enc_list
        model_inputs['ka_dec'] = ka_dec_list
        #model_inputs['ka_dec'] = (ka_dec_list, model_inputs['labels'])
        
        #a.append(model_inputs)
        print("----Complete----")
        return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
#NIos + MKB--12K approx sentence-- 56mins

#from datasets import save_to_disk

tokenized_datasets.save_to_disk("/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_layer5")
