import pandas as pd

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

filepath = "/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/data/parallel_mkb_hsk_train_data.csv"

df = pd.read_csv(filepath)

def convert_devanagri(sentence):
    return UnicodeIndicTransliterator.transliterate(sentence, 'kn', 'hi')

df['devanagari_kannada'] = df.Kannada.apply(lambda x: convert_devanagri(x))

# print(df.head())
# print(df.columns)
df.drop(["Kannada"],axis=1,inplace = True)
df.rename(columns = {'devanagari_kannada':'Kannada'}, inplace = True)
# print(df.head())

from datasets import Dataset

dataset = Dataset.from_pandas(df,split='train')

if '__index_level_0__' in dataset.column_names:
        dataset = dataset.remove_columns(['__index_level_0__'])

dataset = dataset.train_test_split(test_size=0.10, shuffle=False)    

#Utils setup
import transformers

s_lang = 'hi'
t_lang = 'ka'
source_lang = 'Hindi'
target_lang = 'Kannada'

max_input_length = 128
max_target_length = 128
batch_size = 16

model_name = "ai4bharat/IndicBART"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)

model_name = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/models/IndicBART_Hindi_Sanskrit"
tokenizer_hi = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False, keep_accents=True)
model_hi = AutoModelForSeq2SeqLM.from_pretrained(model_name)

max_length = 128
num_beams = 4
bos_id = tokenizer_hi._convert_token_to_id_with_added_voc('<s>')
eos_id = tokenizer_hi._convert_token_to_id_with_added_voc('</s>')
pad_id = tokenizer_hi._convert_token_to_id_with_added_voc('<pad>')

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_name = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/models/IndicBART_Sanskrit_Kannada"

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
bos_id = tokenizer_sa._convert_token_to_id_with_added_voc('<s>')
eos_id = tokenizer_sa._convert_token_to_id_with_added_voc('</s>')
pad_id = tokenizer_sa._convert_token_to_id_with_added_voc('<pad>')

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

tokenized_datasets.save_to_disk("/nlsasfs/home/ttbhashini/prathosh/divyanshu/kd_exp/data/token_data")
