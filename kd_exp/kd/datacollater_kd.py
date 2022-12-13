import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
import torch.nn.functional as F
import torch



@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np
        # print("datacollater-called")
        # self.features = features
        # print(len(features))
        # print(features[0].keys())
        # print(len(features[0]['labels']))
        # print("---------------------------------*****_________________")
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)  # max length
            #print("max-label-length in collater-",max_label_length) #84,74,..changing for each batch
            if self.pad_to_multiple_of is not None:  # not executing
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
                #print("max-label-length in collater-",max_label_length)
            feature_input_ids_1 = []
            feature_input_ids_2 = []
            feature_hi_enc = []
            feature_ka_dec = []
            padding_side = self.tokenizer.padding_side
            for feature in features: # across each sentence: dictionary
                if "hi_enc" in feature.keys():
                                hi_enc_lis = feature["hi_enc"] # list converted to tensor
                                #pos_tensor_padded = F.pad(torch.Tensor(pos_tensor),pad=(0,max_len - len(pos_tensor)), mode='constant',value = 0)
                                feature_hi_enc.append(hi_enc_lis)
                                del feature["hi_enc"]  ###to delete pos
                if "ka_dec" in feature.keys():
                                ka_dec_lis = feature["ka_dec"] # list converted to tensor
                                #pos_tensor_padded = F.pad(torch.Tensor(pos_tensor),pad=(0,max_len - len(pos_tensor)), mode='constant',value = 0)
                                feature_ka_dec.append(ka_dec_lis)
                                del feature["ka_dec"]  ###to delete pos

                if "input_ids_1" in feature.keys():
                                input_ids_1_lis = feature["input_ids_1"] # list converted to tensor
                                #pos_tensor_padded = F.pad(torch.Tensor(pos_tensor),pad=(0,max_len - len(pos_tensor)), mode='constant',value = 0)
                                feature_input_ids_1.append(input_ids_1_lis)
                                del feature["input_ids_1"]  ###to delete pos

                if "input_ids_2" in feature.keys():
                                input_ids_2_lis = feature["input_ids_2"] # list converted to tensor
                                #pos_tensor_padded = F.pad(torch.Tensor(pos_tensor),pad=(0,max_len - len(pos_tensor)), mode='constant',value = 0)
                                feature_input_ids_2.append(input_ids_2_lis)
                                del feature["input_ids_2"]  ###to delete pos


                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"])) #padding
                if isinstance(feature["labels"], list): # true
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        ### dict_keys(['input_ids', 'attention_mask', 'labels', 'pos']) for pos
        # features1 = features[0]['input_ids','attention_mask','labels']
        # del features[0]['pos']


        #self.features_before_pad = features
        #del features
        #list of 16 dictionaries--each dictionary 1 data with keys: dict_keys(['input_ids', 'attention_mask', 'labels', 'pos'])
        """
        features_pos = {}
        # features_lis = []
        features_pos['pos'] = [feature.pop('pos') for feature in features]
        # features_pos['pos'] = features_lis
        self.features_pos = features_pos
        # self.features_lis = features_lis
        
        # features_pos = features.copy()
        # for i in['input_ids', 'attention_mask', 'labels']:
        #     del features_pos[0][i]
        # del features[0]["pos"]
        import torch.nn.functional as F
        import torch
        pos_lis = features_pos["pos"] # list of list
        self.pos_lis = pos_lis
        self.features_pos = features_pos
        """
        #beware of max- length --- here assigning
        """
        max_len = 128
        # self.max_length = 128
        pos_pad = [F.pad(torch.Tensor(pos),pad=(0,max_len - len(pos)), mode='constant',value = 0) for pos in pos_lis ]
        #pos_pad is a list of tensor--> converted to tensor of tensor in next step:
        pos_pad = torch.stack((pos_pad))
        features_pos["pos"] = pos_pad

        """
        # print(features[0].keys())
        # print(type(features[0]))

        # features = features[0][]
        # features_ = [{i:features[j][i] for i in ('hi_enc', 'ka_dec') if i in features[j]} for j in range(len(features))]
        # print(features_[0].keys())
        # features = [{i:features[j][i] for i in ('input_ids', 'attention_mask', 'labels') if i in features[0]} for j in range(len(features))] 

        # print(len(features_))
        # print(features_.keys(),features.keys())     
        # features_ = features[0][]
        # print("Just Checking")
        features = self.tokenizer.pad(   # gives 4 dictionaries:  dict_keys(['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']))  16 in each
            features,
            padding=self.padding,
            max_length=self.max_length, #None
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        ) 
        # print("After Padding")
        # print(features.keys())
        # print("After Updating again")
        # print(features.keys())
        # input_ids_max_len = len(features["input_ids"][0])
        # d_type_input_ids = features["input_ids"][0].dtype
        #print(d_type_input_ids)

        feature_hi_enc_ = [hi_enc for hi_enc in feature_hi_enc ]
        # feature_hi_enc_tensor = torch.cat((feature_hi_enc_padded))
        feature_hi_enc_dict_final = {"hi_enc":feature_hi_enc_}
        # print(len(feature_hi_enc_dict_final['hi_enc'][0]))
        # print(len(feature_hi_enc),len(feature_hi_enc_))
        feature_ka_dec_ = [ka_dec for ka_dec in feature_ka_dec]
        # feature_ka_dec_tensor = torch.cat((feature_ka_dec_padded))
        feature_ka_dec_dict_final = {"ka_dec":feature_ka_dec_}
        # print(len(feature_ka_dec_dict_final))

        feature_input_ids_1_ = [i for i in feature_input_ids_1]
        # feature_hi_enc_tensor = torch.cat((feature_hi_enc_padded))
        feature_input_ids_1_dict_final = {"input_ids_1":feature_input_ids_1_}
        # print(len(feature_hi_enc_dict_final['hi_enc'][0]))
        # print(len(feature_hi_enc),len(feature_hi_enc_))
        feature_input_ids_2_ = [i for i in feature_input_ids_2]
        # feature_ka_dec_tensor = torch.cat((feature_ka_dec_padded))
        feature_input_ids_2_dict_final = {"input_ids_2":feature_input_ids_2_}
        # print(len(feature_ka_dec_dict_final))

        features.update(feature_hi_enc_dict_final)
        features.update(feature_ka_dec_dict_final)
        features.update(feature_input_ids_1_dict_final)
        features.update(feature_input_ids_2_dict_final)

        #for i in range(len(features)):
        #    features[i]['pos'] = features_pos['pos'][i]
        # final_features = [i['pos'] =  for i in range(len(features))]
        
        # final_features = [{**features[0], **features_pos}]
        # self.final_features = final_features

        # prepare decoder_input_ids
        # print("First Check")
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            # print("Inside Condition")
            # print(features['labels'].shape)
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # print("Second_Check")
        # features["input_ids"] = (features["input_ids"], 
        #                          features["hi_enc"],
        #                          features["ka_dec"],
        #                         #  feature_pp_dict_final["pp"],
        #                          )
        features["input_ids"] = (features["input_ids"], features["hi_enc"], 
                                    features["ka_dec"], features["input_ids_1"], features["input_ids_2"]  )
        del features['hi_enc']
        del features['ka_dec']
        del features['input_ids_1']
        del features['input_ids_2']

        #print(features.keys())
        #self.feat_ = features  
        #final_features = {**features,**feature_pos_dict_final} # 5 dicts
        #self.final_features = final_features
        # print("Are we done here??")
        

        return features