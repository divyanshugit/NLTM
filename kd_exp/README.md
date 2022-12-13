# Knowledge Distillation Experiment for Neural Machine Translation

```
NLTM/kd_exp
├── README.md                             <-- You are here 📌
├── model_weights                         <-- Saves the model weitghts
├── kd_gen_data.py			  <-- Generate data for KD training
├── train_vanilla.py          		  <-- Trains IndicBART for Hindi -- Kannada
└── train_kd.py                           <-- Trains IndicBART with KD for Hindi -- Kannada
```

## Get started:

```python

$ python train.py --s_lang Hindi --t_lang Kannada # You can choose s_lang and t_lang from Hindi, Sanskrit or Kannada

$ python test.py --source_lang=Hindi --target_lang=Kannada # You can source_lang and target_lang from Hindi, Sanskrit or Kannada

```

## Models:

```
- IndicBART-Hindi-Sanskrit(model_1)
- IndicBART-Sanskrit-Kannada(model_2)
- IndicBART-Hindi-Kannada(model_3)
- vanilla IndicBART(model)
```

## Experiment Set-up:

We will train model by utilizing model_1 & model_2 and compare it with model_3


