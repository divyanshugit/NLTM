from transformers import AutoTokenizer, MBartForConditionalGeneration


model_name = "ai4bharat/IndicBART"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, do_lower_case=False, use_fast=False, keep_accents=True
)

model = MBartForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_name
)

filepath = f"/nlsasfs/home/ttbhashini/prathosh/divyanshu/watExp/models/IndicBART"

model.save_pretrained(save_directory=filepath)
tokenizer.save_pretrained(save_directory=filepath)

