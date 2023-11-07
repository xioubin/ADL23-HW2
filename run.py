# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
import nltk

def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

    return preds

device = 1 if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("summarization")

dataset = load_dataset('data', data_files={'public.jsonl'})

test_dataloader = [text for text in dataset['maintext']]
text_ids = [text for text in dataset['id']]
result = []


for text, id in zip(test_dataloader, text_ids):
    model_input = tokenizer.encode(text, return_tensors="pt", max_length=256, truncation=True).to(device)
    model_output = model.generate(model_input, max_length=64, num_beams=10, no_repeat_ngram_size=2, early_stopping=True)
    summary = tokenizer.decode(model_output[0], skip_special_tokens=True)
    result.append({'title': summary, 'id': id})

with open('results/beam_search_results.jsonl', 'w') as f:
    for text in result:
        output_str = '{' + f"\"title\": \"{text['title']}\", \"id\": \"{text['id']}\"" + '}\n'
        f.write(output_str)
