# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers.pipelines import text2text_generation
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


gen_strategy = {
    "greedy": {
        "do_sample": False,
        "num_beams": 1,
    },
    "beam_search": {
        "do_sample": False,
        "num_beams": 30,
        "no_repeat_ngram_size": 2,
    },
    "top_k": {
        "do_sample": True,
        "top_k": 50,
    },
    "top_p": {
        "do_sample": True,
        "top_p": 0.92,
    },
    "temprature": {
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 0,
    },
}


# load the model
pipe = pipeline('summarization', model='summarization', device=0)

# load the dataset
dataset = load_dataset('data', data_files={'test': 'public.jsonl'})
dataset = dataset['test']

test_dataloader = [text for text in dataset['maintext']]
text_ids = [text for text in dataset['id']]
ref = [text for text in dataset['title']]
# print(test_dataloader)

# run the model
for strategy in gen_strategy:
    progress_bar = tqdm(range(len(test_dataloader)), desc=strategy)
    results = []
    for batch in test_dataloader:
        # print(strategy)
        # print(gen_strategy[strategy])
        summaries = pipe(batch, max_length=64, **gen_strategy[strategy])
        results.extend(summaries)
        progress_bar.update(1)

    # write the results to a file
    with open(f'results/{strategy}_results.jsonl', 'w') as f:
        for text, id in zip(results, text_ids):
            output = '{' + f"\"title\": \"{text['summary_text']}\", \"id\": \"{id}\"" + '}\n'
            # print(output)
            f.write(output)


