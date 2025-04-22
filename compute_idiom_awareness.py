from cli import CLI
import os
from transformer_lens import HookedTransformer, HookedEncoder, loading_from_pretrained
import torch as t
from data import EPIE_Data
from idiom_awareness import IdiomAwareness
from transformers import AutoTokenizer, BertForMaskedLM

cli = CLI()
os.makedirs(f"./scores/loss/{cli.model_name}", exist_ok=True)
os.makedirs(f"./scores/next_word_prediction/{cli.model_name}", exist_ok=True)

# Saves computation time
t.set_grad_enabled(False)

# if "bert" in cli.model_name:
#     tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
#     hf_model = BertForMaskedLM.from_pretrained("google-bert/bert-base-cased")
#     model = HookedEncoder(loading_from_pretrained.get_pretrained_model_config("bert-base-cased")).from_pretrained("bert-base-cased", hf_model = hf_model, tokenizer = tokenizer)

# else:
#     model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name)
model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name)
epie = EPIE_Data()
scorer = IdiomAwareness(model)
print(f"Running on device {scorer.device}.")

for i in range(len(cli.data_split)):
    split = cli.data_split[i]
    print("\nProcessing split: ", split)
    if split == "formal":
        data = epie.create_hf_dataset(epie.formal_sents[cli.start:cli.end], epie.tokenized_formal_sents[cli.start:cli.end], epie.tags_formal[cli.start:cli.end])
    elif split == "trans":
        data = epie.create_hf_dataset(epie.trans_formal_sents[cli.start:cli.end], epie.tokenized_trans_formal_sents[cli.start:cli.end], epie.tags_formal[cli.start:cli.end])
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
    data = data.add_column("idiom_pos", scorer.idiom_positions[cli.start:cli.end])

    if cli.batch_sizes[i] == None:
        batch_size = 1
    else:
        batch_size = int(cli.batch_sizes[i])
    
    ckp_file = f"./scores/loss/{cli.model_name}/loss_{split}_{cli.start}_{cli.end}.pt"
    
    print("\nLoss:\n")
    data.map(lambda batch: scorer.compute_loss_batched(batch, ckp_file), batched=True, batch_size = batch_size)

    print(f"\nAverage Loss: {t.mean(scorer.loss)}\n")
    scorer.loss = None
    t.cuda.empty_cache()

    print("\nNext Word Generation:\n")
    data.map(lambda batch: scorer.predict_next_word_batched(batch), batched=True, batch_size = batch_size)

    with open(f"./scores/next_word_prediction/{cli.model_name}/nwg_{split}_{cli.start}_{cli.end}.txt", 'w', encoding = "utf-8") as f:
        f.write(scorer.explore_results())

    scorer.num_correct = 0
    scorer.total = 0
    scorer.incorrect_answers = []
    scorer.correct_answers = []

