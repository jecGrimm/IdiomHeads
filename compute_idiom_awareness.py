from cli import CLI
import os
from transformer_lens import HookedTransformer
import torch as t
from data import EPIE_Data
from idiom_awareness import IdiomAwareness

cli = CLI()
os.makedirs(f"./scores/loss/{cli.model_name}", exist_ok=True)
os.makedirs(f"./scores/next_word_prediction/{cli.model_name}", exist_ok=True)

# Saves computation time
t.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained(cli.full_model_name)

scorer = IdiomAwareness(model, filename = cli.idiom_file)
print(f"\nRunning compute_idiom_awareness on device {scorer.device}.")

for i in range(len(cli.data_split)):
    split = cli.data_split[i]
    print(f"\nProcessing split {split}:")

    epie = EPIE_Data(experiment="awareness", model_id=cli.model_name, split=split)

    if cli.batch_sizes[i] == None:
        batch_size = 1
    else:
        batch_size = int(cli.batch_sizes[i])

    start = cli.start[i]
    end = cli.end[i]

    # get data
    if split == "formal":
        data = epie.create_hf_dataset(epie.formal_sents[start:end], epie.tokenized_formal_sents[start:end], epie.tags_formal[start:end])
    elif split == "trans":
        data = epie.create_hf_dataset(epie.trans_formal_sents[start:end], epie.tokenized_trans_formal_sents[start:end], epie.tags_formal[start:end])
    elif split == "static":
        data = epie.create_hf_dataset(epie.static_sents[start:end], epie.tokenized_static_sents[start:end], epie.tags_static[start:end])
    else:
        raise Exception(f"Split {split} not in the dataset, please choose either formal or trans as optional argument -d")
    
    # get idiom positions
    if scorer.idiom_positions == []:
        data.map(lambda batch: scorer.get_all_idiom_pos(batch), batched=True, batch_size=batch_size)
        scorer.store_all_idiom_pos(cli.idiom_file)
    data = data.add_column("idiom_pos", scorer.idiom_positions[start:end])
    
    # perform experiment
    ckp_file = f"./scores/loss/{cli.model_name}/loss_{split}_{start}_{end}.pt"
    
    print("\nLoss:\n")
    data = data.filter(lambda x: len(x["tokenized"]) > 1)
    data.map(lambda batch: scorer.compute_loss_batched(batch, ckp_file), batched=True, batch_size = batch_size)

    print(f"\nAverage Loss: {t.mean(scorer.loss)}\n")
    scorer.loss = None
    t.cuda.empty_cache()

    print("\nNext Word Generation:\n")
    data.map(lambda batch: scorer.predict_next_word_batched(batch), batched=True, batch_size = batch_size)

    with open(f"./scores/next_word_prediction/{cli.model_name}/nwp_{split}_{start}_{end}.txt", 'w', encoding = "utf-8") as f:
        f.write(scorer.explore_results())

    scorer.num_correct = 0
    scorer.total = 0
    scorer.incorrect_answers = []
    scorer.correct_answers = []
    t.cuda.empty_cache()

