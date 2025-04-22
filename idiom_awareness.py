import torch as t
import os
import json
import re

class IdiomAwareness:
  def __init__(self, model, split: str = "formal"):
    self.model = model
    self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.loss = None
    self.idiom_positions = self.load_all_idiom_pos(split)
    self.num_correct = 0
    self.total = 0
    self.correct_answers = []
    self.incorrect_answers = []

  def load_all_idiom_pos(self, split):
      if os.path.isfile(f"{split}_idiom_pos.json"):
          with open(f"{split}_idiom_pos.json", 'r', encoding = "utf-8") as f:
              return json.load(f) 
      else:
          return [] 
      
  def get_correct_tok(self, tags, toks):
    index = 0
    correct_tok = ""
    if "I-IDIOM" not in tags:
      index = tags.index("B-IDIOM")
      correct_tok = toks[index]
    else:
      reverse_tags = tags[::-1]
      reverse_toks = toks[::-1]
      index = reverse_tags.index("I-IDIOM")
      correct_tok = reverse_toks[index]
    
    return correct_tok
  
  def remove_spaces(self, sent_list: list):
    '''
    This method transforms the tokenized sentences into normal sentences.

    @param sent_list: tokenized sentences 
    @returns cleaned_sents: list of the normal sentences
    '''
    cleaned_sents = []
    for sent in sent_list:
        space_matches = set(re.findall(r"(‘ | ['’\.,?!]| $)", sent))
        del_space_matches = {space_match:space_match.replace(' ', '') for space_match in space_matches}

        for space_match, del_space in del_space_matches.items():
            sent = sent.replace(space_match, del_space)
        cleaned_sents.append(sent)
    return cleaned_sents
  
  def get_correct_toks(self, tags, toks):
    correct_id = max([i for i in range(len(tags)) if "IDIOM" in tags[i]])
    if correct_id >= len(toks):
       correct_id = len(toks)-1
    return self.remove_spaces([" ".join(toks[:correct_id])])[0], toks[correct_id]

  def predict_next_word_batched(self, batch):
    self.total += len(batch["sentence"])
    for i in range(len(batch["sentence"])):
      prompt, correct_tok = self.get_correct_toks(batch["tags"][i], batch["tokenized"][i])
      out_tensor = self.model.generate(prompt, max_new_tokens = 1, verbose = False, return_type = "tokens")
      pred = self.model.to_str_tokens(out_tensor)[-1].strip()

      if pred == correct_tok:
        self.num_correct += 1

        if len(self.correct_answers) < 5:
            self.correct_answers.append(self.model.to_string(out_tensor)[0])
      else:
        if len(self.incorrect_answers) < 5:
          self.incorrect_answers.append((prompt + " -> " + correct_tok, self.model.to_string(out_tensor)[0]))

      del out_tensor
      t.cuda.empty_cache()

  def compute_loss_batched(self, batch, ckp_file):
    batch_loss = t.zeros(len(batch["sentence"]))

    for i in range(len(batch["sentence"])):
      batch_loss[i] = self.model(batch["sentence"][i], return_type="loss")
    
    if self.loss == None:
      self.loss = batch_loss
    else:
      self.loss = t.cat((self.loss, batch_loss), dim = 0)

    del batch_loss
    t.cuda.empty_cache()
    
    t.save(self.loss, ckp_file)

  def explore_results(self):
    output = f"\nThe model generated {self.num_correct} correct answers.\n"
    output += "\nAccuracy: "
    output += "{:.2%}".format(self.num_correct/self.total)
    output += f"\nFive correct outputs: {self.correct_answers}"
    output += f"\nFive incorrect outputs: {self.incorrect_answers}"
    
    print(output)
    return output


