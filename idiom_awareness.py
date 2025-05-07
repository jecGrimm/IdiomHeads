import torch as t
import os
import json
import re
from merge_tokenizers import PythonGreedyCoverageAligner, types

class IdiomAwareness:
  def __init__(self, model, filename: str = "pythia_formal_idiom_pos.json"):
    self.model = model
    self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.loss = None
    self.idiom_positions = self.load_all_idiom_pos(filename)
    self.num_correct = 0
    self.total = 0
    self.correct_answers = []
    self.incorrect_answers = []
    self.aligner = PythonGreedyCoverageAligner()

  def get_all_idiom_pos(self, batch):
    for i in range(len(batch["sentence"])):
        sent = batch["sentence"][i]
        model_str_tokens= self.model.to_str_tokens(sent)
        aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
        self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

  def get_idiom_pos(self, aligned_positions, tags: list):
    epie_idiom_pos = [i for i in range(len(tags)) if "IDIOM" in tags[i]]

    start = None
    end = None
    for epie_pos, model_positions in aligned_positions:
        if epie_pos == epie_idiom_pos[0]:
            start = model_positions[0]
        elif epie_pos == epie_idiom_pos[-1]:
            end = model_positions[-1]
    
    if end == None:
        end = aligned_positions[-1][-1][-1]
    assert start != None and end != None
    return (start, end)
    
  def store_all_idiom_pos(self, filename):
      with open(filename, 'w', encoding = "utf-8") as f:
          json.dump(self.idiom_positions, f)    

  def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
    aligned = self.aligner.align(
        types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
    )
    return list(aligned[0])

  def load_all_idiom_pos(self, filename):
      if os.path.isfile(filename):
          with open(filename, 'r', encoding = "utf-8") as f:
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
    if len(toks) > 1:
      correct_id = max([i for i in range(len(tags)) if "IDIOM" in tags[i]])
      if correct_id >= len(toks):
        correct_id = len(toks)-1
      return self.remove_spaces([" ".join(toks[:correct_id])])[0], toks[correct_id]
    else:
      return None, None

  def predict_next_word_batched(self, batch):
    self.total += len(batch["sentence"])
    for i in range(len(batch["sentence"])):
      prompt, correct_tok = self.get_correct_toks(batch["tags"][i], batch["tokenized"][i])

      if prompt != None and correct_tok != None:
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
      if len(batch["tokenized"][i]) > 1:
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
    output += f"\n\nFive correct outputs: {self.correct_answers}"
    output += f"\n\nFive incorrect outputs: {self.incorrect_answers}"
    
    print(output)
    return output


