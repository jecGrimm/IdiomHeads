import torch as t
import os
import json
import re
from merge_tokenizers import PythonGreedyCoverageAligner, types

class IdiomAwareness:
  def __init__(self, model, filename: str = "pythia_formal_idiom_pos_awareness.json"):
    """
    This class performs the idiom awareness experiment.

    @params
      model: examined model
      filename: file with the idiom positions
    """
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
    """
    This method extracts the idiom positions of a batch.

    @params
        batch: batch of sentences
    """
    for i in range(len(batch["sentence"])):
        sent = batch["sentence"][i]
        model_str_tokens= self.model.to_str_tokens(sent)
        aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
        self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

  def get_idiom_pos(self, aligned_positions, tags: list):
    """
    This method extracts the position of the idioms in the sentence tokenized by the model.

    @params
        aligned_positions: aligned EPIE and model positions
        tags: idiom labels
    @returns 
        start and end position of the idiom
    """
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
    
  def store_all_idiom_pos(self, filename: str):
      """
      This method saves the idiom positions.

      @params
          filename: output file
      """
      with open(filename, 'w', encoding = "utf-8") as f:
          json.dump(self.idiom_positions, f)    

  def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
    """
    This method aligns the tokens retrieved by EPIE with the tokens retrieved by the model.

    @params
        sent: natural sentence
        tokenized_sent: sentence tokenized by EPIE
        model_str_tokens: sentence tokenized by the model
    @returns
        list of the aligned token positions
    """
    aligned = self.aligner.align(
        types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
    )
    return list(aligned[0])

  def load_all_idiom_pos(self, filename: str):
      """
      This methods loads the idiom positions.

      @params
          filename: file with the idiom positions
      @returns the loaded idiom positions
      """
      if os.path.isfile(filename):
          with open(filename, 'r', encoding = "utf-8") as f:
              return json.load(f) 
      else:
          return [] 
  
  def remove_spaces(self, sent_list: list):
    '''
    This method transforms the tokenized sentences into normal sentences.

    @params
      sent_list: tokenized sentences 
    @returns 
      list of the normal sentences
    '''
    cleaned_sents = []
    for sent in sent_list:
        space_matches = set(re.findall(r"(‘ | ['’\.,?!]| $)", sent))
        del_space_matches = {space_match:space_match.replace(' ', '') for space_match in space_matches}

        for space_match, del_space in del_space_matches.items():
            sent = sent.replace(space_match, del_space)
        cleaned_sents.append(sent)
    return cleaned_sents
  
  def get_correct_toks(self, tags: list, toks: list):
    """
    This method extract the correct token and cuts the sentence off before it.

    @param
      tags: idiom labels
      toks: tokenized sentence
    @returns 
      prompt, last idiom token  
    """
    if len(toks) > 1:
      correct_id = max([i for i in range(len(tags)) if "IDIOM" in tags[i]])
      if correct_id >= len(toks):
        correct_id = len(toks)-1
      return self.remove_spaces([" ".join(toks[:correct_id])])[0], toks[correct_id]
    else:
      return None, None

  def predict_next_word_batched(self, batch):
    """
    This method predicts the next token for a batch.

    @params
      batch: batch of sentences
    """
    self.total += len(batch["sentence"])
    for i in range(len(batch["sentence"])):
      prompt, correct_tok = self.get_correct_toks(batch["tags"][i], batch["tokenized"][i])

      if prompt != None and correct_tok != None:
        out_tensor = self.model.generate(prompt, max_new_tokens = 1, verbose = False, return_type = "tokens") # greedy decoding
        pred = self.model.to_str_tokens(out_tensor)[-1].strip()

        if pred == correct_tok: # exact string match
          self.num_correct += 1

          if len(self.correct_answers) < 5: # examples of correct answers
              self.correct_answers.append(self.model.to_string(out_tensor)[0])
        else:
          if len(self.incorrect_answers) < 5: # examples of incorrect answers
            self.incorrect_answers.append((prompt + " -> " + correct_tok, self.model.to_string(out_tensor)[0]))

        del out_tensor
        t.cuda.empty_cache()

  def compute_loss_batched(self, batch, ckp_file: str):
    """
    This function computes the loss for a batch of sentences.

    @params
      batch: batch of sentences
      ckp_file: file for the intermediate results
    """
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
    """
    Sanity check for the experiment outputs.

    @returns output: experiment results
    """
    output = f"\nThe model generated {self.num_correct} correct answers.\n"
    output += "\nAccuracy: "
    output += "{:.2%}".format(self.num_correct/self.total)
    output += f"\n\nFive correct outputs: {self.correct_answers}"
    output += f"\n\nFive incorrect outputs: {self.incorrect_answers}"
    
    print(output)
    return output


