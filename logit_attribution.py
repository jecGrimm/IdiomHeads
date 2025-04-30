import torch as t
import os
import json
from transformer_lens.utils import Slice
from merge_tokenizers import PythonGreedyCoverageAligner, types

class LogitAttribution:
    def __init__(self, model, filename = "pythia_formal_idiom_pos.json"):
        self.model = model
        self.model.cfg.use_attn_result = True
        if self.model.cfg.normalization_type in ["LN", "LNPre", "RMS", "RMSPre"]:
            print("\nModel uses LayerNorm!\n")

        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.token_attr = None
        self.split_attr = None
        self.idiom_positions = self.load_all_idiom_pos(filename)
        self.labels = None
        self.cache = None
        self.residual_stack = None
        self.aligner = PythonGreedyCoverageAligner()

    def load_all_idiom_pos(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'r', encoding = "utf-8") as f:
                return json.load(f) 
        else:
            return [] 
        
    def get_all_idiom_pos(self, batch):
        for i in range(len(batch["sentence"])):
            sent = batch["sentence"][i]
            model_str_tokens= self.model.to_str_tokens(sent)
            aligned_positions = self.align_tokens(sent, batch["tokenized"][i], model_str_tokens)
            self.idiom_positions.append(self.get_idiom_pos(aligned_positions, batch["tags"][i]))

    def align_tokens(self, sent: str, tokenized_sent: list, model_str_tokens: list):
        aligned = self.aligner.align(
            types.TokenizedSet(tokens=[tokenized_sent, model_str_tokens], text=sent)
        )
        return list(aligned[0])
    
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

    def get_cache(self, sent):
        _, self.cache = self.model.run_with_cache(sent, remove_batch_dim=True)
        self.cache.to(self.device)

    def split_logit_attribution(self, logit_attr, idiom_pos):
        idiom_attr = self.mean_idiom_attribution(logit_attr, idiom_pos)
        literal_attr = self.mean_literal_attribution(logit_attr, idiom_pos)
        return t.vstack((idiom_attr, literal_attr))

    def mean_idiom_attribution(self, logit_attr, idiom_pos):
        return t.mean(logit_attr[idiom_pos[0]:idiom_pos[1]], dim = 0)

    def mean_literal_attribution(self, logit_attr, idiom_pos):
        return t.mean(t.cat((logit_attr[:idiom_pos[0]], logit_attr[idiom_pos[1]+1:])), dim = 0)
    
    def compute_logit_attr(self, sent):
        if self.cache == None:
            self.get_cache(sent)
        tokens = self.model.to_tokens(sent)

        with t.inference_mode():
            if self.residual_stack == None:
                self.residual_stack = self.cache.get_full_resid_decomposition(expand_neurons=False, return_labels=False)
            logit_attr = self.logit_attrs(tokens, has_batch_dim=False)

            self.residual_stack = None
            self.cache = None
            del tokens
            t.cuda.empty_cache()

            return t.einsum("ij->ji", logit_attr)
        
    def compute_logit_attr_batched(self, batch, split_file):
        if self.labels == None:
            self.get_labels(batch["sentence"][0])
        
        batch_split_attr = t.zeros(len(batch["sentence"]), 2, len(self.labels), device=self.device)

        for i in range(len(batch["sentence"])):
            sent_score = self.compute_logit_attr(batch["sentence"][i])
            
            idiom_pos = batch["idiom_pos"][i]
            if batch["idiom_pos"][i][1] >= sent_score.size(0): 
                idiom_pos = [batch["idiom_pos"][i][0], sent_score.size(0)-1]
            batch_split_attr[i] = self.split_logit_attribution(sent_score, idiom_pos)

            del sent_score
            del idiom_pos
            t.cuda.empty_cache()
        
        if self.split_attr != None:
            self.split_attr = t.cat((self.split_attr, batch_split_attr), dim = 0)
        else:
            self.split_attr = batch_split_attr  
        t.save(self.split_attr, split_file) 

        del batch_split_attr
        t.cuda.empty_cache()

    def get_labels(self, sent):
        self.get_cache(sent)

        with t.inference_mode():
            self.residual_stack, self.labels = self.cache.get_full_resid_decomposition(expand_neurons=False, return_labels=True)
        print(f"\nComputing logit attribution for the following components:\n{self.labels}")
    
    def logit_attrs(
        self,
        tokens,
        incorrect_tokens = None,
        pos_slice = None,
        batch_slice = None,
        has_batch_dim: bool = True,
    ):
        """
        AUS TRANSFORMERLENS: https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/ActivationCache.py#L561
        Logit Attributions.

        Takes a residual stack (typically the residual stream decomposed by components), and
        calculates how much each item in the stack "contributes" to specific tokens.

        It does this by:
            1. Getting the residual directions of the tokens (i.e. reversing the unembed)
            2. Taking the dot product of each item in the residual stack, with the token residual
               directions.

        Note that if incorrect tokens are provided, it instead takes the difference between the
        correct and incorrect tokens (to calculate the residual directions). This is useful as
        sometimes we want to know e.g. which components are most responsible for selecting the
        correct token rather than an incorrect one. For example in the `Interpretability in the Wild
        paper <https://arxiv.org/abs/2211.00593>` prompts such as "John and Mary went to the shops,
        John gave a bag to" were investigated, and it was therefore useful to calculate attribution
        for the :math:`\\text{Mary} - \\text{John}` residual direction.

        Warning:

        Choosing the correct `tokens` and `incorrect_tokens` is both important and difficult. When
        investigating specific components it's also useful to look at it's impact on all tokens
        (i.e. :math:`\\text{final_ln}(\\text{residual_stack_item}) W_U`).

        Args:
            residual_stack:
                Stack of components of residual stream to get logit attributions for.
            tokens:
                Tokens to compute logit attributions on.
            incorrect_tokens:
                If provided, compute attributions on logit difference between tokens and
                incorrect_tokens. Must have the same shape as tokens.
            pos_slice:
                The slice to apply layer norm scaling on. Defaults to None, do nothing.
            batch_slice:
                The slice to take on the batch dimension during layer norm scaling. Defaults to
                None, do nothing.
            has_batch_dim:
                Whether residual_stack has a batch dimension. Defaults to True.

        Returns:
            A tensor of the logit attributions or logit difference attributions if incorrect_tokens
            was provided.
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        if not isinstance(batch_slice, Slice):
            batch_slice = Slice(batch_slice)

        if isinstance(tokens, str):
            tokens = t.as_tensor(self.model.to_single_token(tokens))

        elif isinstance(tokens, int):
            tokens = t.as_tensor(tokens)

        logit_directions = self.model.tokens_to_residual_directions(tokens)

        if incorrect_tokens is not None:
            if isinstance(incorrect_tokens, str):
                incorrect_tokens = t.as_tensor(self.model.to_single_token(incorrect_tokens))

            elif isinstance(incorrect_tokens, int):
                incorrect_tokens = t.as_tensor(incorrect_tokens)

            if tokens.shape != incorrect_tokens.shape:
                raise ValueError(
                    f"tokens and incorrect_tokens must have the same shape! \
                        (tokens.shape={tokens.shape}, \
                        incorrect_tokens.shape={incorrect_tokens.shape})"
                )

            # If incorrect_tokens was provided, take the logit difference
            logit_directions = logit_directions - self.model.tokens_to_residual_directions(
                incorrect_tokens
            )

        scaled_residual_stack = self.cache.apply_ln_to_stack(
            self.residual_stack,
            layer=-1,
            pos_slice=pos_slice,
            batch_slice=batch_slice,
            has_batch_dim=has_batch_dim,
        ) 

        # Element-wise multiplication and sum over the d_model dimension
        #logit_attrs = (residual_stack * logit_directions).sum(dim=-1)
        logit_attrs = (scaled_residual_stack * logit_directions).sum(dim=-1)
        return logit_attrs
        
    def explore_tensor(self):
        print(f"The grouped attribution of the first sentence for {self.labels[0]} is:\n{self.split_attr[0, :, 0]}")

if __name__ == "__main__":
    loaded_tensor = t.load("./scores/logit_attribution/pythia-14m/grouped_attr_formal_0_3.pt", map_location=t.device("cpu"))
    print(f"Loaded tensor with size: {loaded_tensor.size()}")