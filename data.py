import re 
from datasets import Dataset

class EPIE_Data:
    def __init__(self, experiment: str = "", split: str = "", model_id: str = ""):
        """
        This class preprocesses the Labelled EPIE dataset.


        """
        formal_idioms_labels = self.read_file("./EPIE_Corpus/Formal_Idioms_Corpus/Formal_Idioms_Labels.txt")
        translated_sentences = self.read_file("./EPIE_Corpus/Formal_Idioms_Corpus/translated_sentences.txt")
        formal_idioms_words = self.read_file("./EPIE_Corpus/Formal_Idioms_Corpus/Formal_Idioms_Words.txt")
        formal_idioms_tags = self.read_file("./EPIE_Corpus/Formal_Idioms_Corpus/Formal_Idioms_Tags.txt")
        static_idioms_words = self.read_file("./EPIE_Corpus/Static_Idioms_Corpus/Static_Idioms_Words.txt")
        static_idioms_tags = self.read_file("./EPIE_Corpus/Static_Idioms_Corpus/Static_Idioms_Tags.txt")

        self.split_idiom_from_literal(formal_idioms_words, translated_sentences, formal_idioms_tags, formal_idioms_labels)
        self.tokenized_static_sents = [static_idiom.strip() for static_idiom in static_idioms_words]
        self.tags_static = [tag.strip() for tag in static_idioms_tags]

        self.formal_sents = self.remove_spaces(self.tokenized_formal_sents)
        self.trans_formal_sents = self.remove_spaces(self.tokenized_trans_formal_sents)
        self.static_sents = self.remove_spaces(self.tokenized_static_sents)
        
        excluded_ids = []
        if experiment == "awareness":
            if model_id == "Llama-3.2-1B-Instruct":
                excluded_ids = [1231]
        elif experiment in ["idiom_score", "literal_score"]:
            excluded_ids = [1231]
        elif experiment == "dla":
            if model_id == "Pythia-1.4B":
                excluded_ids = [1231]
            elif model_id == "Llama-3.2-1B-Instruct":
                if split == "formal":
                    excluded_ids = [1231, 1280, 1281, 1282, 1386, 2210]
                elif split == "trans":
                    excluded_ids = [1231, 1280, 1281, 1282, 1379, 1386, 2200, 2201, 2210]
        elif experiment == "ablation":
            if model_id == "Pythia-1.4B":
                excluded_ids = [1231]
            elif model_id == "Llama-3.2-1B-Instruct":
                excluded_ids = [1231, 1280, 1281, 1282, 1379, 1386, 2200, 2201, 2210]
        
        for sent_idx in excluded_ids:
            self.formal_sents.pop(sent_idx)
            self.tokenized_formal_sents.pop(sent_idx)
            self.trans_formal_sents.pop(sent_idx)
            self.tokenized_trans_formal_sents.pop(sent_idx)
            self.tags_formal.pop(sent_idx)

        # # # 1773 und 1817 ab pythia idiom score
        # static_long_sent_ids = [1773, 1817]
        # for sent_idx in static_long_sent_ids:
        #     self.static_sents.pop(sent_idx)
        #     self.tokenized_static_sents.pop(sent_idx)
        #     self.tags_static.pop(sent_idx)

    def read_file(self, path: str):
        '''
        This method reads the files from the EPIE_Corpus and stores them in a list.

        @params
            path: path of the corpus file
        @returns
            lines of text in the file
        '''
        with open(path, 'r', encoding = "utf-8") as file:
            return file.readlines()

    def split_idiom_from_literal(self, formal_idioms_words: list, translated_sentences: list, formal_idioms_tags: list, formal_idioms_labels: list):
        '''
        This method splits the formal idiom sentences into occurences with a literal sense and with a figurative sense.

        @params
            formal_idioms_words: sentences of all occurences
            formal_idioms_tags: idiom tags for all occurences
            translated_sentences: translated sentences of all occurences
            formal_idioms_labels: sentence tags (1 = figurative, 0 = literal)
        '''
        self.tokenized_formal_sents = []
        self.tokenized_literal_sents = []
        self.tokenized_trans_formal_sents = []
        self.tokenized_trans_literal_sents = []
        self.tags_formal = []
        self.tags_literal = []
        for i in range(len(formal_idioms_labels)):
            if formal_idioms_labels[i].strip() == '1':
                self.tokenized_formal_sents.append(formal_idioms_words[i].strip())
                self.tokenized_trans_formal_sents.append(translated_sentences[i].strip())
                self.tags_formal.append(formal_idioms_tags[i].strip())
            else:
                self.tokenized_literal_sents.append(formal_idioms_words[i].strip())
                self.tokenized_trans_literal_sents.append(translated_sentences[i].strip())
                self.tags_literal.append(formal_idioms_tags[i].strip())

    def remove_spaces(self, sent_list: list):
        '''
        This method transforms the tokenized sentences into normal sentences.

        @params
            sent_list: tokenized sentences 
        @returns 
            cleaned_sents: list of the normal sentences
        '''
        cleaned_sents = []
        for sent in sent_list:
            space_matches = set(re.findall(r"(‘ | ['’\.,?!]| $)", sent))
            del_space_matches = {space_match:space_match.replace(' ', '') for space_match in space_matches}

            for space_match, del_space in del_space_matches.items():
                sent = sent.replace(space_match, del_space)
            cleaned_sents.append(sent)
        return cleaned_sents
    
    def get_idiom_toks(self, tokenized_sent: list, tags: list):
        '''
        This method extracts the idiom tokens of a sentence.

        @params 
            tokenized_sent: tokenized sentence
            tags: tags of the words 
        @returns 
            list of idiom tokens in the sentence
        '''
        return [tokenized_sent[i] for i in range(len(tokenized_sent)) if "IDIOM" in tags[i]]
    
    def create_hf_dataset(self, sents: list, tokenized_sents: list, tags: list):
        """
        This method transforms the processed dataset into a Hugging Face dataset.

        @params
            sents: list of sentences
            tokenized_sents: tokenized sentences
            tags: idiom labels
        @returns
            hf_dataset: Hugging Face dataset
        """
        data = {
            "sentence": sents,
            "tokenized": [sent.split(' ') for sent in tokenized_sents],
            "tags": [sent_tags.split(' ') for sent_tags in tags]
        }
        hf_dataset = Dataset.from_dict(data)
        return hf_dataset

    
if __name__ == "__main__":
    epie = EPIE_Data()
    print([sent for sent in epie.formal_sents if "spill" in sent.lower()])