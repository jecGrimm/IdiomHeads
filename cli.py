import argparse
class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', '-m', help='model to run the experiment with', default = "EleutherAI/pythia-1.4b")
        parser.add_argument('--data', '-d', help='list of data split that should be processed', nargs='*', default=["formal"], type=str)
        parser.add_argument('--start', '-s', help='start index (inclusive)', nargs='*', default = [0], type=int)
        parser.add_argument('--end', '-e', help='end index (exclusive)', nargs='*', default = [None])
        parser.add_argument('--batch_size', '-b', help='batch size', nargs='*', default = [None])
        parser.add_argument('--idiom_file', '-i', help='File with the idiom positions', default = "pythia_formal_idiom_pos.json")
        parser.add_argument('--ablation', '-a', help='Key of the ablation heads', default = "pythia-1.4b_formal")

        self.full_model_name = parser.parse_args().model_name
        self.model_name = self.full_model_name.split('/')[-1]
        self.data_split = parser.parse_args().data
        self.start = [int(start) for start in parser.parse_args().start]

        self.end = parser.parse_args().end
        for i in range(len(self.end)):
            if self.end[i] != "None" and self.end[i] != None:
                self.end[i] = int(self.end[i])
            else:
                self.end[i] = None

        while len(self.start) < len(self.data_split):
            self.start.append(self.start[-1])

        while len(self.end) < len(self.data_split):
            self.end.append(self.end[-1])

        self.batch_sizes = parser.parse_args().batch_size
        while len(self.batch_sizes) < len(self.data_split):
            self.batch_sizes.append(self.batch_sizes[-1])

        self.idiom_file = parser.parse_args().idiom_file
        self.ablation = parser.parse_args().ablation

if __name__ == "__main__":
    cli = CLI()