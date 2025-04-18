import argparse
class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', '-m', help='model to run the experiment with', default="EleutherAI/pythia-1.4b")
        parser.add_argument('--data', '-d', help='list of data split that should be processed', nargs='*', default=["formal"], type=str)
        parser.add_argument('--start', '-s', help='start index (inclusive)', default = 0, type=int)
        parser.add_argument('--end', '-e', help='end index (exclusive)', default = None)
        parser.add_argument('--batch_size', '-b', help='batch size', nargs='*', default = [None])

        self.full_model_name = parser.parse_args().model_name
        self.model_name = self.full_model_name.split('/')[-1]
        self.data_split = parser.parse_args().data
        self.start = parser.parse_args().start

        self.end = parser.parse_args().end
        if self.end:
            self.end = int(self.end)

        self.batch_sizes = parser.parse_args().batch_size
        while len(self.batch_sizes) < len(self.data_split):
            self.batch_sizes.append(self.batch_sizes[-1])

if __name__ == "__main__":
    cli = CLI()