import torch as t

def create_colab_requirements():
    pkgs = []
    with open("./requirements.txt", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if '@' not in line:
                pkgs.append(line)

    with open("./conda_requirements.txt", 'w', encoding="utf-8") as f:
        f.write(''.join(pkgs))

def merge_tensors(file1, file2, outfile):
    device = "cuda" if t.cuda.is_available() else "cpu"
    tensor1 = t.load(file1, map_location=t.device(device))
    tensor2 = t.load(file2, map_location=t.device(device))

    concat_tensor = t.cat((tensor1, tensor2))
    print("Size tensor1: ", tensor1.size())
    print("Size tensor2: ", tensor2.size())
    print("Size concat_tensor: ", concat_tensor.size())

    t.save(concat_tensor, outfile)

if __name__ == "__main__":
    #merge_tensors("./scores/idiom_components/pythia-1.4b/idiom_only_formal_0_1231_comp.pt", "./scores/idiom_components/pythia-1.4b/idiom_only_formal_1232_None_comp.pt", "./scores/idiom_scores/pythia-1.4b/idiom_only_formal_0_None.pt")

    file1 = "./scores/idiom_components/pythia-1.4b/idiom_only_trans_0_None_comp.pt"
    device = "cuda" if t.cuda.is_available() else "cpu"
    tensor1 = t.load(file1, map_location=t.device(device))
    print("Size tensor1: ", tensor1.size())