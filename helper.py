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

def print_tensor_size(file):
    device = "cuda" if t.cuda.is_available() else "cpu"
    tensor1 = t.load(file, map_location=t.device(device))
    print("Size tensor1: ", tensor1.size())
    print("tensor1 nan: ", (t.isnan(tensor1) == True).nonzero())
    print("tensor1:", tensor1)

def get_logit_component(num, model_name):
    x_labels = {
        "pythia-14m": [["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3", "L2H0", "L2H1", "L2H2", "L2H3", "L3H0", "L3H1", "L3H2", "L3H3", "L4H0", "L4H1", "L4H2", "L4H3", "L5H0", "L5H1", "L5H2", "L5H3", "0_mlp_out", "1_mlp_out", "2_mlp_out", "3_mlp_out", "4_mlp_out", "5_mlp_out", "embed", "bias"]],
        "pythia-1.4b": [['L0H0', 'L0H1', 'L0H2', 'L0H3', 'L0H4', 'L0H5', 'L0H6', 'L0H7', 'L0H8', 'L0H9', 'L0H10', 'L0H11', 'L0H12', 'L0H13', 'L0H14', 'L0H15'], ['L1H0', 'L1H1', 'L1H2', 'L1H3', 'L1H4', 'L1H5', 'L1H6', 'L1H7', 'L1H8', 'L1H9', 'L1H10', 'L1H11', 'L1H12', 'L1H13', 'L1H14', 'L1H15'], ['L2H0', 'L2H1', 'L2H2', 'L2H3', 'L2H4', 'L2H5', 'L2H6', 'L2H7', 'L2H8', 'L2H9', 'L2H10', 'L2H11', 'L2H12', 'L2H13', 'L2H14', 'L2H15'], ['L3H0', 'L3H1', 'L3H2', 'L3H3', 'L3H4', 'L3H5', 'L3H6', 'L3H7', 'L3H8', 'L3H9', 'L3H10', 'L3H11', 'L3H12', 'L3H13', 'L3H14', 'L3H15'], ['L4H0', 'L4H1', 'L4H2', 'L4H3', 'L4H4', 'L4H5', 'L4H6', 'L4H7', 'L4H8', 'L4H9', 'L4H10', 'L4H11', 'L4H12', 'L4H13', 'L4H14', 'L4H15'], ['L5H0', 'L5H1', 'L5H2', 'L5H3', 'L5H4', 'L5H5', 'L5H6', 'L5H7', 'L5H8', 'L5H9', 'L5H10', 'L5H11', 'L5H12', 'L5H13', 'L5H14', 'L5H15'], ['L6H0', 'L6H1', 'L6H2', 'L6H3', 'L6H4', 'L6H5', 'L6H6', 'L6H7', 'L6H8', 'L6H9', 'L6H10', 'L6H11', 'L6H12', 'L6H13', 'L6H14', 'L6H15'], ['L7H0', 'L7H1', 'L7H2', 'L7H3', 'L7H4', 'L7H5', 'L7H6', 'L7H7', 'L7H8', 'L7H9', 'L7H10', 'L7H11', 'L7H12', 'L7H13', 'L7H14', 'L7H15'], ['L8H0', 'L8H1', 'L8H2', 'L8H3', 'L8H4', 'L8H5', 'L8H6', 'L8H7', 'L8H8', 'L8H9', 'L8H10', 'L8H11', 'L8H12', 'L8H13', 'L8H14', 'L8H15'], ['L9H0', 'L9H1', 'L9H2', 'L9H3', 'L9H4', 'L9H5', 'L9H6', 'L9H7', 'L9H8', 'L9H9', 'L9H10', 'L9H11', 'L9H12', 'L9H13', 'L9H14', 'L9H15'], ['L10H0', 'L10H1', 'L10H2', 'L10H3', 'L10H4', 'L10H5', 'L10H6', 'L10H7', 'L10H8', 'L10H9', 'L10H10', 'L10H11', 'L10H12', 'L10H13', 'L10H14', 'L10H15'], ['L11H0', 'L11H1', 'L11H2', 'L11H3', 'L11H4', 'L11H5', 'L11H6', 'L11H7', 'L11H8', 'L11H9', 'L11H10', 'L11H11', 'L11H12', 'L11H13', 'L11H14', 'L11H15'], ['L12H0', 'L12H1', 'L12H2', 'L12H3', 'L12H4', 'L12H5', 'L12H6', 'L12H7', 'L12H8', 'L12H9', 'L12H10', 'L12H11', 'L12H12', 'L12H13', 'L12H14', 'L12H15'], ['L13H0', 'L13H1', 'L13H2', 'L13H3', 'L13H4', 'L13H5', 'L13H6', 'L13H7', 'L13H8', 'L13H9', 'L13H10', 'L13H11', 'L13H12', 'L13H13', 'L13H14', 'L13H15'], ['L14H0', 'L14H1', 'L14H2', 'L14H3', 'L14H4', 'L14H5', 'L14H6', 'L14H7', 'L14H8', 'L14H9', 'L14H10', 'L14H11', 'L14H12', 'L14H13', 'L14H14', 'L14H15'], ['L15H0', 'L15H1', 'L15H2', 'L15H3', 'L15H4', 'L15H5', 'L15H6', 'L15H7', 'L15H8', 'L15H9', 'L15H10', 'L15H11', 'L15H12', 'L15H13', 'L15H14', 'L15H15'], ['L16H0', 'L16H1', 'L16H2', 'L16H3', 'L16H4', 'L16H5', 'L16H6', 'L16H7', 'L16H8', 'L16H9', 'L16H10', 'L16H11', 'L16H12', 'L16H13', 'L16H14', 'L16H15'], ['L17H0', 'L17H1', 'L17H2', 'L17H3', 'L17H4', 'L17H5', 'L17H6', 'L17H7', 'L17H8', 'L17H9', 'L17H10', 'L17H11', 'L17H12', 'L17H13', 'L17H14', 'L17H15'], ['L18H0', 'L18H1', 'L18H2', 'L18H3', 'L18H4', 'L18H5', 'L18H6', 'L18H7', 'L18H8', 'L18H9', 'L18H10', 'L18H11', 'L18H12', 'L18H13', 'L18H14', 'L18H15'], ['L19H0', 'L19H1', 'L19H2', 'L19H3', 'L19H4', 'L19H5', 'L19H6', 'L19H7', 'L19H8', 'L19H9', 'L19H10', 'L19H11', 'L19H12', 'L19H13', 'L19H14', 'L19H15'], ['L20H0', 'L20H1', 'L20H2', 'L20H3', 'L20H4', 'L20H5', 'L20H6', 'L20H7', 'L20H8', 'L20H9', 'L20H10', 'L20H11', 'L20H12', 'L20H13', 'L20H14', 'L20H15'], ['L21H0', 'L21H1', 'L21H2', 'L21H3', 'L21H4', 'L21H5', 'L21H6', 'L21H7', 'L21H8', 'L21H9', 'L21H10', 'L21H11', 'L21H12', 'L21H13', 'L21H14', 'L21H15'], ['L22H0', 'L22H1', 'L22H2', 'L22H3', 'L22H4', 'L22H5', 'L22H6', 'L22H7', 'L22H8', 'L22H9', 'L22H10', 'L22H11', 'L22H12', 'L22H13', 'L22H14', 'L22H15'], ['L23H0', 'L23H1', 'L23H2', 'L23H3', 'L23H4', 'L23H5', 'L23H6', 'L23H7', 'L23H8', 'L23H9', 'L23H10', 'L23H11', 'L23H12', 'L23H13', 'L23H14', 'L23H15'], ['0_mlp_out', '1_mlp_out', '2_mlp_out', '3_mlp_out', '4_mlp_out', '5_mlp_out', '6_mlp_out', '7_mlp_out', '8_mlp_out', '9_mlp_out', '10_mlp_out', '11_mlp_out', '12_mlp_out', '13_mlp_out', '14_mlp_out', '15_mlp_out', '16_mlp_out', '17_mlp_out', '18_mlp_out', '19_mlp_out', '20_mlp_out', '21_mlp_out', '22_mlp_out', '23_mlp_out'], ['embed', 'bias']]
    }
    start = -1
    for comp_group in x_labels[model_name]:
        for comp in comp_group:
            start += 1
            if num == start:
                print(f"Component with number {num}: {comp}")
                return comp

if __name__ == "__main__":
    #merge_tensors("./scores/idiom_components/pythia-1.4b/idiom_only_trans_0_1231_comp.pt", "./scores/idiom_components/pythia-1.4b/idiom_only_trans_1232_None_comp.pt", "./scores/idiom_scores/pythia-1.4b/idiom_only_trans_0_None.pt")

    #file1= "scores/logit_attribution/pythia-1.4b/grouped_attr_formal_0_None.pt"
    #file1 = "scores/logit_attribution/pythia-1.4b/grouped_attr_trans_0_None.pt"
    #file1 = "scores/literal_components/pythia-1.4b/literal_only_formal_0_None_comp.pt"
    #print_tensor_size(file1)

    get_logit_component(384, "pythia-1.4b")
