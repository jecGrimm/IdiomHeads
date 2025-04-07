pkgs = []
with open("./requirements.txt", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        if '@' not in line:
            pkgs.append(line)

with open("./conda_requirements.txt", 'w', encoding="utf-8") as f:
    f.write(''.join(pkgs))