import click

import shutil

# convert the multiple choice outputs into a single choice using sentencebert from a csv output in Michael's style

def double_id(line):
    linesplit = line.split(",")
    id = linesplit[0]
    count = 0
    for elem in linesplit:
        if elem == id:
            count += 1
        if count > 1:
            return True
    return False

@click.command()
@click.option('--fix')
@click.option('--endcap', default="</s>\n")
def main(fix, endcap):
    shutil.copyfile(fix, fix + ".bak")

    with open(fix, "r") as f:
        fix_lines = f.readlines()

    # iterate through fix lines and reference lines
    i = 1
    while i < len(fix_lines):
        # if the endcap isn't at the end of the line, delete the /n and merge with next line until satisfied
        while not fix_lines[i].endswith(endcap):
            fix_lines[i] = fix_lines[i].strip("\n") + fix_lines[i+1]
            del fix_lines[i+1]
            if double_id(fix_lines[i]):
                print(fix_lines[i])
        i += 1
    
    with open(fix, "w") as f:
        f.writelines(fix_lines)


if __name__ == "__main__":
    main()
