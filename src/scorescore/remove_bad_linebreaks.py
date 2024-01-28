import argparse
import shutil

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

def main():
    parser = argparse.ArgumentParser(description='Convert multiple choice outputs into a single choice using SentenceBERT from a CSV output in Michael\'s style.')
    parser.add_argument('--fix', required=True, help='Path to the CSV file to be fixed')
    parser.add_argument('--endcap', default="</s>\n", help='String to be used as endcap (default: "</s>\n")')

    args = parser.parse_args()

    shutil.copyfile(args.fix, args.fix + ".bak")

    with open(args.fix, "r") as f:
        fix_lines = f.readlines()

    i = 1
    while i < len(fix_lines):
        while not fix_lines[i].endswith(args.endcap):
            if double_id(fix_lines[i].strip("\n") + "," + fix_lines[i+1]):
                print(fix_lines[i].strip("\n") + " " + fix_lines[i+1])
            fix_lines[i] = fix_lines[i].strip("\n") + " " + fix_lines[i+1]
            del fix_lines[i+1]
        i += 1

    with open(args.fix, "w") as f:
        f.writelines(fix_lines)

if __name__ == "__main__":
    main()
