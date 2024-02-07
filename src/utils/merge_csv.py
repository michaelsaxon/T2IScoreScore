import csv

import argparse

def main():
    parser = argparse.ArgumentParser(description='Take some source csv f1, add the last column of f2 to it and save result to output path')
    parser.add_argument('--f1', required=True, help='Input CSV file 1')
    parser.add_argument('--f2', required=True, help='Input CSV file 2')
    parser.add_argument('--o', required=True, help='output path')
    parser.add_argument('--no_headline', action='store_true', help='No titles in input 2')

    args = parser.parse_args()
    lines_1 = open(args.f1, "r").readlines()
    lines_2 = open(args.f2, "r").readlines()

    # if there's no headline in the second file, we add a blank title to its column and start from first line
    start_idx_2 = int(args.no_headline)


    print(len(lines_1))
    print(len(lines_2))

    for i in range(len(lines_1)):
        if args.no_headline and i == 0:
            lines_1[i] = lines_1[i].strip() + f",{args.f2}\n"
            continue
        #print(lines_1[i])
        last_elem_line_2 = next(csv.reader([lines_2[i - start_idx_2]]))[-1].strip()
        lines_1[i] = lines_1[i].strip() + f",{last_elem_line_2}\n"
        ##print(lines_1[i])

    with open(args.o, "w") as f:
        f.writelines(lines_1)

if __name__ == "__main__":
    main()