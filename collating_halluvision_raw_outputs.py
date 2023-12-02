def fname_iterator(beginning):
    for start in range(0,2900,350):
        if start == 2800:
            end = ":"
        else:
            end = start + 350
        if start == 0:
            start = 1
        yield beginning + f"{start}-{end}.csv"

for fname in fname_iterator("a_fuyu_dsg.csv."):
    print(fname)


with open("a_mplug_dsg.csv","w") as f:
    outlines = []
    for fname in fname_iterator("a_mplug_dsg.csv."):
        outlines += open(fname,"r").readlines()
    f.writelines(outlines)

with open("a_mplug_tifa.csv","w") as f:
    outlines = []
    for fname in fname_iterator("a_mplug_tifa.csv."):
        outlines += open(fname,"r").readlines()
    f.writelines(outlines)