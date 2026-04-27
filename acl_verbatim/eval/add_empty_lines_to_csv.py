import csv
import sys

infile, outfile = sys.argv[1:3]
k = int(sys.argv[3])

with open(infile) as in_f:
    reader = csv.reader(in_f, delimiter=",")
    with open(outfile, "wt") as out_f:
        writer = csv.writer(out_f, delimiter=",")
        for i, row in enumerate(reader):
            writer.writerow(row)
            if (i + 1) % k == 0:
                writer.writerow([])
