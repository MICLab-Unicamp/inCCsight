import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', nargs='*', dest='parents')

args = parser.parse_args()

time1 = args.parents[0]
time2 = args.parents[1]

def save(time1, time2):
    data = {"WatershedROQS": time1, "based_CNN": time2}
    with open('./src/data/time.json', 'w') as outfile:
        json.dump(data, outfile)

save(time1, time2)