import collections
import csv
import glob
import os
import sys

import numpy as np
import tensorflow as tf


def load_events_file(path, step_factor=0.25):
    """Load an IMPALA TF event file and return x and y values."""
    entries = collections.defaultdict(lambda: ([], []))

    for event in tf.train.summary_iterator(path):
        for value in event.summary.value:
            if not value.tag.endswith("episode_return"):
                continue

            xs, ys = entries[value.tag]
            xs.append(event.step * step_factor)
            ys.append(value.simple_value)

    return entries


def mean_xs_ys(xs, ys):
    """Compute the mean y for non-unique x values."""
    res_xs = []
    res_ys = []

    n = 0
    cur_x = xs[0]
    sum_y = 0

    for x, y in zip(xs, ys):
        if x == cur_x:
            sum_y += y
            n += 1
            continue
        res_xs.append(cur_x)
        res_ys.append(sum_y / n)

        cur_x = x
        sum_y = y
        n = 1
    res_xs.append(cur_x)
    res_ys.append(sum_y / n)

    return res_xs, res_ys


def create_csvs(directories):
    for directory in directories:
        for path in glob.glob(directory + "/*/events.out.tfevents.*"):
            filename = os.path.join(os.path.dirname(path), "logs.csv")
            if os.path.exists(filename):
                print("Skipping", path, "as csv file already exists.")
                continue
            
            print("Loading", path)
            try:
                entries = load_events_file(path)
            except tf.errors.DataLossError as e:
                print(path + ": Error:", e)
                continue
            for name in entries.keys():
                if name.endswith("episode_return"):
                    xs, ys = entries[name]
                    break

            xs, ys = mean_xs_ys(xs, ys)

            with open(filename, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "mean_episode_return"])
                for row in zip(xs, ys):
                    writer.writerow(row)


def main():
    directories = sys.argv[1:]
    create_csvs(directories)


if __name__ == "__main__":
    main()
