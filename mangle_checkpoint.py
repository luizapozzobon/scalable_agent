import os
import sys
import pickle

import numpy as np
import tensorflow as tf

logdir = "/tmp/agent"

np.set_printoptions(threshold=10)


def get_variables(logdir):
    variables = {}
    for name, shape in tf.contrib.framework.list_variables(logdir):
        variables[name] = tf.contrib.framework.load_variable(logdir, name)
    return variables


def main():
    variables = get_variables(logdir)

    mode = sys.argv[1]
    if mode.startswith("print"):
        num_entries = 0
        for name, var in variables.items():
            if "/RMSProp" in name:
                pass
            else:
                print(name, var.shape, np.sum(var))
                if mode == "print-all":
                    print(var)
                num_entries += int(np.prod(var.shape))
        print("Total number of entries:", num_entries)

    if mode == "save":
        with open("checkpoint.pkl", "wb") as f:
            pickle.dump(variables, f)

    if mode == "change":
        checkpoint = tf.train.get_checkpoint_state(logdir)
        with tf.Session() as sess:
            for name, var in variables.items():
                var.fill(42.0)
                tfvar = tf.Variable(var, name=name)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)


if __name__ == "__main__":
    main()
