"""
connect4_loader
~~~~~~~~~~~~

A library to load connect4 positions.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import gzip
import re
import numpy as np

LINE = re.compile(r"(\d*) ((?:\S+ )*)(\d+)\s*", re.ASCII)
HEIGHT = 6
WIDTH  = 7
AREA   = WIDTH * HEIGHT
ALL_MULTIPLIER = np.ones((WIDTH,1))

def load_file(file, weak=False):
    data = []
    with gzip.open(file, 'rt', encoding='utf-8') as f:
        for line in f:
            match = LINE.fullmatch(line)
            if match:
                how, scores, id = match.groups()
                scores = scores.split()

                indices  = []
                multiplier = np.ones((WIDTH,1))
                min = 1000
                all = True
                for i,v in enumerate(scores):
                    if v == 'x':
                        multiplier[i][0] = 0.0
                        all = False
                    else:
                        v = int(v)
                        if v <= min:
                            if v < min:
                                min = v
                                indices = [i]
                            else:
                                indices.append(i)
                if all:
                    multiplier = ALL_MULTIPLIER

                if weak:
                    if min > 0:
                        indices = []
                    else:
                        if min < 0:
                            indices = []
                            for i,v in enumerate(scores):
                                if v != 'x':
                                    v = int(v)
                                    if v < 0:
                                        indices.append(i)

                y=[HEIGHT]*WIDTH
                board=np.zeros([HEIGHT,WIDTH])
                mover = 1.0
                for ch in how:
                    x = int(ch)-1
                    y[x] -= 1
                    board[y[x]][x] = mover
                    mover *= -1.0
                data.append((board, indices, multiplier, how, id))
    return data

def load_data(weak=False):
    return (load_file("../data/connect4_training.txt.gz", weak),
            load_file("../data/connect4_validation.txt.gz", weak),
            load_file("../data/connect4_test.txt.gz", weak))

def load_data_wrapper(weak=False):
    tr_d, va_d, te_d = load_data(weak)
    training_data  = [(np.reshape(board, (AREA, 1)),
                       vectorized_result(indices), multiplier)
                     for board, indices,multiplier,*ignore in tr_d]
    validation_data= [(np.reshape(board, (AREA, 1)), indices, multiplier)
                      for board, indices,multiplier,*ignore in va_d]
    test_data      = [(np.reshape(board, (AREA, 1)),
                       indices, multiplier, how, id)
                      for board, indices, multiplier, how, id in te_d]
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a size WIDTH vector with a 1.0 in the play
    position and zeroes elsewhere.  This is used to convert a column
    (0...6) into a corresponding desired output from the neural
    network."""
    e = np.zeros((WIDTH, 1))
    for i in j:
        e[i] = 1.0
    return e
