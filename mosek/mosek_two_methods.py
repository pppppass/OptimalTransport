# ########################### 70 Characters ##########################

"""
This Python file solve optimal transport problem using interior point
method and simplex method by calling Mosek. The input data will be
given randomly. After executing this Python file, it will generate
two log files (logs for each method), and two txt files (solutions
through each method).
"""

from mosek import *
import numpy.random as rdm
import numpy as np
import math

def distance(n):
    """Returns the Euclidean distance matrix between pixels in two images,
    exactly a :math:`n^2 \times n^2` matrix.

    Arguments:
        n: An integer to tells the size of the image.
    """

    dmatrix = np.zeros([n * n, n * n])

    # TODO: perform optimization here

    for i1 in range(n):
        for j1 in range(n):
            for i2 in range(n):
                for j2 in range(n):
                    dmatrix[i1 * n + i2, j1 * n + j2] = math.sqrt(
                          (j1 - i1) ** 2
                        + (j2 - i2) ** 2
                    )

    return dmatrix

class StreamBuffer():
    """A class to buffer strings from a stream.

    Arguments:
        explicit_echo: A boolean to indicate whether explicit
        echo, namely outputs to stdout, should be sent.
    """

    def __init__(self, explicit_echo=False):
        self.buffer = ""
        self.explicit_echo = explicit_echo

    def streamprinter(self, text):
        """A virtual stream printer derived from the buffer.

        Arguments:
            text: Some messages in the stream, which is added
                to the buffer.
        """
        self.buffer += text
        if (self.explicit_echo):
            print(text, end="")

def calculate(m, n, c, mu, nu, opttype, printer):
    """Perform optimization by calling mosek.

    Arguments:
        opttype: Indicator to the optimization method, should be
            ``optimizertype.intpnt`` or ``optimizertype.free_simplex``.
        printer (function): A function to handle logs to be printed.
    """

    # TODO: Add necessary information about arguments
    with Env() as env:
        with env.Task(0,0) as task:
            task.putintparam(iparam.optimizer, opttype)
            task.set_Stream(streamtype.log, printer)

            bkc = [boundkey.fx] * (m + n)
            bkx = [boundkey.lo] * m * n
            blc = [i for i in mu] + [i for i in nu]
            buc = [i for i in mu] + [i for i in nu]
            blx = [0] * m * n
            bux = [0.0] * m * n
            asub = [[int(i / n), m + (i % n)] for i in range(m * n)]
            aval = [[1, 1] for i in range(m * n)]
            numvar = len(bkx)
            numcon = len(bkc)

            task.appendvars(numvar)
            task.appendcons(numcon)

            for j in range(numvar):
                task.putcj(j, c[j])
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                task.putacol(j, asub[j], aval[j])

            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            task.putobjsense(objsense.minimize)

            task.optimize()

            task.solutionsummary(streamtype.msg)
            xx = [0.] * numvar
            task.getxx(soltype.bas, xx)

            return xx

def routine(t, method, explicit_echo=False):
    """Using an given method to solve OT.

    Arguments:
        t: An integer indicate the size of the image.
        method: An given method, should be ``optimizertype.intpnt``
        or ``optimizertype.free_simplex``.
    """

    m = t ** 2
    n = t ** 2
    c1 = distance(t)
    c = c1.reshape(m * n)

    # Generate random value for mu, nu (mu, nu represent pixel values in each images)
    total = 10000
    mu1 = rdm.rand(m)
    nu1 = rdm.rand(n)
    mu = [i * total / sum(mu1) for i in mu1]
    nu = [i * total / sum(nu1) for i in nu1]

    log = StreamBuffer(explicit_echo)
    xx = calculate(m, n, c, mu, nu, method, log.streamprinter)

    return {"xx": xx, "log": log.buffer}

result_intpnt = routine(32, optimizertype.intpnt, explicit_echo=True)

np.savetxt("xx_intpnt.txt", result_intpnt["xx"])
with open("log_intpnt.txt", "w") as f:
    f.write(result_intpnt["log"])

result_simplex = routine(32, optimizertype.free_simplex, explicit_echo=True)

np.savetxt("xx_simplex.txt", result_simplex["xx"])
with open("log_simplex.txt", "w") as f:
    f.write(result_simplex["log"])
