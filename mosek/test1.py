#This Python file solve optimal transport problem using interior point method and simplex method by calling Mosek. The input data will be given randomly.
#After executing this Python file, it will generate two log files (logs for each method), and two txt files (solutions through each method).

import sys
from mosek import *
import numpy.random as rdm
import numpy as np
import math

inf = 0.0

#define the Euclidean distance between two units of the image
def distance(n):
    dmatrix = np.zeros([n * n, n * n])
    for i1 in range(n):
        for j1 in range(n):
            for i2 in range(n):
                for j2 in range(n):
                    dmatrix[i1 * n + i2, j1 * n + j2] = math.sqrt((j1 - i1) ** 2 + (j2 - i2) ** 2)
    return dmatrix

def streamprinter(text):
    print(text, end="")

#call mosek 
def calculate(m, n, c, mu, nu, opttype):
    with Env() as env:
        with env.Task(0,0) as task:
            #Use optimization method free_simplex & intpnt & dual_simplex
            task.putintparam(iparam.optimizer, opttype)
            task.set_Stream(streamtype.log,streamprinter)
            bkc = [boundkey.fx] * (m + n)
            bkx = [boundkey.lo] * m * n
            blc = [i for i in mu] + [i for i in nu]
            buc = [i for i in mu] + [i for i in nu]
            blx = [0] * m * n
            bux = [+inf] * m * n
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

def main():
    #Give the size of the problem (the size of the image would be t*t)
    t = 32
    m = t ** 2
    n = t ** 2
    c1 = distance(t)
    c = c1.reshape(m * n)
    #Generate random value for mu, nu (mu, nu represent pixel values in each images)
    total = 10000
    mu1 = rdm.rand(m)
    nu1 = rdm.rand(n)
    mu = [i * total / sum(mu1) for i in mu1]
    nu = [i * total / sum(nu1) for i in nu1]

    #Solve LP problem for optimal transport using interior point method
    solution_intpnt_log = open('solution_intpnt.log', 'w')
    sys.stdout = solution_intpnt_log
    xx = calculate(m, n, c, mu, nu, optimizertype.intpnt)
    #Write the solution into a TXT file
    file1 = open('solution_intpnt.txt', 'w')
    for i in range(m * n):
        #print('x[' + str(int(i/n)) + '][' + str(i%n) + ']=' + str(xx[i]))
        file1.write(str('%.2f' % xx[i])+" ")
        if (i + 1) % n == 0:
            file1.write('\n')
    file1.close()
    solution_intpnt_log.close()

    #Solve LP problem for optimal transport using simplex method
    solution_simplex_log = open('solution_simplex.log', 'w')
    sys.stdout = solution_simplex_log
    xx = calculate(m, n, c, mu, nu, optimizertype.free_simplex)
    #Write the solution into a TXT file
    file2 = open('solution_simplex.txt', 'w')
    for i in range(m * n):
        #print('x[' + str(int(i/n)) + '][' + str(i%n) + ']=' + str(xx[i]))
        file2.write(str('%.2f' % xx[i])+" ")
        if (i + 1) % n == 0:
            file2.write('\n')
    file2.close()
    solution_simplex_log.close()

if __name__ == "__main__":
    main()
