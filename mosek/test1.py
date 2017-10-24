import sys
import mosek
import numpy.random as rdm

inf = 0.0

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def calculate(m, n, c, mu, nu):
    with mosek.Env() as env:
        with env.Task(0,0) as task:
            task.set_Stream(mosek.streamtype.log,streamprinter)
            bkc = [mosek.boundkey.fx] * (m + n)
            bkx = [mosek.boundkey.lo] * m * n
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
            task.putobjsense(mosek.objsense.minimize)
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            xx = [0.] * numvar
            task.getxx(mosek.soltype.bas, xx)
            return xx

def main():
    #Give the size of the problem
    m = 1000
    n = 1000
    #Generate random value for c, mu, nu
    c1 = rdm.rand(m, n)
    c = c1.reshape(m * n)
    total = 10
    mu1 = rdm.rand(m)
    nu1 = rdm.rand(n)
    mu = [i * total / sum(mu1) for i in mu1]
    nu = [i * total / sum(nu1) for i in nu1]
    #Solve LP problem for optimal transport
    xx = calculate(m, n, c, mu, nu)
    #Write the solution into a TXT file
    file = open('solution.txt', 'w')
    for i in range(m * n):
        #print('x[' + str(int(i/n)) + '][' + str(i%n) + ']=' + str(xx[i]))
        file.write(str('%.2f' % xx[i])+" ")
        if (i + 1) % n == 0:
            file.write('\n')
    file.close()

if __name__ == "__main__":
    main()
