#! /usr/bin/env python
#####
"""
1.0000  1.1000
2.0000  1.8000
3.0000  2.8900
4.0000  4.2300
5.0000  5.0200
6.0000  5.9700
7.0000  6.9800
8.0000  8.1200
9.0000  9.1200
sample_data.dat

"""




import numpy as np
from scipy.optimize import leastsq
from leastsqbound import leastsqbound


def func(p, x):
    """model data as y = m*x+b """
    m, b = p
    return m * np.array(x) + b


def err(p, y, x):
    return y - func(p, x)


# extract data
temp = np.genfromtxt("sample_data.dat")
x = temp[:, 0]
y = temp[:, 1]

# perform unbounded least squares fitting
p0 = [1.0, 0.0]
p, cov_x, infodic, mesg, ier = leastsq(err, p0, args=(y, x), full_output=True)

# print out results
print "Standard Least Squares fitting results:"
print "p:", p
print "cov_x:", cov_x
print "infodic['nfev']:", infodic['nfev']
print "infodic['fvec']:", infodic['fvec']
print "infodic['fjac']:", infodic['fjac']
print "infodic['ipvt']:", infodic['ipvt']
print "infodic['qtf']:", infodic['qtf']
print "mesg:", mesg
print "ier:", ier
print ""

# same as above using no bounds
p0 = [1.0, 0.0]
p, cov_x, infodic, mesg, ier = leastsqbound(err, p0, args=(y, x),
                                            full_output=True)

# print out results
print "Bounded Least Squares fitting with no bounds results:"
print "p:", p
print "cov_x:", cov_x
print "infodic['nfev']:", infodic['nfev']
print "infodic['fvec']:", infodic['fvec']
print "infodic['fjac']:", infodic['fjac']
print "infodic['ipvt']:", infodic['ipvt']
print "infodic['qtf']:", infodic['qtf']
print "mesg:", mesg
print "ier:", ier
print ""


# perform bounded least squares fitting
p0 = [1.0, 0.0]
bounds = [(0.0, 2.0), (-10.0, 10.0)]
p, cov_x, infodic, mesg, ier = leastsqbound(err, p0, args=(y, x),
                                            bounds = bounds, full_output=True)

# print out results
print "Bounded Least Squares fitting results:"
print "p:", p
print "cov_x:", cov_x
print "infodic['nfev']:", infodic['nfev']
print "infodic['fvec']:", infodic['fvec']
print "infodic['fjac']:", infodic['fjac']
print "infodic['ipvt']:", infodic['ipvt']
print "infodic['qtf']:", infodic['qtf']
print "mesg:", mesg
print "ier:", ier
