#! /usr/bin/env python

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import numpy as np
import sys
import timeit
from matplotlib import pyplot as plt
from ffht import fht

from sparse_rptree import HD_x
from rnd_rot_kdtree import CC_x
from ff_kdtree import HGPHD_x


def matmul(dim, reps) :
    A = np.random.normal(size=[dim,dim])
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        c = np.dot(A, x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def FFTx(dim, reps) :
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        d = np.fft.fft(x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def rFFTx(dim, reps) :
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        d = np.fft.rfft(x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def IFFTFFTx(dim, reps) :
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        d = np.fft.ifft(np.fft.fft(x))
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def rIFFTFFTx(dim, reps) :
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        d = np.fft.irfft(np.fft.rfft(x), dim)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def CCx(dim, reps) :
    y = np.random.normal(size=dim)
    fft_y = np.fft.fft(y)
    D = np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        d = CC_x(D, fft_y, x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def Hx(dim, reps) :
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        d = fht(x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def HDx(dim, reps) :
    D = np.random.binomial(n=1, p=0.5, size=dim).astype(float) * 2.0 - 1.0
    pad_vec = np.array([])
    scale_factor = 1.0
    D /= scale_factor
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        x1 = HD_x(D, pad_vec, x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def FFx(dim, reps) :
    D = np.random.binomial(n=1, p=0.5, size=dim) * 2 - 1
    G = np.random.normal(size=dim)
    P_seed = np.random.randint(9999)
    pad_vec = np.array([])
    D_by_d = D.astype(float) / float(dim)
    t = 0
    for i in range(reps) :
        x = np.random.normal(size=dim)
        start = timeit.default_timer()
        x1 = HGPHD_x(D_by_d, pad_vec, P_seed, G, x)
        stop = timeit.default_timer()
        t += (stop - start)
    return t
# -- end function

def get_lists(tuple_list) :
    x1, x2 = [], []
    for a, b in tuple_list :
        x1.append(a)
        x2.append(b)
    return x1, x2
# -- end function

def plot_figures(results, fname, reps, tlog=False) :
    my_colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive'
    ]
    my_markers = 'o^sxdpv123'

    color_dict = {}
    marker_dict = {}
    idx = 0
    for k in results :
        if k is 'dims' : continue
        color_dict[k] = my_colors[idx]
        marker_dict[k] = my_markers[idx]
        idx += 1

    xvec = results['dims']
    for k in results :
        if k is 'dims' : continue
        yvec, yerr = get_lists(results[k])
        plt.errorbar(
            xvec,
            yvec,
            yerr=yerr,
            label=k,
            color=color_dict[k],
            fmt=marker_dict[k],
            capsize=2
        )
    plt.xlabel('Dimension (vector length)')
    ylabel_str = ('Computation for %i operations (in secs)' % reps)
    plt.ylabel(ylabel_str)
    plt.legend()
    plt.title('Computation comparison')
    plt.savefig(fname=(fname + '.png'))
    if tlog :
        plt.title('Computation comparison (log scale)')
        plt.xscale('log', basex=2)
        plt.yscale('log', basey=2)
        plt.savefig(fname=(fname + '.log.png'))
# -- end function

def compare_computations() :
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reps', help='Number of repetitions', type=int)
    parser.add_argument('-R', '--runs', help='Number of runs to average over', type=int)
    parser.add_argument(
        '-F',
        '--res_file',
        help='The file in which to save the results',
        type=str,
        default='test'
    )
    args = parser.parse_args()
    
    results = {
        'dims' : [ 16, 32, 64, 128, 256, 512, 1024 ],
        'MatMul' : [],
        'FFTx' : [],
        'IFFTFFTx' : [],
        'rFFTx' : [],
        'rIFFTFFTx' : [],
        'CCx' : [],
        'Hx' : [],
        'HDx' : [],
        'FFx' : []
    }

    func_list = [
        ( matmul, 'MatMul' ),
        ( FFTx, 'FFTx' ),
        ( IFFTFFTx, 'IFFTFFTx' ),
        ( rFFTx, 'rFFTx' ),
        ( rIFFTFFTx, 'rIFFTFFTx' ),
        ( CCx, 'CCx' ),
        ( Hx, 'Hx' ),
        ( HDx, 'HDx' ),
        ( FFx, 'FFx' )
    ]
    
    nreps = args.reps
    nruns = args.runs
    for dim in results['dims'] :
        print('Processing dim = %i' % dim)

        for f, fkey in func_list :
            #print('Processing %s now ..' % fkey)
            tlist = []
            for i in range(nruns) :
                t = f(dim, nreps)
                tlist.append(t)
            t_mean = np.mean(tlist)
            t_std = np.std(tlist)
            print('%s\t: %g +- %g' % (fkey, t_mean, t_std))
            results[fkey].append((t_mean, t_std))

    plot_figures(results, args.res_file, nreps, tlog=True)    
#-- end function

if __name__ == '__main__' :
    status = compare_computations()
    sys.exit(status)
# -- end function
