#!/usr/bin/python

"""
Analize the result from main_svm.py.

Plot figures to compare various error estimators.
"""

PERCENTILE = 1.0/2
from pylab import *
import numpy as np
from scipy.stats import beta
ds = [5, 10, 15, 20]
dt = 0.3
stds = [0.5, 1.0, 2.0]
def main():
    for std in stds:
        biasResubs = []
        biasCV10s = []
        biasBS632s = []
        biasBRnews = []
        biasBRolds = []
        biasBRnew2sigs = []

        varResubs = []
        varCV10s = []
        varBS632s = []
        varBRnews = []
        varBRolds = []
        varBRnew2sigs = []

        rmsResubs = []
        rmsCV10s = []
        rmsBS632s = []
        rmsBRnews = []
        rmsBRolds = []
        rmsBRnew2sigs = []

        errTrueMeans = []

        for d in ds:
            fname = "svm_dt%d_ttest_d%d_std_%.1f.txt" % ((dt*10),d,std)
            path = './SVM_std' + str(std) + '_all/'
            data = np.loadtxt(fname, skiprows=1)
            # true    resub   cv10    loo     bs0     bs632   brNEWD  brOLD   bs_br_mixed     brNEW2sig
            # 0       1       2       3       4       5       6       7       8               9
            # extract error rates of interest
            errTrue = data[:, 0]
            errResub = data[:, 1]
            errCV10 = data[:, 2]
            errBS632 = data[:, 5]
            errBRnew = data[:, 6]
            errBRold = data[:, 7]
            errBRnew2sig = data[:, 9]
            # calculate difference various error rates from true error
            diffResub = errResub - errTrue
            diffCV10 = errCV10 - errTrue
            diffBS632 = errBS632 - errTrue
            diffBRnew = errBRnew - errTrue
            diffBRold = errBRold - errTrue
            diffBRnew2sig = errBRnew2sig - errTrue
            # calculate bias, variance and root mean square(rms) of various error rates
            biasResub = np.mean(diffResub)
            biasCV10 = np.mean(diffCV10)
            biasBS632 = np.mean(diffBS632)
            biasBRnew = np.mean(diffBRnew)
            biasBRold = np.mean(diffBRold)
            biasBRnew2sig = np.mean(diffBRnew2sig)

            varResub = np.var(diffResub)
            varCV10 = np.var(diffCV10)
            varBS632 = np.var(diffBS632)
            varBRnew = np.var(diffBRnew)
            varBRold = np.var(diffBRold)
            varBRnew2sig = np.var(diffBRnew2sig)

            rmsResub = np.sqrt(np.mean(diffResub**2))
            rmsCV10 = np.sqrt(np.mean(diffCV10**2))
            rmsBS632 = np.sqrt(np.mean(diffBS632**2))
            rmsBRnew = np.sqrt(np.mean(diffBRnew**2))
            rmsBRold = np.sqrt(np.mean(diffBRold**2))
            rmsBRnew2sig = np.sqrt(np.mean(diffBRnew2sig**2))
            # print True

            # boxplots
            figure()
            boxplot([diffResub, diffCV10, diffBRnew, diffBRold, diffBS632, diffBRnew2sig])
            axhline(0, color = 'r')
            names = ['Besub','CV10', 'BRnew', 'BRold', 'BS632', 'BRnew2sig']
            xticks(range(1, 7), names)
            title('Box plot of error estimator, std=%s, d=%s'%(str(std), str(d)))
            xlabel('Error estimators')
            ylabel('Box plot')
            fgrName = path + 'box_d' + str(d) + '.png'
            savefig(fgrName, bbox_inches = 0)
            # show()

            # beta distribution fit

            l = -.5
            r = 0.5
            dd = r - l
            xAxis = arange(l, r, 0.01)
            yscale = 0.01

            paraResub = beta.fit(diffResub, floc = l, fscale = dd)
            betaResub = yscale * beta.pdf(xAxis, paraResub[0], paraResub[1], loc = l, fscale = dd)
            paraCV10 = beta.fit(diffCV10, floc = l, fscale = dd)
            betaCV10 = yscale * beta.pdf(xAxis, paraCV10[0], paraCV10[1], loc = l, fscale = dd)
            paraBS632 = beta.fit(diffBS632, floc = l, fscale = dd)
            betaBS632 = yscale * beta.pdf(xAxis, paraBS632[0], paraBS632[1], loc = l, fscale = dd)
            paraBRnew = beta.fit(diffBRnew, floc = l, fscale = dd)
            betaBRnew = yscale * beta.pdf(xAxis, paraBRnew[0], paraBRnew[1], loc = l, fscale = dd)
            paraBRold = beta.fit(diffBRold, floc = l, fscale = dd)
            betaBRold = yscale * beta.pdf(xAxis, paraBRold[0], paraBRold[1], loc = l, fscale = dd)
            paraBRnew2sig = beta.fit(diffBRnew2sig, floc = l, fscale = dd)
            betaBRnew2sig = yscale * beta.pdf(xAxis, paraBRnew2sig[0], paraBRnew2sig[1], loc = l, fscale = dd)

            figure()
            plot(xAxis, betaResub, label = 'Resub', linewidth = 2.0)
            plot(xAxis, betaCV10, label = 'CV10', linewidth = 2.0)

            plot(xAxis, betaBRnew, label = 'BRnew', linewidth = 2.0)
            plot(xAxis, betaBRold, label = 'BRold', linewidth = 2.0)
            plot(xAxis, betaBS632, label = 'BS632', linewidth = 2.0)
            plot(xAxis, betaBRnew2sig, label = 'BRnew2sig', linewidth = 2.0)
            axvline(0, color = 'r')
            # axis('tight')
            legend()
            title('Beta fit of error estimator, std=%s, d=%s'%(str(std), str(d)))
            xlabel('x (-0.5, 0.5)')
            ylabel('Beta probability density function (pdf)')
            fgrName = path + 'betafit_d' + str(d) + '.png'
            savefig(fgrName, bbox_inches = 0)
            # show()

            biasResubs.append(biasResub)
            biasCV10s.append(biasCV10)
            biasBS632s.append(biasBS632)
            biasBRnews.append(biasBRnew)
            biasBRolds.append(biasBRold)
            biasBRnew2sigs.append(biasBRnew2sig)

            varResubs.append(varResub)
            varCV10s.append(varCV10)
            varBS632s.append(varBS632)
            varBRnews.append(varBRnew)
            varBRolds.append(varBRold)
            varBRnew2sigs.append(varBRnew2sig)

            rmsResubs.append(rmsResub)
            rmsCV10s.append(rmsCV10)
            rmsBS632s.append(rmsBS632)
            rmsBRnews.append(rmsBRnew)
            rmsBRolds.append(rmsBRold)
            rmsBRnew2sigs.append(rmsBRnew2sig)

            errTrueMeans.append(np.mean(errTrue))

        figure()
        xAxis = array(ds)
        plot(xAxis, biasResubs, label = 'Resub', linewidth = 2.0)
        plot(xAxis, biasCV10s, label = 'CV10', linewidth = 2.0)

        plot(xAxis, biasBRnews, label = 'BRnew', linewidth = 2.0)
        plot(xAxis, biasBRolds, label = 'BRold', linewidth = 2.0)
        plot(xAxis, biasBS632s, label = 'BS632', linewidth = 2.0)
        plot(xAxis, biasBRnew2sigs, label = 'BRnew2sig', linewidth = 2.0)
        axhline(0, color = 'r')
        # axis('tight')
        legend()
        title('Bias of error estimator, std=%s'%(str(std)))
        xlabel('Selected feature size d')
        ylabel('Bias')
        fgrName = path + 'bias' + '.png'
        savefig(fgrName, bbox_inches = 0)
        # show()

        figure()
        xAxis = array(ds)
        plot(xAxis, varResubs, label = 'Resub', linewidth = 2.0)
        plot(xAxis, varCV10s, label = 'CV10', linewidth = 2.0)

        plot(xAxis, varBRnews, label = 'BRnew', linewidth = 2.0)
        plot(xAxis, varBRolds, label = 'BRold', linewidth = 2.0)
        plot(xAxis, varBS632s, label = 'BS632', linewidth = 2.0)
        plot(xAxis, varBRnew2sigs, label = 'BRnew2sig', linewidth = 2.0)
        # axhline(0, color = 'r')
        # axis('tight')
        legend()
        title('Var of error estimator, std=%s'%(str(std)))
        xlabel('Selected feature size d')
        ylabel('Variance (var)')
        fgrName = path + 'var' + '.png'
        savefig(fgrName, bbox_inches = 0)
        # show()
        figure()
        xAxis = array(ds)
        plot(xAxis, rmsResubs, label = 'Resub', linewidth = 2.0)
        plot(xAxis, rmsCV10s, label = 'CV10', linewidth = 2.0)

        plot(xAxis, rmsBRnews, label = 'BRnew', linewidth = 2.0)
        plot(xAxis, rmsBRolds, label = 'BRold', linewidth = 2.0)
        plot(xAxis, rmsBS632s, label = 'BS632', linewidth = 2.0)
        plot(xAxis, rmsBRnew2sigs, label = 'BRnew2sig', linewidth = 2.0)
        # axhline(0, color = 'r')
        # axis('tight')
        legend()
        title('Rms of error estimator, std=%s'%(str(std)))
        xlabel('Selected feature size d')
        ylabel('Root mean square (rms)')
        fgrName = path + 'rms' + '.png'
        savefig(fgrName, bbox_inches = 0)
        # show()

        figure()
        xAxis = array(ds)
        plot(xAxis, errTrueMeans, linewidth = 2.0)
        # axhline(0, color = 'r')
        # axis('tight')
        # legend()
        title('True errors, std=%s'%(str(std)))
        xlabel('Selected feature size d')
        ylabel('Error')
        fgrName = path + 'error' + '.png'
        savefig(fgrName, bbox_inches = 0)
        # show()
if __name__ == '__main__':
    print __doc__
    main()
