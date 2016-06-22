import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.metrics.pairwise import pairwise_kernels, euclidean_distances

class ActivePreference():
    def __init__(self, bound):
        self._bound = np.array(bound)
        assert len(self._bound.shape) == 2
        assert self._bound.shape[1] == 2
        self._x = np.array([self._bound.mean(axis=1)])
        self._ndim = self._bound.shape[0]
        self._i = []
        self._j = []
        self._last_query = None
        self._sgm = 1.0

        self._kernel = RBF(self._x, gamma=10/self._ndim)

    def query(self):
        pass

    def answer(self, choice, query=None):

        if not query:
            query = self._last_query
        assert type(query) == list and len(query) == 2

        assert choice in [0,1]
        if choice == 0:
            p = query[0]
            q = query[1]
        else:
            p = query[1]
            q = query[0]

        if p not in self.x:
            self._x.append(p)

        if q not in self.x:
            self._x.append(q)

        self._i.append(self._x.index(p))
        self._j.append(self._x.index(q))

        self._kernel.X = self._x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = np.array(x)
        self._kernel.X = self._x
        self._maximize_posterior()

    @property
    def lng(self):
        return np.log(self._kernel.gamma)

    @lng.setter
    def lng(self, lng):
        self._kernel.gamma = np.exp(lng)

    def best(self):
        pass

    def _log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        zk = (f[np.array(self._i)]-f[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        return np.sum(np.log(norm.cdf(zk)))

    def _log_prior(self, f=None):
        if f is None: f = self._fMAP
        alpha = np.linalg.solve(np.linalg.cholesky(self._kernel()),f)
        return -0.5*np.sum(alpha**2)

    def _log_posterior(self, f=None):
        if f is None: f = self._fMAP
        return self._log_likelihood(f) + self._log_prior(f)

    def _d_log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        k = len(self._i)
        zk = (f[np.array(self._i)]-f[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        sik = np.zeros((len(f),k))
        sik[self._i,np.arange(k)] = 1/np.sqrt(2)*self._sgm
        sik[self._j,np.arange(k)] = -1/np.sqrt(2)*self._sgm
        return np.matmul(sik, norm.pdf(zk)/norm.cdf(zk))

    def _d_log_prior(self, f=None):
        if f is None: f = self._fMAP
        return -np.linalg.solve(self._kernel(),f)

    def _d_log_posterior(self, f=None):
        if f is None: f = self._fMAP
        return self._d_log_likelihood(f) + self._d_log_prior(f)

    def _dd_log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        k = len(self._i)
        zk = (f[np.array(self._i)]-f[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        sik = np.zeros((len(f),k))
        sik[self._i,np.arange(k)] =  1/np.sqrt(2)*self._sgm
        sik[self._j,np.arange(k)] = -1/np.sqrt(2)*self._sgm
        sijk = np.tensordot(sik,sik,axes=0)[:,np.arange(k),:,np.arange(k)].swapaxes(0,2)
        pdf = norm.pdf(zk)
        cdf = norm.cdf(zk)
        return np.tensordot(sijk, -pdf**2/cdf**2-zk*pdf/cdf,axes=([2],[0]))

    def _dd_log_prior(self, f=None):
        if f is None: f = self._fMAP
        return -np.linalg.inv(self._kernel())

    def _dd_log_posterior(self, f=None):
        if f is None: f = self._fMAP
        return self._dd_log_likelihood(f) + self._dd_log_prior(f)

    def _argmax_posterior(self):
        def obj(f):
            return f - np.matmul(self._kernel(), self._d_log_likelihood(f))

        def dobj(f):
            return np.identity(len(f)) -\
                np.matmul(self._kernel(), self._dd_log_likelihood(f))

        return sp.optimize.fsolve(obj, np.zeros(len(self._x)), fprime=dobj)

    def _maximize_posterior(self):
        self._fMAP = self._argmax_posterior()

    def _log_marginal_likelihood(self, lns, lng):
        self.lng = lng
        self._sgm = np.exp(lns)
        self._maximize_posterior()
        m = np.identity(len(self._x))+np.matmul(self._kernel(),-self._dd_log_likelihood())
        s = -self._log_posterior()
        t = -0.5*np.log(np.linalg.det(m))
        return s+t

    def _d_log_marginal_likelihood_lng(self, lns, lng):
        self.lng = lng
        self._sgm = np.exp(lns)
        self._maximize_posterior()
        alpha = np.linalg.solve(self._kernel(), self._fMAP)
        dg = self._kernel.d_gamma()
        lam = -self._dd_log_likelihood()
        m = np.linalg.solve(np.identity(len(self._x))+
                            np.matmul(self._kernel(),lam),
                            np.matmul(dg,lam))
        s = 0.5*np.exp(lng)*np.dot(alpha, np.matmul(dg,alpha))
        t = -0.5*np.exp(lng)*np.trace(m)
        return s+t

    def _d_log_marginal_likelihood_lns(self, lns, lng):
        self._kernel.gamma = np.exp(lng)
        self._sgm = np.exp(lns)
        self._maximize_posterior()
        zk = (self._fMAP[np.array(self._i)]-self._fMAP[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        return self._sgm*(np.sum(dzk_ds * pdf) - 0.5*np.trace())

    def _d_log_likelihood_lns(self):
        return

    def _argmax_marginal_likelihood(self):
        pass

    def _expected_improvement(self,x):
        pass

    def _argmax_expected_improvement(self):
        pass


class RBF():
    def __init__(self, X, Y=None, gamma=None):
        self._gamma = gamma
        self._X = X
        self._Y = Y
        self._K = self(self._X, self._Y, self._gamma)

    @property
    def X(self):
        return self._X
    @property
    def Y(self):
        return self._Y
    @property
    def gamma(self):
        return self._gamma

    @X.setter
    def X(self, X):
        self._K = self(X=X)
    @Y.setter
    def Y(self, Y):
        self._K = self(Y=Y)
    @X.setter
    def gamma(self, gamma):
        self._K = self(gamma=gamma)

    def __call__(self, X=None, Y=None, gamma=None):
        if X is None and Y is None and gamma is None: return self._K
        if X is not None: self._X = X
        if Y is not None: self._Y = Y
        if gamma is not None: self._gamma = gamma
        self._ssd = euclidean_distances(self._X,self._Y,squared=True)
        self._K = pairwise_kernels(self._X, self._Y, metric='rbf', gamma=self._gamma)
        return self._K

    def d_gamma(self, gamma=None):
        self._K = self(gamma=gamma)
        return -self._ssd*self._K
