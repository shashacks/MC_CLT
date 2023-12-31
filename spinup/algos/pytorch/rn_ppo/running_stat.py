import numpy as np

# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self):
        self._n = 0
        self._M = 0
        self._S = 0
    def push(self, x):
        self._n += 1
        if self._n == 1:
            self._M = x
        else:
            oldM = self._M.copy()
            self._M = oldM + (x - oldM)/self._n
            self._S = self._S + (x - oldM)*(x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

def test_running_stat():
    for shp in ((), (3,), (3,4)):
        li = []
        rs = RunningStat(shp)
        for _ in range(5):
            val = np.random.randn(*shp)
            rs.push(val)
            li.append(val)
            m = np.mean(li, axis=0)
            assert np.allclose(rs.mean, m)
            v = np.square(m) if (len(li) == 1) else np.var(li, ddof=1, axis=0)
            assert np.allclose(rs.var, v)