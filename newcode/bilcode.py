import numpy as np

class BilMaxent(object):
    def __init__(self, encoding, weight):
        self.encoding = encoding
        self.weight = weight

    def set_weight(self, new_weight):
        self.weight = new_weight

    def get_grad(self):
        return self.grad

    def get_nlogl(self):
        return self.nlogl

    def gradients(self):

        emb_v, emb_n, bil_inn, meanvb, meanbn = self.encoding.ext_emps()
        est_x = np.matrix(np.zeros(self.encoding.shape(), 'd'))

        tot_samples = len(self.encoding.train_toks())
        ll = []
        correct = 0

        for tok, label in self.encoding.train_toks():

            score = 0

            v, n, m = self.encoding.bil_encode([(tok, label)])
            featureset = word_features(tok)

            v = np.matrix(v).transpose()
            n = np.matrix(n).transpose()
            m = np.matrix(m).transpose()

            score = np.trace(self.weight*(z*(x-y).transpose()))

