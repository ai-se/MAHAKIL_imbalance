from __future__ import division
import sys

sys.dont_write_bytecode = True


class counter():
    def __init__(self, before, after, indx):
        self.indx = indx
        self.actual = before
        self.predicted = after
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        for a, b in zip(self.actual, self.predicted):
            if a == indx and b == indx:
                self.TP += 1
            elif a == b and a != indx:
                self.TN += 1
            elif a != indx and b == indx:
                self.FP += 1
            elif a == indx and b != indx:
                self.FN += 1
            elif a != indx and b != indx:
                pass

    def stats(self):
        try:
            # recall
            Sen = self.TP / (self.TP + self.FN)
            Spec = self.TN / (self.TN + self.FP)
            Prec = self.TP / (self.TP + self.FP)
            Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
            F = 2 * (Prec * Sen) / (Prec + Sen)
            F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
            F2 = 5 * (Prec * Sen) / (4 * Prec + Sen)
            G = 2 * Sen * Spec / (Sen + Spec)
            G1 = Sen * Spec / (Sen + Spec)
            pf = self.FP / (self.FP + self.TN)
            return Sen, pf, Spec, Prec, Acc, F, G, F2
        except ZeroDivisionError:
            return 0, 0, 0, 0, 0, 0, 0, 0


class ABCD():
    "Statistics Stuff, confusion matrix, all that jazz..."

    def __init__(self, before, after):
        self.actual = before
        self.predicted = after

    def __call__(self):
        uniques = set(self.actual)
        for u in list(uniques):
            yield counter(self.actual, self.predicted, indx=u)
