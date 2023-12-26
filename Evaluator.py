class Evaluator():
    """
    Used to evaluate a retrieval system regarding precision(@N), recall and f1-score.
    Call evalSingle to evaluate for one ground truth example. After doing the same for the entire ground truth, call
    finalEval to compute the averages over all examples.
    """
    def __init__(self, groundtruth, n):
        """
        Initiates a new Evaluator

        Parameters
        ----------
        groundtruth : dict : the collection of ground truth examples
        n : int : where to measure precision
        """
        self.groundtruth = groundtruth
        self.n = n

        self.totalRetrievals = 0
        self.totalTruePositives = 0
        self.totalTruePositivesAtN = 0
        self.totalTruth = 0


    def getTruth(self, title):
        return set(self.groundtruth[title])

    def truePositives(self, title, retrieved):
        truth = self.getTruth(title)
        tp = truth.intersection(set(retrieved))
        return len(tp)

    def truePositivesAtN(self, title, retrieved):
        truth = self.getTruth(title)
        return len(truth.intersection(set(retrieved[:self.n])))

    def precision(self, title, retrieved):
        tp = self.truePositives(title, retrieved)
        r = len(retrieved)
        if r > 0:
            return tp / r
        return 0

    def precisionAtN(self, title, retrieved):
        tp = self.truePositivesAtN(title, retrieved)
        return tp/self.n

    def recall(self, title, retrieved):
        tp = self.truePositives(title, retrieved)
        t = len(self.getTruth(title))
        if t > 0:
            return tp / t
        return 0

    def f1(self, precision, recall):
        if precision + recall > 0:
            return 2*(precision*recall)/(precision+recall)
        return 0

    def evalSingle(self, title, retrieved):
        self.totalRetrievals += len(retrieved)
        self.totalTruePositives += self.truePositives(title, retrieved)
        self.totalTruePositivesAtN += self.truePositivesAtN(title, retrieved)
        self.totalTruth += len(self.getTruth(title))

        precision = self.precision(title, retrieved)
        precisionAtN = self.precisionAtN(title, retrieved)
        recall = self.recall(title, retrieved)
        f1 = self.f1(precision, recall)

        return {"precision": precision,
                "precision@"+str(self.n): precisionAtN,
                "recall": recall,
                "f1": f1}

    def finalEval(self):
        precision = self.totalTruePositives/self.totalRetrievals
        recall = self.totalTruePositives/self.totalTruth
        precisionAtN = self.totalTruePositivesAtN/(self.n*len(self.groundtruth))
        f1 = self.f1(precision, recall)

        return {"precision": precision,
                "precision@"+str(self.n): precisionAtN,
                "recall": recall,
                "f1": f1}