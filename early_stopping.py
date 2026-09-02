class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, best=None, bad_evals=0):
        self.patience=patience; self.min_delta=min_delta
        self.best=float('inf') if best is None else best; self.bad_evals=bad_evals
    def update(self, value):
        if value < self.best - self.min_delta:
            self.best=value; self.bad_evals=0; return True, False
        self.bad_evals += 1
        return False, self.bad_evals >= self.patience
