import numpy as np

def ACC(eta, y, tau=0):
    yhat = (eta > tau).astype(int)
    acc = np.mean(yhat.flatten() == y)
    return acc