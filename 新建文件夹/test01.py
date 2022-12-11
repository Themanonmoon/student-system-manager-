import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])

    return theta