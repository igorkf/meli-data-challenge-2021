import numpy as np


def ranked_probability_score(y_true, y_pred):
    '''
    Provided by the workshop: https://www.youtube.com/watch?v=WqIXnWHyMVA

    Input:
        y_true: np.array of shape 30
        y_pred np.array of shape 30
    '''
    
    diff_cum = y_true.cumsum(axis=1) - y_pred.cumsum(axis=1)
    diff_cum_squared = diff_cum ** 2

    return diff_cum_squared.sum(axis=1).mean()


def scoring_function(y_true, y_pred):
    '''
    Provided by the workshop: https://www.youtube.com/watch?v=WqIXnWHyMVA
    '''

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_one_hot = np.zeros_like(y_pred, dtype=float)
    y_true_one_hot[range(len(y_true)), y_true - 1] = 1
    
    return ranked_probability_score(y_true_one_hot, y_pred)
