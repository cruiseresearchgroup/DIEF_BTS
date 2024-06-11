import numpy as np
import copy

# def RSE(pred, true):
#     return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
#
#
# def CORR(pred, true):
#     u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
#     d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
#     return (u / d).mean(-1)
#
#
# def MAE(pred, true):
#     return np.mean(np.abs(pred - true))
#
#
# def MSE(pred, true):
#     return np.mean((pred - true) ** 2)
#
#
# def RMSE(pred, true):
#     return np.sqrt(MSE(pred, true))
#
#
# def MAPE(pred, true):
#     return np.mean(np.abs((pred - true) / true))
#
#
# def MSPE(pred, true):
#     return np.mean(np.square((pred - true) / true))
#
#
# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)
#
#     return mae, mse, rmse, mape, mspe


def MAE(y_pred, y_true):
    mae_err= []
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(y_true.shape[1]):
            y_true[:, i, :][y_true[:, i, :] < 1e-5] = 0
            y_pred[:, i, :][y_pred[:, i, :] < 1e-5] = 0
            mask = np.not_equal(y_true[:, i, :], 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)
            mae = np.abs(y_pred[:, i, :] - y_true[:, i, :])
            mae = np.nan_to_num(mae * mask)
            mae = np.mean(mae)
            mae_err.append(mae)
        return mae_err

def MSE(y_pred, y_true):
    mse_err= []
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(y_true.shape[1]):
            y_true[:, i, :][y_true[:, i, :] < 1e-5] = 0
            y_pred[:, i, :][y_pred[:, i, :] < 1e-5] = 0
            mask = np.not_equal(y_true[:, i, :], 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)
            mse = np.square(y_pred[:, i, :] - y_true[:, i, :])
            mse = np.nan_to_num(mse * mask)
            mse = np.mean(mse)
            mse_err.append(mse)
        return mse_err

def RMSE(y_pred, y_true):
    rmse_err = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(y_true.shape[1]):
            y_true[:, i, :][y_true[:, i, :] < 1e-5] = 0
            y_pred[:, i, :][y_pred[:, i, :] < 1e-5] = 0
            mask = np.not_equal(y_true[:, i, :], 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)
            rmse = np.square(y_pred[:, i, :] - y_true[:, i, :])
            rmse = np.nan_to_num(rmse * mask)
            rmse = np.sqrt(np.mean(rmse))
            rmse_err.append(rmse)
        return rmse_err

def SMAPE(y_pred, y_true):
    smape_err = []
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(y_true.shape[1]):
            y_true[:, i, :][y_true[:, i, :] < 1e-5] = 0
            y_pred[:, i, :][y_pred[:, i, :] < 1e-5] = 0
            mask = np.not_equal(y_true[:, i, :], 0)
            mask = mask.astype(np.float32)
            mask /= np.mean(mask)
            # smape = np.mean(np.abs(y_pred[:, i, :] - y_true[:, i, :]) / (np.abs(y_pred[:, i, :]) + np.abs(y_true[:, i, :]) + 1e-5)) * 100
            # smape = np.nan_to_num(smape * mask)
            # smape = np.mean(smape)
            smape = np.abs(y_pred[:, i, :] - y_true[:, i, :]) / (np.abs(y_pred[:, i, :]) + np.abs(y_true[:, i, :]) + 1e-5) * 100
            smape = np.nanmean(smape * mask)
            smape_err.append(smape)
        return smape_err

def r2score(y_pred, y_true):
    r2_err = []
    for i in range(y_true.shape[1]):
        gt = y_true[:, i, :]
        pred = y_pred[:, i, :]
        gt = np.ravel(gt)
        pred = np.ravel(pred)
        y_mean = np.mean(gt)
        ss_total = np.sum((gt - y_mean) ** 2)
        ss_residual = np.sum((gt - pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        r2_err.append(r_squared)
    return r2_err


def metric(pred, true):

    pred, true = copy.deepcopy(pred), copy.deepcopy(true)

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(pred, true)
    r2 = r2score(pred, true)

    print(r2)

    mae = np.mean(mae)
    mse = np.mean(mse)
    rmse = np.mean(rmse)
    smape = np.mean(smape)
    r2 = np.mean(r2)

    return mae, rmse, mse, smape, r2