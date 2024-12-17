import numpy as np
import ql
import math
import pandas as pd
def swapRates(tau, p, mat):

        tmax = mat[-1]

        ttemp = np.arange(1/12, tmax + 1/12, 1/12)
        ptemp = np.interp(ttemp, tau, p)

        dis = np.cumsum(ptemp)
        #dis = dis(:);

        # linear interpolation
        pmat = np.interp(mat, tau, p)

        index = (2 * mat).astype(int) - 1
        S = 100 * 2 * (1 - pmat) / dis[index]
        return S


def liborRates(tau, p, mat):
    pmat = np.interp(mat, tau, p)
    L = 100 * (1. / pmat - 1) / mat
    return L

def calibration_report(swaptions, vols_data):
    
    columns = ["Model Price", "Market Price", "Implied Vol", "Market Vol", "Rel Er\
    ror Price", "Rel Error Vols", "Expiry", "Maturity"]
    
    report_data = []
    cum_err = 0.0
    cum_err2 = 0.0
    
    for i, s in enumerate(swaptions):
        model_price = s.modelValue()
        market_vol = vols_data[i]['vol']
        black_price = s.blackPrice(market_vol)
        rel_error = model_price/black_price - 1.0
        implied_vol = s.impliedVolatility(model_price,
        1e-1, 50, 0, 1.00)
     
        rel_error2 = implied_vol/market_vol-1.0
        cum_err += rel_error*rel_error
        cum_err2 += rel_error2*rel_error2
        report_data.append((model_price, black_price, implied_vol, market_vol, rel_error, rel_error2, vols_data[i]['expiry'], vols_data[i]['maturity']))
        
    print("Cumulative Error Price: %7.5f" % math.sqrt(cum_err))
    print("Cumulative Error Vols : %7.5f" % math.sqrt(cum_err2))
    
    return pd.DataFrame(report_data,columns= columns, index=['']*len(report_data))

def cost_function_generator(model, helpers, norm=False):
    def cost_function(params):
        params_ = ql.Array(list(params));
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers];
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_function