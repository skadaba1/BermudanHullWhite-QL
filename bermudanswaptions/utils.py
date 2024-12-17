import pandas as pd
import numpy as np

""" Generate calibration_report after HW fitting """
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
        
    #print("Cumulative Error Price: %7.5f" % math.sqrt(cum_err))
    #print("Cumulative Error Vols : %7.5f" % math.sqrt(cum_err2))
    
    return pd.DataFrame(report_data,columns= columns, index=['']*len(report_data))

# functions to compute optimal hedges
def function_A_vec(t, delta_S_hat, data_mat, reg_param):
    X_mat = data_mat[t, :, :]
    num_basis_funcs = X_mat.shape[1]

    this_dS = delta_S_hat.loc[:, t]
    hat_dS2 = np.array((this_dS ** 2)).reshape(-1, 1)
    A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)

    return A_mat

# Computes optimal hedges along with function_A_vec
def function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat, gamma, risk_lambda):
    tmp = Pi_hat.loc[:,t+1].values * delta_S_hat.loc[:, t].values
    X_mat = data_mat[t, :, :]  # matrix of dimension N_MC x num_basis
    B_vec = np.dot(X_mat.T, tmp)

    return B_vec

# Computes optimal Q-table via backward solve
def function_C_vec(t, data_mat, reg_param):
    X_mat = data_mat[t, :, :]
    num_basis_funcs = X_mat.shape[1]
    C_mat = np.dot(X_mat.T, X_mat) + reg_param * np.eye(num_basis_funcs)
    return C_mat

# Computes optimal Q-table via backward solve
def function_D_vec(t, Q, R, data_mat, gamma):
    X_mat = data_mat[t, :, :]
    D_vec = np.dot(X_mat.T, R.loc[:,t] + gamma * Q.loc[:, t+1])

    return D_vec

# Computes terminal payoff of swaption as positive part of swaption payoff (redundant)
def terminal_payoff(K, x): #so
    # ST   final stock price
    # K    strike - not used
    payoff = max(x,0)
    return payoff