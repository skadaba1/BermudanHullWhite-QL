import numpy as np
from utils import *
from time import time
from ts import TS
from tqdm import tqdm
import display
def european():
    ## multi- sample run with European swaption #
    glob = dict()
    forward_terms = [1/12, 3/12, 6/12, 1, 2, 5, 10]
    expiry_terms = [5, 10]
    results_model = np.zeros((len(forward_terms), len(expiry_terms)))

    N_MC = 8192;   
    calibration_path = 'bloomberg_calibration.csv'
    vols_path = 'vols_liquid.csv'
    start = time.time()
    tsobject = TS(calibration_path, vols_path, payment_periods, tstart, N_MC, notional)
    print("Time to build Term Structure objct = " + str(time.time() - start) + " seconds")

    results_not = np.zeros((len(forward_terms), len(expiry_terms)))
    for i in range(len(forward_terms)):
        for j in range(len(expiry_terms)):
            file_path = 'bloomberg_calibration.csv'
            forward_time = forward_terms[i]
            maturity = expiry_terms[j]
            notional = 1e8
            fixed_rate = None
            payment_periods = 3
            payment_periods *= 30
            tau_n = payment_periods/360
            tstart = 0;
            bs = european(file_path, forward_time, maturity, notional, tsobject, fixed_rate, payment_periods, tau_n, N_MC, tstart, True)
            print("\nPayoff of swaption %.3f" % bs.payoff)
            results_model[i,j] = bs.err
            results_not[i,j] = bs.err_mc
    
    mat = [('Underlying Swap Tenor', str(_)) for _ in expiry_terms]
    exp = [('Option Expiry', str(_)) for _ in forward_terms]
    df0 = pd.DataFrame(data=results_model, columns = pd.MultiIndex.from_tuples(mat),
                    index=pd.MultiIndex.from_tuples(exp))
    df1 = pd.DataFrame(data=results_not, columns = pd.MultiIndex.from_tuples(mat),
                    index=pd.MultiIndex.from_tuples(exp))
    display(df0)
    display(df1)

def bermudan():
    # Run #
    glob = dict()
    forward_terms = [1/12, 3/12, 6/12, 1, 2, 5, 10]
    expiry_terms = [5, 10]
    results_black = np.zeros((len(forward_terms), len(expiry_terms)))
    results_notional = np.zeros((len(forward_terms), len(expiry_terms)))
    for i in range(len(forward_terms)):
        for j in range(len(expiry_terms)):
            try:
                file_path = 'bloomberg_calibration.csv'
                tforward = forward_terms[i];
                maturity = expiry_terms[j];
                payment_periods = 6
                payment_periods *= 30
                notional = 1e8
                tstart = 0;
                N_MC = 8192;
                bp1 = bermudan(file_path, tforward, maturity, notional, payment_periods, tstart, N_MC)
                err1 = bp1.black_error
                err2 = bp1.notional_error
            except:
                err1 = np.nan
                err2 = np.nan
            results_black[i, j] = err1
            results_notional[i, j] = err2
    mat = [('Underlying Swap Tenor', str(_)) for _ in expiry_terms]
    exp = [('Option Expiry', str(_)) for _ in forward_terms]
    df2 = pd.DataFrame(data=results_black, columns = pd.MultiIndex.from_tuples(mat),
                    index=pd.MultiIndex.from_tuples(exp))
    df3 = pd.DataFrame(data=results_notional, columns = pd.MultiIndex.from_tuples(mat),
                    index=pd.MultiIndex.from_tuples(exp))
    display(df2) # black error
    display(df3) # notional error

def __main__():
    european()
    bermudan()
    return 0

if __name__ == "__main__":
    __main__()