from time import time
import glob
from assets import european, bermudan
from ts import TS
def __main__():
    ## Single sample run with European swaption #
    glob = dict()
    forward_time = 1/12
    maturity = 10
    notional = 1e8
    fixed_rate = None;
    payment_periods = 3
    payment_periods *= 30
    tau_n = payment_periods/360
    tstart = 0;
    N_MC = 8192;

    calibration_path = 'bloomberg_calibration.csv'
    vols_path = 'vols_liquid.csv'
    start = time.time()
    tsobject = TS(calibration_path, vols_path, payment_periods, tstart, N_MC, notional)
    print("Time to build Term Structure objct = " + str(time.time() - start) + " seconds")

    bs = european(calibration_path, forward_time, maturity, notional, tsobject, fixed_rate, payment_periods, tau_n, N_MC,tstart, True)
    print("\nPayoff of swaption %.3f" % bs.payoff)

    glob = dict()
    file_path = 'bloomberg_calibration.csv'
    tforward = 1/12;
    maturity = 5;
    payment_periods = 6;
    payment_periods *= 30
    notional = 1e8
    tstart = 0;
    N_MC = 8192;
    bp1 = bermudan(file_path, tforward, maturity, notional, payment_periods, tstart, N_MC)

if __name__ == "__main__":
    __main__()