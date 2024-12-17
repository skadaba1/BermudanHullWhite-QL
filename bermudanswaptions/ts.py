# import the used libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import warnings
import cvxpy as cp
import QuantLib as ql
from tqdm import tqdm
import glob
from utils import calibration_report
warnings.filterwarnings('ignore')

""" Term Structure handler, intialized via calibration against swaptions and 
              connstruction from liquid instruments"""
tstart = 0
## Term Structure Model ##
class TS:
    def __init__(self, calibration_path, vols_path, payment_periods, tstart, num_paths, notional):
        self.discountBoundPaths = None
        self.num_paths = num_paths
        self.term_structure = self.build_curve(calibration_path, tstart);
        self.calibrated_model, gauss_generator = self.calibrate(vols_path, notional, payment_periods)
        _, self.short_rate_paths = self.generate_paths(num_paths, gauss_generator)
        #plt.plot(self.short_rate_paths.T)
        
    def generate_paths(self, num_paths, seq):
        timestep = 5400
        arr = np.zeros((num_paths, timestep+1))
        for i in range(num_paths):
            sample_path = seq.next()
            path = sample_path.value()
            time = [path.time(j) for j in range(len(path))]
            value = [path[j] for j in range(len(path))]
            arr[i, :] = np.array(value)
        return np.array(time), arr   
    
    def build_curve(self, calibration_path, tstart):
        df = pd.read_csv(calibration_path)
        data = []
        for i in range(len(df)):
            dt = dict(df.loc[i])
            dt['Date'] = '2023-06-30 00:00:00'
            data.append(dt)

        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        zeros = []
        deposits = ['1M', '3M', '6M', '12M']
        swaps = ['2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']
        for row in data[tstart:tstart+1]:
          
            # Build Curve for the date
            curve_date = ql.Date(row['Date'][:10], '%Y-%m-%d')
            ql.Settings.instance().evaluationDate = curve_date
            spot_date = calendar.advance(curve_date, 0, ql.Days) # beginning of eval - 2 days forward
            helpers = ql.RateHelperVector()
            for tenor in deposits:
                index = ql.USDLibor(ql.Period(tenor))
                helpers.append(
                    ql.DepositRateHelper(row[tenor] / 100, index)
                )
            for tenor in swaps:
                swap_index = ql.UsdLiborSwapIsdaFixAm(ql.Period(tenor))
                helpers.append(
                    ql.SwapRateHelper(row[tenor] / 100, swap_index)
                )
            curve = ql.PiecewiseCubicZero(curve_date, helpers, ql.Actual360())
        curve.enableExtrapolation()
        term_structure = ql.YieldTermStructureHandle(curve)
        return term_structure

    def calibrate(self, vols_path, notional, payment_periods):
      
        timestep = 5400
        model = ql.HullWhite(self.term_structure);
        engine = ql.JamshidianSwaptionEngine(model)

        index = ql.Euribor6M(self.term_structure)
        fixedLegTenor = ql.Period(payment_periods, ql.Months) # originally 1Y
        fixedLegDayCounter = ql.Actual360()
        floatingLegDayCounter = ql.Actual360()

        ## Grab volatilities ##
        df  = pd.read_csv(vols_path)
        date1 = df.loc[tstart] # Index important here
        volcub = np.reshape(date1[1:].values, (7,2))
        tenor_labels = [_ for _ in [5, 10]]
        expiry_labels = [_ for _ in [1/12, 3/12, 6/12, 1, 2, 5, 10]]
        self.swaption_vols = pd.DataFrame(volcub, index=expiry_labels, columns=tenor_labels)
        swaptions = []
        ql.Settings.instance().evaluationDate = ql.Date(30, 6, 2023)
        vols_data = []
   
        cnt = 0
        for maturity in self.swaption_vols.index:
            for tenor in self.swaption_vols.columns:
                vol = self.swaption_vols.at[maturity, tenor] / 100
                if(vol != 0):
                        vols_data.append({'vol':vol, 'expiry':maturity, 'maturity':tenor})
                        volatility = ql.QuoteHandle(ql.SimpleQuote(vol))
                        helper = ql.SwaptionHelper(
                            ql.Period(int(maturity*12), ql.Months),
                            ql.Period(int(tenor), ql.Years),
                            volatility,
                            index,
                            fixedLegTenor,
                            fixedLegDayCounter,
                            floatingLegDayCounter,
                            self.term_structure,
                            ql.BlackCalibrationHelper.RelativePriceError,
                            ql.nullDouble(),
                            notional, # nominal
                            ql.ShiftedLognormal, # ShiftedLogn
                            0.00 #shift to make rates non-negative
                        )
                        helper.setPricingEngine(engine)
                        swaptions.append(helper)
                        glob[(maturity, tenor)] = cnt;
                        cnt += 1;
                    
                else:
                    raise ValueError("0 vol. provied!")
    

        optimization_method = ql.LevenbergMarquardt(1.0e-8,1.0e-8,1.0e-8)
        end_criteria = ql.EndCriteria(500000, 1000, 1e-6, 1e-8, 1e-8)
        model.calibrate(swaptions, optimization_method, end_criteria)
        
        self.params = model.params()
        
        df = calibration_report(swaptions, vols_data)
        #display(df)
        
        sigma = self.params[1]
        a = self.params[0]
        length = 15 # in years
        day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        todays_date =  ql.Date(30, 6, 2023)

        ql.Settings.instance().evaluationDate = todays_date

        hw_process = ql.HullWhiteProcess(self.term_structure, a, sigma)
        
        rng = ql.GaussianLowDiscrepancySequenceGenerator(ql.UniformLowDiscrepancySequenceGenerator(timestep))
        seq = ql.GaussianSobolPathGenerator(hw_process, length, timestep, rng, False)

        #rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
        #seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)
        
        self.swaptions = swaptions;
            
        return model, seq
    
    def ZCBCurve(self, t, k=None, sched=None):
        """ P(t, T) for all t, T """
        if(sched is None):
            raise ValueError("Payment schedule not provided!")
        sched = sched/360
        def discountBond(T, k):
            r = self.short_rate_paths[k,int(t*360)]
            return self.calibrated_model.discountBond(t, T, r)
        
        if(k is not None):
            path = self.short_rate_paths[k, :];
            PTt = [self.calibrated_model.discountBond(t, i+t, path[int(t*360)]) for i in sched] #path[int(t*12)]
            return PTt;

        ts = sched + t
        ks = np.arange(self.num_paths)
        T, K = np.meshgrid(ts, ks)
        PTts = np.fromiter(map(discountBond, T.ravel(), K.ravel()), T.dtype).reshape(T.shape)
        self.discountBondPaths = PTts
        num_traj, traj_length = PTts.shape
        PTtsAvg = np.mean(PTts, axis=0)
        
        return PTtsAvg
    