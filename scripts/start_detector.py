import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import stats

from ai import AI_FPGA
from dataloader import extract_std_range
from modelling_utils import scale_vals

# Make Class
class Detector:

# Store
# CLASSES: {0: Exit, 1: Grenade, 2: EXIT, 3: Reload, 4: Shield}
    def __init__(self, model='cnn_mix_moretrng.bit', cols=['ax','ay','az','gx','gy','gz']):
        self.cols = cols
        self.prev_data = pd.DataFrame(columns=cols)
        self.cur_data = pd.DataFrame(columns=cols)
        self.fpga = AI_FPGA(model)
        self.res_ls = [self.fpga.idle_code]
        self.feat_df = pd.DataFrame()
        self.margin = 0
        self.counter = 0

    # Collect and accumulate rows into dataframes of 20
    def process_data(self, raw_data_row):
        raw_data_row = np.array(raw_data_row).reshape(1,6)
        new_row = pd.DataFrame(raw_data_row, columns=self.cols)
        self.cur_data=pd.concat([self.prev_data, new_row], ignore_index=True)
        self.prev_data=self.cur_data.iloc[:]

    # Check if selected threshold is exceeded
    def check_df_threshold(self, data):
        feat = extract_std_range(data, self.fpga.frame_size)
        max_std = (max(feat['std_a']))
        return max_std > 0.065

    # Check for return value
    def checkRetVal(self):
        length = len(self.res_ls)
        ret_val = int(stats.mode(self.res_ls)[0][0])
        if ret_val==0 and length>15:
            # self.res_ls = [self.fpga.idle_code]
            return ret_val
        elif length > 4 and ret_val!=0:
            # self.res_ls = [self.fpga.idle_code]
            return ret_val
        else:
            return self.fpga.idle_code
    
    # margins help with acccidental classification of idle randomly
    def checkMargins(self):
        if self.margin==0:
            ret_val = self.checkRetVal()
            self.res_ls = [self.res_ls[-1]]
        else:
            self.margin-=1
            ret_val = self.fpga.idle_code
        return ret_val

# Function - Constantly called by External Comms after initializing class
    def eval_data(self, raw_data, errMarg=1, sensitivity=0.70):
        self.counter+=1
        self.process_data(raw_data) # Return df of raw data
        if self.counter < 20:
            return self.fpga.idle_code
        elif self.counter==20:
            self.counter-=self.fpga.hop_size # Hop size acording to AIFPGA
            self.prev_data=self.cur_data.iloc[-self.counter:]
        
        data = self.cur_data.copy()
        data = scale_vals(data)
        pass_threshold = self.check_df_threshold(data)

        # Std Activated -> get chances and res_fpga
        if pass_threshold:
            chance_fpga, res_fpga = self.fpga.fpga_predict(data)
            if res_fpga!=self.fpga.idle_code:
            # NOTE on actual fpga, run softmax first
            # if chances greater than 0.88 append
                if chance_fpga[res_fpga] > sensitivity or res_fpga==0:
                    # Reset margin to errMarg
                    self.margin=errMarg
                    self.res_ls.append(res_fpga)
        #         else:
        #             ret_val = self.checkMargins()
        #             return ret_val
        #     else: # the res_fpga IS IDLE
        #         ret_val = self.checkMargins()
        #         return ret_val
        # else:
        #     ret_val = self.checkMargins()
        #     return ret_val
        ret_val = self.checkMargins()
        return ret_val

        # return self.fpga.idle_code
