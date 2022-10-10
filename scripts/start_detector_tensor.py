import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.stats import stats

from ai_tensor import AI_FPGA
from dataloader import extract_std_range
from model_utils_training import extract_frames, scale_vals

# Make Class
class Detector:

# Store
# Global feat_df (init empty)
# Global prev later half of data
# Global prev_std_bool = init false
# Global res_ls
    def __init__(self, cols=['ax','ay','az','gx','gy','gz']):
        self.cols = cols
        self.prev_data = pd.DataFrame(columns=cols)
        self.cur_data = pd.DataFrame(columns=cols)
        self.fpga = AI_FPGA()
        self.res_ls = [self.fpga.idle_code]
        self.feat_df = pd.DataFrame()
        self.prev_std_bool = False
        self.counter = 0

    def process_data(self, raw_data_row):
        raw_data_row = np.array(raw_data_row).reshape(1,6)
        new_row = pd.DataFrame(raw_data_row, columns=self.cols)
        self.cur_data=pd.concat([self.prev_data, new_row], ignore_index=True)
        self.prev_data=self.cur_data.iloc[:]

    def check_df_threshold(self, data):
    # current data
    # current tmp
    # Append data to prev -> into tmp
    # prev = later half of data
        # tmp = pd.concat([self.prev_data, data], ignore_index=True)
        # self.prev_data=data.iloc[-self.counter:]
        feat = extract_std_range(data, self.fpga.Fs)
        # self.cur_data=data
        max_std = (max(feat['std_a']))
        return max_std > 0.0


# Function - Constantly called by External Comms after initializing class
    def eval_data(self, raw_data):
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
        # if self.prev_std_bool:


        # if pass_threshold:
        #     self.prev_std_bool=True
        #     chance_fpga, res_fpga = self.fpga.fpga_predict(data)
        #     if res_fpga!=2:
        #         print(res_fpga)
            
        #     # NOTE on actual fpga, run softmax first
        #     # if chances greater than 0.88 append
        #     if chance_fpga[0][res_fpga] > 0.00:
        #         self.res_ls.append(res_fpga)
        #     else:
        #         self.res_ls.append(self.fpga.idle_code)
        #     # if latest append is IDLE
        #     if self.res_ls[-1]==self.fpga.idle_code or self.res_ls[-1]!=self.res_ls[-2]:
        #         if len(self.res_ls) > 1:
        #             ret_val = self.res_ls[-2]
        #             self.res_ls=[self.res_ls[-1]]
        #             if ret_val==None:
        #                 ret_val=self.fpga.idle_code
        #             return ret_val
        #         ## CAUSING PROBLEMS
        #         else:
        #             self.res_ls=[self.fpga.idle_code]
        #             return self.fpga.idle_code
        # elif self.prev_std_bool:
        #     print("MAYBE THIS", stats.mode(self.res_ls, keepdims=True)[0][0])
        #     return 1
        # else:
        #     self.prev_std_bool=False
        #     return self.fpga.idle_code

        chance_fpga, res_fpga = self.fpga.fpga_predict(data)

        return res_fpga

