import numpy as np
import pandas as pd
from scipy.stats import stats

from ai_tensor import AI_FPGA
from dataloader import extract_std_range
from model_utils_training import extract_frames


# Permanent
# fs (frequency)
fs = 20

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
    
    feat_df = pd.DataFrame()
    prev_std_bool = False
    res_ls = [2]
    counter = -1

# Setup DMA stuff
    

    def process_data(self, raw_data_row):
        raw_data_row = np.array(raw_data_row).reshape(1,6)
        new_row = pd.DataFrame(raw_data_row, columns=self.cols)
        self.cur_data=pd.concat([self.cur_data, new_row], ignore_index=True)
        return


# Function - Constantly called by External Comms after initializing class
    def eval_data(self, raw_data):
        self.counter+=1
        self.process_data(raw_data) # Return df of raw data
        if self.counter < 20:
            return 2
        elif self.counter==20:
            self.counter=-1

        data = self.cur_data
        self.cur_data = pd.DataFrame(columns=self.cols)

    # current data
    # current tmp
    # Append data to prev -> into tmp
    # prev = later half of data
        tmp = pd.concat([self.prev_data,data], ignore_index=True)
        self.prev_data = data.iloc[int(fs/2):]
    
    # Extract features (range & std only)
        feat = extract_std_range(tmp, fs, timestart=0)

    # check if std > threshold (ONCE - so check max)
    # if True: prev_std_bool = True 
        max_std = (max(feat['std_a']))
        if max_std > 0.075:
            self.prev_std_bool = True
    # if False: 
        # if prev_std_bool -> set False
        # release res
        else:
            if self.prev_std_bool==True:
                self.prev_std_bool = False
                # RELEASE RES
                # print(self.res_ls)
                return stats.mode(self.res_ls)[0][0] # NOT SURE
            else:
                return 2 # (4 IS IDLE)

        # store index where std > threshold -> Start Call AI from index
        # True -> call Ai Predict Function -> store in res_ls               2 out of 3 predictions should not be IDLE & keep last val to compare with next half if std > thres
        if self.prev_std_bool:
            idx = feat.std_a[feat.std_a==max_std].index.tolist()[0]*self.fpga.frame_size
            frames = extract_frames(data,self.fpga.frame_size,self.fpga.hop_size,start_idx=idx)
            tmp_res=[]
            print(frames)
            for frame in frames:
                res_fpga = self.fpga.fpga_predict(frame)
                tmp_res.append(res_fpga)
                print("NOW", res_fpga, "AND", tmp_res)
        # if Predict res list mode == IDLE -> Ignore Threshold Rise
        # res_ls = res_ls[last]
            res_mode = stats.mode(tmp_res)[0][0]
            if res_mode==2:
                
                self.res_ls=[self.res_ls[-1]]
        # else res_ls append res > check res_lsmode
            else:
                if res_mode!=self.res_ls[-1]:
                    ret_val = self.res_ls[-1]
                    self.res_ls=[res_mode]
                    return ret_val
                else:
                    self.res_ls.append(res_mode)
        # if res_mode != prevres:
            # return res_ls content                                         Ensure action done before return res
        return 2 # IDLE
