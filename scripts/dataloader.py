import pandas as pd
import numpy as np
from filepaths_extComms import paths

def get_training_files():
    trainfiles = list(paths.get('SELF_DIR').glob('*'))
    return trainfiles

def get_real_testdata():
    testfiles = list(paths.get('TEST_DIR').glob('*'))
    return testfiles

def load_online_data():
    gyrofiles = list(paths.get('GYRO_DIR').glob('*'))
    accelfiles = list(paths.get('ACCEL_DIR').glob('*'))

    a_column = ['Subject_id', 'Activity_code', 'time', 'ax', 'ay', 'az']
    g_column = ['Subject_id', 'Activity_code', 'time', 'gx', 'gy', 'gz']
    
    files = gyrofiles.__add__(accelfiles)
    files = unlink_DStore(files)

    for i in range(len(gyrofiles)):
        f_g = sorted(gyrofiles)[i]
        f_a = sorted(accelfiles)[i]
        tmp_dfg = pd.read_csv(f_g, names=g_column)
        tmp_dfa = pd.read_csv(f_a, names=a_column)
        # tmp_dfg = tmp_dfg.drop('time', axis=1)
        # tmp_dfg = tmp_dfg.drop('time', axis=1).drop(['Subject_id', 'Activity_code'], axis=1)
        if i == 0: 
            full_df = pd.merge(tmp_dfa, tmp_dfg, left_index=True, right_index=True)
        elif i > 0:
            f_df = pd.merge(tmp_dfa, tmp_dfg, left_index=True, right_index=True)
            full_df = pd.concat([full_df, f_df], axis=0, join='outer')
    # Filter to keep these
    # Walking Brushing Dribbling Clapping
    filt_df = full_df[full_df['Activity_code'].isin(['A','G','P','R'])]

    filt_df['az'] = filt_df['az'].str.slice(start=0, stop=-1)
    filt_df['gz'] = filt_df['gz'].str.slice(start=0, stop=-1)
    filt_df['az'] = filt_df['az'].astype('float')
    filt_df['gz'] = filt_df['gz'].astype('float')

    return filt_df

def unlink_DStore(files):
    for f in files:
        try:
            pd.read_csv(f)
        except:
            f.unlink()
    return files

def load_to_df(trainfiles, col_names):
    for i in range(len(trainfiles)):
        f = trainfiles[i]
        tmp_df = pd.read_csv(f, names=col_names)
        if i == 0:
            df = tmp_df
        if i > 0:
            df = pd.concat([df, tmp_df], axis=0, join='outer')
    return df

def get_range(series):
    return max(series)-min(series)

def get_mean(series):
    return series.mean()

def get_std(series):
    return series.std()

def get_features(series, target=['max','min','range','mean','std'], verbose=False):
    com_dict = {
        'max': max,
        'min': min,
        'range': get_range,
        'mean': get_mean,
        'std': get_std
    }
    if verbose:
        return target
    features = []
    for t in target:
        features.append(com_dict[t](series))
    return features    

def get_m_features(fdf, verbose=False):
    features=[]
    fdf["a"] = np.linalg.norm(
        (fdf["ax"],fdf["ay"],fdf["az"]), axis=0
    )
    fdf["g"] = np.linalg.norm(
        (fdf["gx"],fdf["gy"],fdf["gz"]), axis=0
    )

    for v in ['a','g']:
        features_tmp = get_features(fdf[v], verbose=verbose)
        if verbose:
            features.extend(x + '_' + v for x in features_tmp)
        else:
            features.extend(features_tmp)
    
    return features

def extract_features(df_copy, timestep, timestart=0):
    idx = 0-timestep
    all_time_features=[]
    for i in range(int(df_copy.shape[0]/timestep)):
        idx += timestep
        df_c_tmp = df_copy.iloc[idx:idx+20,:]
        features_names = get_m_features(df_c_tmp, verbose=True)
        features = get_m_features(df_c_tmp)
        all_time_features.append(features)

    feat_df = pd.DataFrame(all_time_features,columns=features_names)
    feat_df['time'] = [x+timestart for x in range(feat_df.shape[0])]

    return feat_df

def get_target_feat(data, verbose=False):
    features=[]
    data["a"]=np.linalg.norm((data["ax"],data["ay"],data["az"]), axis=0)
    data["g"]=np.linalg.norm((data["gx"],data["gy"],data["gz"]), axis=0)

    for v in ['a', 'g']:
        f_tmp = get_features(data[v], target=['range', 'std'], verbose=verbose)
        if verbose:
            features.extend(x+'_'+v for x in f_tmp)
        else:
            features.extend(f_tmp)
    return features

def extract_std_range(df_copy, timestep, timestart=0):
    idx = 0-timestep
    all_time_features=[]
    # for i in range(int(df_copy.shape[0]/timestep)):
    idx += timestep
    df_c_tmp = df_copy.iloc[idx:idx+timestep,:]
    features_names = get_target_feat(df_c_tmp, verbose=True)
    features = get_target_feat(df_c_tmp)
    all_time_features.append(features)

    feat_df = pd.DataFrame(all_time_features, columns=features_names)
    feat_df['time'] = [x+timestart for x in range(feat_df.shape[0])]

    return feat_df