import numpy as np
from pandas import test
from filepaths import paths

import tensorflow as tf
from pathlib import Path
from filepaths import paths

seconds = 1
Fs = 20

class AI_FPGA:
    def __init__(self):
        model_name = 'cnn_monday_right'
        MODEL_DIR = Path(paths.get('MODEL_DIR'), f'{model_name}.h5')
        self.model = tf.keras.models.load_model(MODEL_DIR)

    seconds = seconds
    Fs = Fs
    frame_size = int(Fs*seconds) # 20Hz * 1 = 20
    hop_size = int(Fs*seconds/4)
    idle_code = 2

    def fpga_predict(self, test_data):
        data = np.array(test_data).reshape(1,20,6)
        ans = self.model.predict(data)
        print(ans)
        return ans, np.argmax(ans)
