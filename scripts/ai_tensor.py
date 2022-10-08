import numpy as np
from filepaths import paths

import tensorflow as tf
from pathlib import Path
from filepaths import paths

seconds = 1
Fs = 40

class AI_FPGA:
    def __init__(self):
        model_name = 'model_cnn_real_0'
        MODEL_DIR = Path(paths.get('MODEL_DIR'), f'{model_name}.h5')
        self.model = tf.keras.models.load_model(MODEL_DIR)

    seconds = seconds
    Fs = Fs
    frame_size = int(Fs*seconds/4) # 20Hz * 1 = 20
    hop_size = int(Fs*seconds/2)

    def fpga_predict(self, test_data):
        # print("CEHCKME!!!!", test_data.shape)
        test_data=tf.expand_dims(test_data, axis=0)
        ans = self.model.predict(test_data)
        return np.argmax(ans)
