# from pynq import Overlay
# from pynq import allocate
import numpy as np
import os

root = os.getcwd()
BIT = os.path.join(root, 'conv_design_0.bit')

# print(BIT)

# class AI_FPGA:
#     overlay = Overlay(BIT)
#     dma = overlay.axi_dma_0

#     in_buffer0 = allocate(shape=(120,), dtype=np.float32)
#     out_buffer0 = allocate(shape=(1,), dtype=np.float32)
#     seconds = 1
#     Fs = 20
#     frame_size = Fs*seconds # 20Hz * 1 = 20
#     hop_size = int(Fs*seconds/2)

#     def fpga_predict(self, test_data):
#         for i in range(120):
#             self.in_buffer0[i]=test_data[i]
#         self.dma.sendchannel.transfer(self.in_buffer0)
#         self.dma.recvchannel.transfer(self.out_buffer0)
#         self.dma.sendchannel.wait()
#         self.dma.recvchannel.wait()
#         return self.out_buffer0
import tensorflow as tf
from pathlib import Path
from filepaths import paths

class AI_FPGA:
    def __init__(self):
        model_name = 'model_cnn_real_0'
        MODEL_DIR = Path(paths.get('MODEL_DIR'), f'{model_name}.h5')
        self.model = tf.keras.models.load_model(MODEL_DIR)

    # overlay = Overlay(BIT)
    # dma = overlay.axi_dma_0

    # in_buffer0 = allocate(shape=(120,), dtype=np.float32)
    # out_buffer0 = allocate(shape=(1,), dtype=np.float32)
    seconds = 1
    Fs = 20
    frame_size = Fs*seconds # 20Hz * 1 = 20
    hop_size = int(Fs*seconds/2)

    # def fpga_predict(self, test_data):
    #     for i in range(120):
    #         self.in_buffer0[i]=test_data[i]
    #     self.dma.sendchannel.transfer(self.in_buffer0)
    #     self.dma.recvchannel.transfer(self.out_buffer0)
    #     self.dma.sendchannel.wait()
    #     self.dma.recvchannel.wait()
    #     return self.out_buffer0
    def fpga_predict(self, test_data):
        # print("CEHCKME!!!!", test_data.shape)
        test_data=tf.expand_dims(test_data, axis=0)
        ans = self.model.predict(test_data)
        return np.argmax(ans)
