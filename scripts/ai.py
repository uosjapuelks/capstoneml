from pynq import Overlay
from pynq import allocate
import numpy as np
import os

# from filepaths_extComms import paths
from filepaths import paths
from modelling_utils import softmax


class AI_FPGA:
    def __init__(self, bitstream='cnn_mix_moretrng.bit'):
        BIT = os.path.join(paths.get('BITS_T_DIR'), bitstream)
        print(BIT)
        self.overlay = Overlay(BIT)
        self.dma = self.overlay.axi_dma_0

        self.in_buffer0 = allocate(shape=(120,), dtype=np.float32)
        self.out_buffer0 = allocate(shape=(5,), dtype=np.float32)

    ###
    ### TAKE NOTE! Fs MAY CHANGE
    ###
    seconds = 1
    Fs = 40
    frame_size = int(Fs*seconds/2) # 20Hz * 1 = 20
    hop_size = int(Fs*seconds/8)
    idle_code = 2

    def fpga_predict(self, test_data):
        chances=[]
        test_data = np.array(test_data).reshape(120)
        for i in range(120):
            self.in_buffer0[i]=test_data[i]
        self.dma.sendchannel.transfer(self.in_buffer0)
        self.dma.recvchannel.transfer(self.out_buffer0)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        for i in range(5):
            chances.append(self.out_buffer0[i])

        chances = np.array(chances)
        chances = softmax(chances)
        return chances, np.argmax(chances)