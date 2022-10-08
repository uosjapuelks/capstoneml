from pynq import Overlay
from pynq import allocate
import numpy as np
import os

from filepaths import paths

BIT = os.path.join(paths.get('BITS_DIR'), 'conv_design_0.bit')

class AI_FPGA:
    overlay = Overlay(BIT)
    dma = overlay.axi_dma_0

    in_buffer0 = allocate(shape=(120,), dtype=np.float32)
    out_buffer0 = allocate(shape=(1,), dtype=np.float32)
    seconds = 1
    Fs = 20
    frame_size = Fs*seconds # 20Hz * 1 = 20
    hop_size = int(Fs*seconds/2)

    def fpga_predict(self, test_data):
        test_data = np.array(test_data).reshape(120)
        for i in range(120):
            self.in_buffer0[i]=test_data[i]
        self.dma.sendchannel.transfer(self.in_buffer0)
        self.dma.recvchannel.transfer(self.out_buffer0)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        return int(self.out_buffer0[0])