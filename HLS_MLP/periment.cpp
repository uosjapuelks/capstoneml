#include "ap_axi_sdata.h"
#include "hls_stream.h"

#define INPUT_LAYER 120
#define LAYER_0 128
#define LAYER_1 128
#define LAYER_2 64
#define OUTPUT 4

struct AXIS_wLAST{
	float data;
	bool last;
};



void mlp(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS

	float in_buffer_0[INPUT_LAYER];
	float in_buffer_1[LAYER_0];
	float in_buffer_2[LAYER_1];
	float in_buffer_3[LAYER_2];
	float out_buffer[OUTPUT];

	AXIS_wLAST read_input, write_output;

	while (true) {
		myip_HLS_for1:for (int i = 0; i < INPUT_LAYER; i++) {
			read_input = S_AXIS.read();
			// read_input is the element (data + other signals) received by our ip through S_AXIS in one clock cycle (which contains one word).
			// read() extracts it from the stream. Overloaded operator >> can also be used.
			in_buffer_0[i] = read_input.data; //extracting that word
			// We are not making using of S_AXIS_TLAST in this example.
			// S_AXIS_TLAST is required only when we are receiving an unknown number of words.
		}

		for (int i = 0; i < 4; i++) {
			out_buffer[i] = in_buffer_0[i];
		}

		myip_HLS_for2:for (int i = 0; i < OUTPUT; i++) {
			//write_output.data = sum.to_int();	// using arbitrary precision
			write_output.data = out_buffer[i];			// using 32 bit precision
			// write_output is the element sent by our ip through M_AXIS in one clock cycle.
			write_output.last = 0;
			if (i == OUTPUT - 1)
			{
				write_output.last = 1;
				// M_AXIS_TLAST is required to be asserted for the last word.
				// Else, the AXI Stream FIFO / AXI DMA will not know if all the words have been received from the co-processor.
			}
			M_AXIS.write(write_output);
			// write() inserts it into the stream. Overloaded operator << can also be used.
		}
	}

}
