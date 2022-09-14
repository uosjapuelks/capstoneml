/*
----------------------------------------------------------------------------------
--	(c) Rajesh C Panicker, NUS,
--  Description : AXI Stream Coprocessor (HLS), implementing the sum of 4 numbers
--	License terms :
--	You are free to use this code as long as you
--		(i) DO NOT post a modified version of this on any public repository;
--		(ii) use it only for educational purposes;
--		(iii) accept the responsibility to ensure that your implementation does not violate any intellectual property of any entity.
--		(iv) accept that the program is provided "as is" without warranty of any kind or assurance regarding its suitability for any particular purpose;
--		(v) send an email to rajesh.panicker@ieee.org briefly mentioning its use (except when used for the course EE4218 at the National University of Singapore);
--		(vi) retain this notice in this file or any files derived from this.
----------------------------------------------------------------------------------
*/

//#include "ap_axi_sdata.h" // ap_axis can also be used, but it will include all sideband signals which we don't need
#include "hls_stream.h"
#include "ap_int.h"
#include "params.h"
#include "mlfunctions.h"

// Creating a custom structure which includes the data word and TLAST signal.
// ACLK, ARESETN, TREADY, TDATA, TVALID are essential signals for AXIS.
// TLAST is a sideband signal which is optional in AXIS.
// However, it is necessary for us since we connecting M_AXIS to AXI Stream FIFO / AXI DMA.
// So, we create a struct with data (TDATA) and last (TLAST). The rest of the essential AXIS signals are automatically dealt with by the HLS tool.

void myip_HLS(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS

	data_t in_buffer_0[INPUT_LAYER];
	data_t in_buffer_1[LAYER_0];
	data_t in_buffer_2[LAYER_1];
	data_t in_buffer_3[LAYER_2];
	data_t out_buffer[OUTPUT];

	// Matrix<data_t, INPUT_LAYER, LAYER_0> m0;
	// Matrix<data_t, LAYER_0, LAYER_1> m1;
	// Matrix<data_t, LAYER_1,LAYER_2> m2;
	// Matrix<data_t, LAYER_2,OUTPUT> m3;

	int listening = 1;
	bool not_done = true;

	//ap_uint<8> sum = 0; // using arbitrary precision
	int sum = 0;		 // using 32 bit precision
	AXIS_wLAST read_input, write_output;

	while(not_done)
	{

		myip_HLS_for1:for(int i = 0; i < INPUT_LAYER; i++){
			read_input = S_AXIS.read();
			// read_input is the element (data + other signals) received by our ip through S_AXIS in one clock cycle (which contains one word).
			// read() extracts it from the stream. Overloaded operator >> can also be used.
			in_buffer_0[i] = read_input.data; //extracting that word
			// We are not making using of S_AXIS_TLAST in this example.
			// S_AXIS_TLAST is required only when we are receiving an unknown number of words.
		}

		// BEGIN DOT
		myip_HLS_0: for (i = 0; i < LAYER_0; i++) {
		#pragma HLS pipeline
			in_buffer_1[i] = dot_(w_layer_0, in_buffer_0, INPUT_LAYER, i) + bias_0[i];
			in_buffer_1[i] = ReLu(in_buffer_1[i]);
		}
		myip_HLS_1: for (i = 0; i < LAYER_1; i++) {
		#pragma HLS pipeline
			in_buffer_2[i] = dot_(w_layer_1, in_buffer_1, LAYER_0, i) + bias_1[i];
			in_buffer_2[i] = ReLu(in_buffer_2[i]);
		}
		myip_HLS_1: for (i = 0; i < LAYER_2; i++) {
		#pragma HLS pipeline
			in_buffer_3[i] = dot_(w_layer_2, in_buffer_3, LAYER_1, i) + bias_2[i];
			in_buffer_3[i] = ReLu(in_buffer_3[i]);
		}
		myip_HLS_1: for (i = 0; i < LAYER_1; i++) {
		#pragma HLS pipeline
			out_buffer[i] = dot_(w_layer_3, in_buffer_3, LAYER_2, i) + out_b[i];
			out_buffer[i] = ReLu(out_buffer[i]);
		}


		myip_HLS_for2:for(int i = 0; i < OUTPUT; i++){
			//write_output.data = sum.to_int();	// using arbitrary precision
			write_output.data = out_buffer[i];			// using 32 bit precision
			// write_output is the element sent by our ip through M_AXIS in one clock cycle.
			write_output.last = 0;
			if(i==OUTPUT-1)
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
