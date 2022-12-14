//#include "ap_axi_sdata.h" // ap_axis can also be used, but it will include all sideband signals which we don't need
#include "hls_stream.h"
//#include "ap_int.h"
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
	data_t out_buffer[OUTPUT];

	Matrix<data_t, LAYER_0, INPUT_LAYER> m1;
	Matrix<data_t, LAYER_1, LAYER_0> m2;
	Matrix<data_t, OUTPUT, LAYER_1> m4;

	//ap_uint<8> sum = 0; // using arbitrary precision
	AXIS_wLAST read_input, write_output;
	int test = 1;
	while(test < 2)
	{

		myip_HLS_for1:for(int i = 0; i < INPUT_LAYER; i++){
			read_input = S_AXIS.read();
			// read_input is the element (data + other signals) received by our ip through S_AXIS in one clock cycle (which contains one word).
			// read() extracts it from the stream. Overloaded operator >> can also be used.
			in_buffer_0[i] = read_input.data; //extracting that word
			// We are not making using of S_AXIS_TLAST in this example.
			// S_AXIS_TLAST is required only when we are receiving an unknown number of words.
		}

		data_t val;
		l0:for(int i=0; i<LAYER_0; i++){
			m1.dot_prod(in_buffer_0,w_layer_0,i,val);
			in_buffer_1[i]=ReLu<data_t>(val + bias_0[i]);
		}
		l1:for(int i=0; i<LAYER_1; i++){
			m2.dot_prod(in_buffer_1,w_layer_1,i,val);
			in_buffer_2[i]=ReLu<data_t>(val + bias_1[i]);
		}
		l3:for(int i=0; i<OUTPUT; i++){
			m4.dot_prod(in_buffer_2,w_layer_3,i,val);
			out_buffer[i]=val + out_b[i];
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
		test++;
	}
	
}
