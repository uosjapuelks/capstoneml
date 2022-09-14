#include "mlfunctions.h"
#include "params.h"

void mlpLaserTag(stream_t S_AXIS, stream_t& M_AXIS){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS //Slave -IN
#pragma HLS INTERFACE axis port=M_AXIS //Master-OUT

	data_t input[INPUT_LAYER];

	data_t layer_w_0[INPUT_LAYER][LAYER_0];
	data_t layer_w_1[LAYER_0][LAYER_1];
	data_t layer_w_2[LAYER_1][LAYER_2];
	data_t out_w[LAYER_2][OUTPUT];

	data_t bias_0[LAYER_0];
	data_t bias_1[LAYER_1];
	data_t bias_2[LAYER_2];
	data_t out_b[OUTPUT];

	data_t in_buffer_0[INPUT_LAYER];
	data_t in_buffer_1[LAYER_0];
	data_t in_buffer_2[LAYER_1];
	data_t in_buffer_3[LAYER_2];
	data_t out_buffer[OUTPUT];

	Matrix<data_t, INPUT_LAYER, LAYER_0> m0;
	Matrix<data_t, LAYER_0, LAYER_1> m1;
	Matrix<data_t, LAYER_1,LAYER_2> m2;
	Matrix<data_t, LAYER_2,OUTPUT> m3;

	int initialised = 0;
	int listening = 1;

	while (true) {
		AXI_L in, out;
		if (initialised==0){
			load_w0:for (int i=0; i<INPUT_LAYER;i++){
				for (int j=0; j<LAYER_0;j++){
					S_AXIS >> in;
					layer_w_0[i][j] = in.data;
				}
			}
			load_w1:for (int i=0; i<LAYER_0;i++){
				for (int j=0; j<LAYER_1;j++){
					S_AXIS >> in;
					layer_w_1[i][j] = in.data;
				}
			}
			load_w2:for (int i=0; i<LAYER_1;i++){
				for (int j=0; j<LAYER_2;j++){
					S_AXIS >> in;
					layer_w_2[i][j] = in.data;
				}
			}
			load_w3:for (int i=0; i<LAYER_2;i++){
				for (int j=0; j<OUTPUT;j++){
					S_AXIS >> in;
					out_w[i][j] = in.data;
				}
			}
			load_b0:for (int i=0; i<LAYER_0;i++){
				S_AXIS >> in;
				bias_0[i] = in.data;
			}
			load_b1:for (int i=0; i<LAYER_1;i++){
				S_AXIS >> in;
				bias_1[i] = in.data;
			}
			load_b2:for (int i=0; i<LAYER_2;i++){
				S_AXIS >> in;
				bias_2[i] = in.data;
			}
			output_b:for (int i=0; i<OUTPUT;i++){
				S_AXIS >> in;
				out_b[i] = in.data;
			}
			initialised=1;
		}
		else if (initialised==1) {
			if (listening==1){
				listening=0;
				listen:for (int i=0; i<INPUT_LAYER;i++) {
					S_AXIS >> in;
					input[i] = in.data;
				}
			}
			else if (listening==0) {
				data_t val;
				l0:for(int i=0; i<LAYER_0; i++){
				#pragma HLS pipeline
					m0.dot_prod(input,layer_w_0,i,val);
					in_buffer_0[i]=ReLu<data_t>(val + bias_0[i]);
				}
				l1:for(int i=0; i<LAYER_1; i++){
				#pragma HLS pipeline
					m1.dot_prod(in_buffer_0,layer_w_1,i,val);
					in_buffer_1[i]=ReLu<data_t>(val + bias_1[i]);
				}
				l2:for(int i=0; i<LAYER_2; i++){
				#pragma HLS pipeline
					m2.dot_prod(in_buffer_1,layer_w_2,i,val);
					in_buffer_2[i]=ReLu<data_t>(val + bias_2[i]);
				}
				l3:for(int i=0; i<OUTPUT; i++){
				#pragma HLS pipeline
					m3.dot_prod(in_buffer_2,out_w,i,val);
					out_buffer[i]=ReLu<data_t>(val + out_b[i]);
				}
				output_0:for(int i=0; i<OUTPUT; i++){
					out.data=out_buffer[i];
					if(i==(OUTPUT-1)){
						out.last=true;
					}
				}
				M_AXIS << out;
				listening=1;
			}
		}
	}

}
