#ifndef _FUNCTS_
#define _FUNCTS_

//#include "ap_fixed.h"

template<class T, int size, int n> class Matrix {
public:
	T a_int[n], b_int[n][size];
	Matrix() {}
	void dot_prod(T a[n], T b[n][size], int j, T &out) {
		#pragma HLS array_partition variable=a_int complete
		#pragma HLS array_partition variable=b_int dim=2 complete
		T dotp = 0;

		for(int i=0; i<n;i++){
		#pragma HLS pipeline
			a_int[i]=a[i];
		}
		for(int i=0; i<n;i++){
		#pragma HLS pipeline
			b_int[i][j]=b[i][j];
		}

		for(int i=0; i<n; i++) {
		#pragma HLS unroll
			dotp += a_int[i] * b_int[i][j];
		}
		out = dotp;
	}
};

template <class T> T ReLu(T a) {
	if (a > 0) {
		return a;
	}
	else {
		return 0;
	}
};

template<class T, int kernelSize, int filterSize, int in_filts, int timesteps> class Conv {
public:
	T in_int[in_filts*kernelSize], conv_kernels[kernelSize][in_filts][filterSize];
	Conv(){}
	void convolute(T input[in_filts*timesteps], T kernel[kernelSize][in_filts][filterSize], int k, int stride, T &out) {
	// filt = 18 or 24 (filterSize)
	#pragma HLS array_partition variable=input complete
	#pragma HLS array_partition variable=kernel dim=3 complete
		T conv_p=0;
		
		//Load input data (size of kernel)
		for (int i=0;i<in_filts*kernelSize;i++){
		#pragma HLS pipeline
			in_int[i]=input[i+stride*in_filts];
		}
		// Load Kernel weights
		for (int i=0; i<in_filts;i++) {
			for (int j=0; j<kernelSize; j++){
			#pragma HLS pipeline
				conv_kernels[j][i][k]=kernel[j][i][k];
			}
		}
		for (int i=0; i<in_filts;i++) {
			for (int j=0; j<kernelSize; j++){
				#pragma HLS unroll
				int idx = j*in_filts+i;
				conv_p += in_int[idx]*conv_kernels[j][i][k];
			}
		}
		out = conv_p;
	}
};

#endif
