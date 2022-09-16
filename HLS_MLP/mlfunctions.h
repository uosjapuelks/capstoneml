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

#endif
