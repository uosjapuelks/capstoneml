#include "ap_fixed.h"

template<class T, int a_size, int b_size> class Matrix {
public:
	T a_int[a_size], b_int[a_size];
	Matrix() {}
	void dot_prod(T a[a_size], T b[a_size], T &out) {
		#pragma HLS array_partition variable=a_int
		#pragma HLS array_partition variable=b_int
		T product = 0;

//		for (int i=0; i<size; i++) {
//		#pragma HLS pipeline
//			a_int[i] = a[i];
//		}
//		for (int i=0; i<size; i++) {
//		#pragma HLS pipeline
//			b_int[i] = b[i];
//		}
//
		for(int i=0; i<b_size; i++) {
		#pragma HLS unroll
			product += a_int[i] * b_int[i];
		}
		out = product;
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

