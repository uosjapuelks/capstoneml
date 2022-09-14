#include "ap_fixed.h"

template<class T, int a_size, int b_size> class Matrix {
public:
	T a[a_size], b[a_size][b_size];
	Matrix() {}
	void dot_prod(T a[a_size], T b[a_size][b_size], int j, T &out) {
		if (a_size<100){
		#pragma HLS array_partition variable=a
		#pragma HLS array_partition variable=b dim=2
		}
		T product = 0;

		for(int i=0; i<a_size; i++) {
		#pragma HLS unroll
			product += a[i] * b[i][j];
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

float dot_(float a[][], float b[], int n, int m) {
	int val = 0;
	for (int i = 0; i<n; i++) {
	#pragma HLS unroll
		val+=a[i][m]*b[i];
	}
	return sum;
}

