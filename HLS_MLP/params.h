#include "ap_fixed.h"
#include "hls_stream.h"

#define FLOAT_WIDTH 16
#define INT_WIDTH 4

typedef ap_fixed<FLOAT_WIDTH, INT_WIDTH> ap_f_t;
typedef float data_t;

#define INPUT_LAYER 120
#define LAYER_0 128
#define LAYER_1 128
#define LAYER_2 64
#define OUTPUT 4

struct AXI_L {
	double data;
	bool last = false;
};

typedef hls::stream<AXI_L> stream_t;
