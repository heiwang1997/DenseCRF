#include "densecrf_cpu.h"
#include "pairwise_cpu.h"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <iostream>
#include "util.h"
using namespace DenseCRF;
using namespace std::chrono; 
using namespace std;

// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];
unsigned int getColor( const unsigned char * c ){
	return c[0] + 256*c[1] + 256*256*c[2];
}
void putColor( unsigned char * c, unsigned int cc ){
	c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize( const short * map, int W, int H ){
	unsigned char * r = new unsigned char[ W*H*3 ];
	for( int k=0; k<W*H; k++ ){
        int c = colors[ map[k] ];
        putColor( r+3*k, c );
	}
	return r;
}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

short * classify( const unsigned char * im, int W, int H, int M ){
	short * res = new short[W*H];
	for( int k=0; k<W*H; k++ ){
		// Map the color to a label
		int c = getColor( im + 3*k );
		short i;
		for( i=0;i<nColors && c!=colors[i]; i++ );
		if (c && i==nColors){
			if (i<M)
				colors[nColors++] = c;
			else
				c=0;
		}
		if (c) res[k] = i;
		else res[k] = -1;
	}
	return res;
}

int main( int argc, char* argv[]){
	if (argc<4){
		printf("Usage: %s image annotations output\n", argv[0] );
		return 1;
	}
	// Number of labels
	const int M = 21;
	// Load the color image and some crude annotations (which are used in a simple classifier)
	int W, H, GW, GH;
	unsigned char * im = readPPM( argv[1], W, H );
	if (!im){
		printf("Failed to load image!\n");
		return 1;
	}
	unsigned char * anno = readPPM( argv[2], GW, GH );
	if (!anno){
		printf("Failed to load annotations!\n");
		return 1;
	}
	if (W!=GW || H!=GH){
		printf("Annotation size doesn't match image!\n");
		return 1;
	}
	
	short * label = classify( anno, W, H, M );
	// Setup the CRF model
	DenseCRFCPU<M> crf(W * H);
	crf.setUnaryEnergyFromLabel( label, GT_PROB );
	// add a color independent term (feature = pixel location 0..W-1, 0..H-1)
	// x_stddev = 3
	// y_stddev = 3
	// weight = 3
	auto* smoothnessPairwise = PottsPotentialCPU<M, 2>::FromImage<>(W, H, 3.0, 3.0);
	crf.addPairwiseEnergy( smoothnessPairwise );
	// add a color dependent term (feature = xyrgb)
	// x_stddev = 60
	// y_stddev = 60
	// r_stddev = g_stddev = b_stddev = 20
	// weight = 10
	auto* appearancePairwise = PottsPotentialCPU<M, 5>::FromImage<unsigned char>(W, H, 10.0, 60.0, im, 20.0);
	crf.addPairwiseEnergy( appearancePairwise );
	
	// Do map inference
	auto start = steady_clock::now(); 
	crf.inference(10, true);
	auto stop = steady_clock::now(); 
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time Elaspsed for inference = " << duration.count() / 1000.0 << "ms" << endl; 
	short * map = crf.getMap();
	
	// Store the result
	unsigned char *res = colorize( map, W, H );
	writePPM( argv[3], W, H, res );
	
	delete[] im;
	delete[] anno;
	delete[] res;
	delete[] label;
}
