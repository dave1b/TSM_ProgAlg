#include <iostream>
#include <memory>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <sycl/sycl.hpp>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wignored-attributes"
#endif
#include "FreeImagePlus.h"

////////////////////////////////////////////////////////////////////////
// compute and return distance 
static BYTE dist(int dx, int dy) {
	const int d = (int)sqrt(dx*dx + dy*dy);
	return (d < 256) ? d : 255;
}

////////////////////////////////////////////////////////////////////////
// Sequential image processing (filtering).
// It computes the first derivative in x- and y-direction
void processSerial(const fipImage& input, fipImage& output, const int* horFilter, const int* verFilter, unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && 
		input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == 32);

	const int halfFilterSize = filterSize/2;

	// iterate all rows of the output image
	for (unsigned v = halfFilterSize; v < output.getHeight() - halfFilterSize; v++) {
		// iterate all pixels of the current row
		for (unsigned u = halfFilterSize; u < output.getWidth() - halfFilterSize; u++) {
			RGBQUAD inputColor;
			int horColor[3]{};
			int verColor[3]{};
			int filterIndex = 0;

			// iterate all filter coefficients
			for (unsigned j = 0; j < filterSize; j++) {
				for (unsigned i = 0; i < filterSize; i++) {
					const int horFilterCoeff = horFilter[filterIndex];
					const int verFilterCoeff = verFilter[filterIndex];

					// read pixel color at position (u + i - halfFilterSize, v + j - halfFilterSize) and store the color in inColor
					input.getPixelColor(u + i - halfFilterSize, v + j - halfFilterSize, &inputColor);

					// apply one filter coefficient of the horitontal filter
					horColor[0] += horFilterCoeff*inputColor.rgbBlue;
					horColor[1] += horFilterCoeff*inputColor.rgbGreen;
					horColor[2] += horFilterCoeff*inputColor.rgbRed;
					// apply one filter coefficient of the vertical filter
					verColor[0] += verFilterCoeff*inputColor.rgbBlue;
					verColor[1] += verFilterCoeff*inputColor.rgbGreen;
					verColor[2] += verFilterCoeff*inputColor.rgbRed;

					filterIndex++;
				}
			}

			// compute filter result and store it in output pixel (col, row)
			RGBQUAD outColor = { 
				dist(horColor[0], verColor[0]), 
				dist(horColor[1], verColor[1]), 
				dist(horColor[2], verColor[2]), 
				255 };
			output.setPixelColor(u, v, &outColor);
		}
	}
}

////////////////////////////////////////////////////////////////////////
// Optimized sequential image processing (filtering).
void processSerialOpt(const fipImage& input, fipImage& output, const int* horFilter, const int* verFilter, unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*8);

	const int halfFilterSize = filterSize/2;

	// TODO
}

////////////////////////////////////////////////////////////////////////
// Parallel image processing (filtering) with OMP.
void processOMP(const fipImage& input, fipImage& output, const int* horFilter, const int* verFilter, unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*8);

	// TODO use OMP
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL.
void processSYCL(sycl::queue& q, const fipImage& input, fipImage& output, const int* horFilter, const int* verFilter, unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*8);

	// TODO use SYCL
}


