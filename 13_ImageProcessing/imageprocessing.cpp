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

// SYCL reference manual
// https://registry.khronos.org/SYCL/specs/sycl-1.2.1.pdf 
// https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html

constexpr int BlockSize = 16;

////////////////////////////////////////////////////////////////////////
// compute and return distance 
static uint8_t dist(int dx, int dy) {
	const int d = (int)sqrt(dx*dx + dy*dy);
	return (d < 256) ? d : 255;
}

static RGBQUAD dist(const sycl::int3& dx, const sycl::int3& dy) {
	const sycl::int3 d2 = sycl::mad24(dx, dx, sycl::mul24(dy, dy));
	const sycl::float3 d = sycl::min(sycl::float3(sqrt(d2[0]), sqrt(d2[1]), sqrt(d2[2])), 255);

	return {(uint8_t)d[0], (uint8_t)d[1], (uint8_t)d[2], 255};
}


////////////////////////////////////////////////////////////////////////
// Sequential image processing (filtering).
// It computes the first derivative in x- and y-direction
void processSerial(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;

	// iterate all rows of the output image
	for (unsigned v = halfFilterSize; v < output.getHeight() - halfFilterSize; v++) {
		// iterate all pixels of the current row
		for (unsigned u = halfFilterSize; u < output.getWidth() - halfFilterSize; u++) {
			RGBQUAD inputColor;
			int horColor[3]{};
			int verColor[3]{};
			int filterPos = 0;

			// iterate all filter coefficients
			for (unsigned j = 0; j < filterSize; j++) {
				for (unsigned i = 0; i < filterSize; i++) {
					const int horFilterCoeff = horFilter[filterPos];
					const int verFilterCoeff = verFilter[filterPos];

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

					filterPos++;
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
void processSerialOpt(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;

	// TODO

	// we use the more efficient getScanLine instead of getPixelColor
	const size_t w = input.getScanWidth()/sizeof(RGBQUAD); // number of bytes of an image row
	RGBQUAD* inpRow = reinterpret_cast<RGBQUAD*>(input.getScanLine(halfFilterSize)) + halfFilterSize;
	RGBQUAD* outRow = reinterpret_cast<RGBQUAD*>(output.getScanLine(halfFilterSize)) + halfFilterSize;

	for (unsigned v = halfFilterSize; v < output.getHeight() - halfFilterSize; v++) {
		RGBQUAD* inpCenter = inpRow;	// pointer to input image pixel at filter center
		RGBQUAD* outPos = outRow;	// pointer to output image pixel

		for (unsigned u = halfFilterSize; u < output.getWidth() - halfFilterSize; u++) {
			int horColor[3]{};
			int verColor[3]{};
			int filterPos = 0;
			RGBQUAD* inpPos = inpCenter - (w + 1)*halfFilterSize;

			for (unsigned j = 0; j < filterSize; j++) {
				for (unsigned i = 0; i < filterSize; i++) {
					const RGBQUAD& inColor = *inpPos;
					const int horFilterCoeff = horFilter[filterPos];
					const int verFilterCoeff = verFilter[filterPos];

					horColor[0] += horFilterCoeff*inColor.rgbBlue;
					horColor[1] += horFilterCoeff*inColor.rgbGreen;
					horColor[2] += horFilterCoeff*inColor.rgbRed;

					verColor[0] += verFilterCoeff*inColor.rgbBlue;
					verColor[1] += verFilterCoeff*inColor.rgbGreen;
					verColor[2] += verFilterCoeff*inColor.rgbRed;

					inpPos++;
					filterPos++;
				}
				inpPos += w - filterSize;
			}

			// save result in output image
			RGBQUAD& outColor = *reinterpret_cast<RGBQUAD*>(outPos);
			outColor.rgbBlue = dist(horColor[0], verColor[0]);
			outColor.rgbGreen = dist(horColor[1], verColor[1]);
			outColor.rgbRed = dist(horColor[2], verColor[2]);
			outColor.rgbReserved = 255;

			// next pixel
			inpCenter++;
			outPos++;
		}

		// next row
		inpRow += w;
		outRow += w;
	}
}

////////////////////////////////////////////////////////////////////////
// Parallel image processing (filtering) with OMP.
void __attribute__((noinline)) processOMP(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	// TODO use OMP
	const size_t w = input.getScanWidth()/sizeof(RGBQUAD);
	const int halfFilterSize = filterSize/2;

	#pragma omp parallel for
	//#pragma omp target teams distribute parallel for map(to: input, horFilter, verFilter, filterSize) map(tofrom: output)
	for (int row = halfFilterSize; row < (int)output.getHeight() - halfFilterSize; row++) {
		RGBQUAD* inpCenter = reinterpret_cast<RGBQUAD*>(input.getScanLine(row)) + halfFilterSize;
		RGBQUAD* outPos = reinterpret_cast<RGBQUAD*>(output.getScanLine(row)) + halfFilterSize;

		for (size_t col = halfFilterSize; col < output.getWidth() - halfFilterSize; col++) {
			int horColor[3]{};
			int verColor[3]{};
			int filterPos = 0;
			RGBQUAD* inpPos = inpCenter - (w + 1)*halfFilterSize;

			for (unsigned j = 0; j < filterSize; j++) {
				for (unsigned i = 0; i < filterSize; i++) {
					const RGBQUAD& inpColor = *inpPos;
					const int horFilterCoeff = horFilter[filterPos];
					const int verFilterCoeff = verFilter[filterPos];

					horColor[0] += horFilterCoeff*inpColor.rgbBlue;
					horColor[1] += horFilterCoeff*inpColor.rgbGreen;
					horColor[2] += horFilterCoeff*inpColor.rgbRed;

					verColor[0] += verFilterCoeff*inpColor.rgbBlue;
					verColor[1] += verFilterCoeff*inpColor.rgbGreen;
					verColor[2] += verFilterCoeff*inpColor.rgbRed;

					inpPos++;
					filterPos++;
				}
				inpPos += w - filterSize;
			}

			// save pixel color
			RGBQUAD& outColor = *reinterpret_cast<RGBQUAD*>(outPos);
			outColor.rgbBlue = dist(horColor[0], verColor[0]);
			outColor.rgbGreen = dist(horColor[1], verColor[1]);
			outColor.rgbRed = dist(horColor[2], verColor[2]);
			outColor.rgbReserved = 255;

			// next pixel
			inpCenter++;
			outPos++;
		}
	}
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL.
// Runs slow because of undefined CUDA block size.
void processSYCLv0(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;
	# prepare ranges
	const sycl::range<2> imgRange(output.getHeight(), output.getWidth());
	const sycl::range<2> fltRange(filterSize, filterSize);
	const sycl::range<2> procRange(output.getHeight() - 2*halfFilterSize, output.getWidth() - 2*halfFilterSize);

	sycl::buffer horBuf(horFilter, fltRange);
	sycl::buffer verBuf(verFilter, fltRange);
	sycl::buffer inpBuf((RGBQUAD*)input.getScanLine(0), imgRange);
	sycl::buffer outBuf((RGBQUAD*)output.getScanLine(0), imgRange);

	q.submit([&](sycl::handler& h) {
		sycl::accessor horAcc(horBuf, h, sycl::read_only);
		sycl::accessor verAcc(verBuf, h, sycl::read_only);
		sycl::accessor inpAcc(inpBuf, h, sycl::read_only);
		sycl::accessor outAcc(outBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(procRange, [=](auto i) {
			const sycl::id<2> outPos(i[0] + halfFilterSize, i[1] + halfFilterSize);
			sycl::int4 horResult{};
			sycl::int4 verResult{};

			for (unsigned y = 0; y < filterSize; y++) {
				for (unsigned x = 0; x < filterSize; x++) {
					const sycl::id<2> fltPos(y, x);
					const sycl::id<2> inpPos(i[0] + y, i[1] + x);
					const RGBQUAD& inpColor = inpAcc[inpPos];
					const int horFilterCoeff = horAcc[fltPos];
					const int verFilterCoeff = verAcc[fltPos];

					horResult[0] += horFilterCoeff*inpColor.rgbBlue;
					horResult[1] += horFilterCoeff*inpColor.rgbGreen;
					horResult[2] += horFilterCoeff*inpColor.rgbRed;

					verResult[0] += verFilterCoeff*inpColor.rgbBlue;
					verResult[1] += verFilterCoeff*inpColor.rgbGreen;
					verResult[2] += verFilterCoeff*inpColor.rgbRed;
				}
			}

			// save output pixel color
			RGBQUAD& outColor = outAcc[outPos];
			outColor.rgbBlue = dist(horResult[0], verResult[0]);
			outColor.rgbGreen = dist(horResult[1], verResult[1]);
			outColor.rgbRed = dist(horResult[2], verResult[2]);
			outColor.rgbReserved = 255;
		});
	});
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL and nd_range.
void processSYCL(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;
	const sycl::range<2> imgRange(output.getHeight(), output.getWidth());
	const sycl::range<2> fltOffset(halfFilterSize, halfFilterSize);
	const sycl::range<2> fltRange(filterSize, filterSize);
	const sycl::range<2> procRange = imgRange - fltOffset;
	const sycl::nd_range<2> ndr(imgRange, {BlockSize, BlockSize});	// workgroup size: 16x16

	# prepare buffers, which are accesable for all threads
	sycl::buffer horBuf(horFilter, fltRange);
	sycl::buffer verBuf(verFilter, fltRange);
	sycl::buffer inpBuf((RGBQUAD*)input.getScanLine(0), imgRange);
	sycl::buffer outBuf((RGBQUAD*)output.getScanLine(0), imgRange);

	q.submit([&](sycl::handler& h) {
		sycl::accessor horAcc(horBuf, h, sycl::read_only);
		sycl::accessor verAcc(verBuf, h, sycl::read_only);
		sycl::accessor inpAcc(inpBuf, h, sycl::read_only);
		sycl::accessor outAcc(outBuf, h, sycl::write_only, sycl::no_init); #no_init -> tells sycl not to do initialization

		h.parallel_for(ndr, [=](auto ii) {
			const sycl::id<2> outPos = ii.get_global_id();
			
			// TODO use SYCL
			if (fltOffset[0] <= outPos[0] && outPos[0] < procRange[0] && fltOffset[1] <= outPos[1] && outPos[1] < procRange[1]) {
				const sycl::id<2> inpBase = outPos - fltOffset; 

				sycl::int3 horResult{};
				sycl::int3 verResult{};

				for (unsigned y = 0; y < filterSize; y++) {
					for (unsigned x = 0; x < filterSize; x++) {
						const sycl::id<2> fltPos(y, x);
						const int horFilterCoeff = horAcc[fltPos];
						const int verFilterCoeff = verAcc[fltPos];
						const RGBQUAD& inpColor = inpAcc[inpBase + fltPos];

						horResult[0] += horFilterCoeff*inpColor.rgbBlue;
						horResult[1] += horFilterCoeff*inpColor.rgbGreen;
						horResult[2] += horFilterCoeff*inpColor.rgbRed;

						verResult[0] += verFilterCoeff*inpColor.rgbBlue;
						verResult[1] += verFilterCoeff*inpColor.rgbGreen;
						verResult[2] += verFilterCoeff*inpColor.rgbRed;
					}
				}

				// save output pixel color
				RGBQUAD& outColor = outAcc[outPos];
				outColor.rgbBlue = dist(horResult[0], verResult[0]);
				outColor.rgbGreen = dist(horResult[1], verResult[1]);
				outColor.rgbRed = dist(horResult[2], verResult[2]);
				outColor.rgbReserved = 255;
			}
		});
	});
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL and nd_range.
// Use vectorization to process all three color channels in parallel.
void processSYCLvec(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	const int halfFilterSize = filterSize/2;
	const sycl::range<2> imgRange(output.getHeight(), output.getWidth());
	const sycl::range<2> fltOffset(halfFilterSize, halfFilterSize);
	const sycl::range<2> fltRange(filterSize, filterSize);
	const sycl::range<2> procRange = imgRange - fltOffset;
	const sycl::nd_range<2> ndr(imgRange, {BlockSize, BlockSize});	// workgroup size: 16x16

	sycl::buffer horBuf(horFilter, fltRange);
	sycl::buffer verBuf(verFilter, fltRange);
	sycl::buffer inpBuf((RGBQUAD*)input.getScanLine(0), imgRange);
	sycl::buffer outBuf((RGBQUAD*)output.getScanLine(0), imgRange);

	q.submit([&](sycl::handler& h) {
		sycl::accessor horAcc(horBuf, h, sycl::read_only);
		sycl::accessor verAcc(verBuf, h, sycl::read_only);
		sycl::accessor inpAcc(inpBuf, h, sycl::read_only);
		sycl::accessor outAcc(outBuf, h, sycl::write_only, sycl::no_init);

		h.parallel_for(ndr, [=](auto ii) {
			const sycl::id<2> outPos = ii.get_global_id();
			
			// TODO use SYCL and vectorization
			if (fltOffset[0] <= outPos[0] && outPos[0] < procRange[0] && fltOffset[1] <= outPos[1] && outPos[1] < procRange[1]) {
				const sycl::id<2> inpBase = outPos - fltOffset; 

				# special sycl vector type for 3D vectors
				sycl::int3 horResult{};
				sycl::int3 verResult{};

				for (unsigned y = 0; y < filterSize; y++) {
					for (unsigned x = 0; x < filterSize; x++) {
						const sycl::id<2> fltPos(y, x);
						const int horFilterCoeff = horAcc[fltPos];
						const int verFilterCoeff = verAcc[fltPos];
						const RGBQUAD& inp = inpAcc[inpBase + fltPos];
						const sycl::int3 inpColor(inp.rgbBlue, inp.rgbGreen, inp.rgbRed);

						# here is different to before
						horResult += horFilterCoeff*inpColor;
						verResult += verFilterCoeff*inpColor;
					}
				}

				// save output pixel color
				outAcc[outPos] = dist(horResult, verResult);
			}
		});
	});
}

////////////////////////////////////////////////////////////////////////
// GPU processing with SYCL and using image
// doesn't work yet
// sampled_image and unsampled_image aren't supported yet
// needs: export SYCL_PI_CUDA_ENABLE_IMAGE_SUPPORT
void processSYCLimg(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize) {
	assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
	assert(input.getBitsPerPixel() == sizeof(RGBQUAD)*CHAR_BIT);

	using co = sycl::image_channel_order;
	using ct = sycl::image_channel_type;

	const int halfFilterSize = filterSize/2;
	const sycl::range<2> imgRange(output.getHeight(), output.getWidth());
	const sycl::range<2> fltOffset(halfFilterSize, halfFilterSize);
	const sycl::range<2> fltRange(filterSize, filterSize);
	const sycl::nd_range<2> ndr(imgRange, {BlockSize, BlockSize});	// workgroup size: 16x16

	sycl::buffer horBuf(horFilter, fltRange);
	sycl::buffer verBuf(verFilter, fltRange);
	sycl::buffer inpBuf((RGBQUAD*)input.getScanLine(0), imgRange);
	sycl::buffer outBuf((RGBQUAD*)output.getScanLine(0), imgRange);
	#instead of buffer -> sycl::image, but not sure if backend supports it
	sycl::image<2> inpImg(co::rgba, ct::unsigned_int32, imgRange);
	sycl::image<2> outImg(co::rgba, ct::unsigned_int32, imgRange);

	q.submit([&](sycl::handler& h) {
		sycl::accessor inpAcc(inpBuf, h, sycl::read_only);
      	sycl::accessor<sycl::uint4, 2, sycl::access::mode::write, sycl::access::target::image> imgAcc(inpImg, h);

      	h.parallel_for(imgRange, [=](sycl::item<2> item) {
			auto coords = sycl::int2(item[1], item[0]);

			const RGBQUAD& px = inpAcc[item];
			sycl::uint4 pixel(px.rgbBlue, px.rgbGreen, px.rgbRed, px.rgbReserved);
			imgAcc.write(coords, pixel);
		});
	});

	q.submit([&](sycl::handler& h) {
		sycl::accessor horAcc(horBuf, h, sycl::read_only);
		sycl::accessor verAcc(verBuf, h, sycl::read_only);
		auto inpAcc = inpImg.get_access<sycl::int4, sycl::access_mode::read>(h);
		auto outAcc = outImg.get_access<sycl::int4, sycl::access_mode::write>(h);
		sycl::sampler smpl(sycl::coordinate_normalization_mode::unnormalized, sycl::addressing_mode::clamp, sycl::filtering_mode::nearest);

		h.parallel_for(ndr, [=](auto ii) {
			const sycl::id<2> outPos = ii.get_global_id();
			const sycl::int2 outCoord(outPos[1], outPos[0]);
			
			sycl::int4 horResult{};
			sycl::int4 verResult{};

			for (unsigned y = 0; y < filterSize; y++) {
				for (unsigned x = 0; x < filterSize; x++) {
					const sycl::id<2> fltPos(y, x);
					const int horFilterCoeff = horAcc[fltPos];
					const int verFilterCoeff = verAcc[fltPos];
					const sycl::int2 inpCoord((int)outPos[1] - (int)fltOffset[1] + (int)fltPos[1], (int)outPos[0] - (int)fltOffset[0] + (int)fltPos[0]);
					const sycl::int4 inpColor = inpAcc.read(inpCoord, smpl);

					horResult += horFilterCoeff*inpColor;
					verResult += verFilterCoeff*inpColor;
				}
			}

			// save output pixel color
			outAcc.write(outCoord, sycl::int4(
				dist(horResult[0], verResult[0]),
				dist(horResult[1], verResult[1]),
				dist(horResult[2], verResult[2]),
				255
			));
		});
	});

	q.submit([&](sycl::handler& h) {
		sycl::accessor outAcc(outBuf, h, sycl::write_only);
      	sycl::accessor<sycl::uint4, 2, sycl::access::mode::read, sycl::access::target::image> imgAcc(outImg, h);

      	h.parallel_for(imgRange, [=](sycl::item<2> item) {
			auto coords = sycl::int2(item[1], item[0]);

			const sycl::uint4& pixel = imgAcc.read(coords);
			outAcc[item] = RGBQUAD((uint8_t)pixel[0], (uint8_t)pixel[1], (uint8_t)pixel[2], (uint8_t)pixel[3]);
		});
	});
}

