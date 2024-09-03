#include <cmath>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sycl/sycl.hpp>
#include "Stopwatch.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wignored-attributes"
#endif
#include "FreeImagePlus.h"

////////////////////////////////////////////////////////////////////////
// Pixel type RGBQUAD
// The used color pictures have 4 channels per pixel: blue, green, red, and alpha (opacity)
#ifndef WIN32 
typedef uint32_t COLORREF;
#endif

////////////////////////////////////////////////////////////////////////
// function declarations
void processSerial(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize);
void processSerialOpt(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize);
void processOMP(const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize);
void processSYCL(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize);
void processSYCLvec(sycl::queue& q, const fipImage& input, fipImage& output, const int horFilter[], const int verFilter[], unsigned filterSize);

////////////////////////////////////////////////////////////////////////
// Equality test for images (image borders are ignored)
static bool equals(const fipImage& image1, const fipImage& image2, unsigned filterSize) {
	assert(image1.getWidth() == image2.getWidth() && image1.getHeight() == image2.getHeight() && image1.getImageSize() == image2.getImageSize());
	assert(image1.getBitsPerPixel() == 32);
	const int halfFilterSize = filterSize/2;

	for (unsigned i = halfFilterSize; i < image1.getHeight() - halfFilterSize; i++) {
		COLORREF* row1 = reinterpret_cast<COLORREF*>(image1.getScanLine(i));
		COLORREF* row2 = reinterpret_cast<COLORREF*>(image2.getScanLine(i));

		for (unsigned j = halfFilterSize; j < image1.getWidth() - halfFilterSize; j++) {
			if (row1[j] != row2[j]) return false;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Check and print results
template<typename T>
static void check(const char text[], const T& ref, const T& result, double ts, double tp, unsigned filterSize) {
	const double S = ts/tp;

	std::cout << std::setw(30) << std::left << text;
	std::cout << " in " << std::right << std::setw(7) << std::setprecision(2) << std::fixed << tp << " ms, S = " << S << std::endl;
	std::cout << std::boolalpha << "The two operations produce the same results: " << equals(ref, result, filterSize) << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////
int main(int argc, const char* argv[]) {
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " filter-size input-file-name output-file-name" << std::endl;
		return -1;
	}
	unsigned fSize = atoi(argv[1]);
	if (fSize < 3 || fSize > 11 || (fSize & 1) == 0) {
		std::cerr << "Wrong filter size. Filter size must be odd and between 3 and 11" << std::endl;
		return -2;
	}

	fipImage image;

	// load image
	if (!image.load(argv[2])) {
		std::cerr << "Image not found: " << argv[2] << std::endl;
		return -4;
	}

	constexpr int hFilter3[] = {
		1, 1, 1,
		0, 0, 0,
	   -1,-1,-1,
	};
	constexpr int vFilter3[] = {
		1, 0,-1,
		1, 0,-1,
		1, 0,-1,
	};
	constexpr int hFilter5[] = {
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0,
	};
	constexpr int vFilter5[] = {
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
		0, 1, 0,-1, 0,
	};
	constexpr int hFilter7[] = {
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
	};
	constexpr int vFilter7[] = {
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
		0, 0, 1, 0,-1, 0, 0,
	};
	constexpr int hFilter9[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
	};
	constexpr int vFilter9[] = {
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
		0, 0, 0, 1, 0,-1, 0, 0, 0,
	};
	constexpr int hFilter11[] = {
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	   -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
	    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	};
	constexpr int vFilter11[] = {
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0,
	};

	constexpr bool SaveImages = false;

	// Create an exception handler for asynchronous SYCL exceptions
	auto exception_handler = [](sycl::exception_list e_list) {
		for (std::exception_ptr const& e : e_list) {
			try {
				std::rethrow_exception(e);
			} catch (std::exception const& e) {
#if _DEBUG
				std::cout << "Failure" << std::endl;
#endif
				std::terminate();
			}
		}
	};

	Stopwatch sw;
	const int *hFilter = nullptr;
	const int *vFilter = nullptr;

	switch(fSize) {
	case 3:
		hFilter = hFilter3;
		vFilter = vFilter3;
		break;
	case 5:
		hFilter = hFilter5;
		vFilter = vFilter5;
		break;
	case 7:
		hFilter = hFilter7;
		vFilter = vFilter7;
		break;
	case 9:
		hFilter = hFilter9;
		vFilter = vFilter9;
		break;
	case 11:
		hFilter = hFilter11;
		vFilter = vFilter11;
		break;
	}

	// create output images
	fipImage out1(image), out2(image), out3(image), out4(image), out5(image);

	std::cout << "Edge detection with filter size " << fSize << std::endl << std::endl;

	// process image sequentially and produce out1
	std::cout << "Start sequential execution" << std::endl;
	sw.Start();
	processSerial(image, out1, hFilter, vFilter, fSize);
	sw.Stop();
	const double ts = sw.GetElapsedTimeMilliseconds();
	std::cout << ts << " ms" << std::endl << std::endl;

	// process image sequentially but optimized and produce out2
	std::cout << "Start optimized sequential execution" << std::endl;
	sw.Restart();
	processSerialOpt(image, out2, hFilter, vFilter, fSize);
	sw.Stop();
	const double tsOpt = sw.GetElapsedTimeMilliseconds();
	check("optimized sequential:", out1, out2, ts, tsOpt, fSize);

	// process image in parallel with OMP and produce out3
	std::cout << "Start parallel OMP execution" << std::endl;
	sw.Restart();
	processOMP(image, out3, hFilter, vFilter, fSize);
	sw.Stop();
	check("OpenMP:", out1, out3, tsOpt, sw.GetElapsedTimeMilliseconds(), fSize);

	// process image on GPU with SYCL and produce out4 and out5
	auto selector = sycl::default_selector_v; // The default device selector will select the most performant device.
	sycl::queue q(selector, exception_handler);

	std::cout << "SYCL on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

	try {
		sw.Restart();
		processSYCL(q, image, out4, hFilter, vFilter, fSize);
		q.wait(); // wait until compute tasks on GPU done
		sw.Stop();
		check("GPU:", out1, out4, tsOpt, sw.GetElapsedTimeMilliseconds(), fSize);
	} catch (const std::exception& e) {
		std::cout << "An exception is caught for processSYCL: " << e.what() << std::endl;
	}

	try {
		sw.Restart();
		processSYCLvec(q, image, out5, hFilter, vFilter, fSize);
		q.wait(); // wait until compute tasks on GPU done
		sw.Stop();
		check("GPUvec:", out1, out5, tsOpt, sw.GetElapsedTimeMilliseconds(), fSize);
	} catch (const std::exception& e) {
		std::cout << "An exception is caught for processSYCLvec: " << e.what() << std::endl;
	}

	// save output images
	if constexpr (SaveImages) {
		std::cout << "Save output images" << std::endl;
		std::string outSuffix(argv[3]), outName;

		outName = "Seq_" + outSuffix;
		if (!out1.save(outName.c_str())) {
			std::cerr << "Image not saved: " << outName << std::endl;
		}
		outName = "Opt_" + outSuffix;
		if (!out2.save(outName.c_str())) {
			std::cerr << "Image not saved: " << outName << std::endl;
		}
		outName = "OpenMP_" + outSuffix;
		if (!out3.save(outName.c_str())) {
			std::cerr << "Image not saved: " << outName << std::endl;
		}
		outName = "SYCL_" + outSuffix;
		if (!out4.save(outName.c_str())) {
			std::cerr << "Image not saved: " << outName << std::endl;
		}
		outName = "Vec_" + outSuffix;
		if (!out5.save(outName.c_str())) {
			std::cerr << "Image not saved: " << outName << std::endl;
		}
	}
}
