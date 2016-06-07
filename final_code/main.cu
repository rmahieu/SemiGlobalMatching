#include <cstdio>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "gpuKernels.cu"


// GLOBAL VARS
bool	  verbose = false;		// flag for outputting information during runtime
int		  dmax = 64;			// maximum pixel offset
int		  desired_width = 450;	// width that input image is scaled to
const int pxPerThread = 4;		// pixels per thread (used for certain kernels)
const int numDirs = 8;			// number of directions to process (<=8)

// loads BGR image onto GPU memory
// returns device pointers to BGR and grayscale images
int loadImage(std::string filename, DeviceImage *d_img)
{
	// load BGR image from file 
	cv::Mat input_bgr;
	input_bgr = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if (!input_bgr.data) {
		std::cerr << "Failed to open '" << filename << "'" << std::endl;
		return -1;
	} else {
		if (verbose)
			std::cout << "Successfully read '" << filename << "'" << std::endl;
	}

	// scale image if necessary
	if (input_bgr.cols > desired_width) {
		const double scale_factor = (double)desired_width / (double)input_bgr.cols;
		cv::resize(input_bgr, input_bgr, cv::Size(), scale_factor, scale_factor, CV_INTER_AREA);
	}

	// write scaled image to file for reference
	//size_t lastindex = filename.find_last_of(".");
	//std::string rawname = filename.substr(0, lastindex);
	//cv::imwrite(rawname + "_scaled.png", input_bgr);

	// set image dimensions
	d_img->rows = input_bgr.rows;
	d_img->cols = input_bgr.cols;

	// get pointer to host image
	uchar3 *h_bgr = (uchar3 *)input_bgr.ptr<uchar>(0);

	// allocate memory on GPU
	const int numPixels = input_bgr.cols * input_bgr.rows;

	d_img->bgr = 0;
	cudaMalloc((void**)&d_img->bgr, numPixels * sizeof(uchar3));
	if (d_img->bgr == 0)
	{
		std::cerr << "Failed to allocate memory for input image" << std::endl;
		return -1;
	}

	// copy BGR image to GPU
	cudaMemcpy(d_img->bgr, &h_bgr[0], numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

	return 0;
}


// copies grayscale image to host then writes to file
void writeImage(std::string filename, DeviceImage d_img)
{
	const int numPixels = d_img.rows * d_img.cols;

	// copy image from GPU
	float *h_img = (float *)malloc(numPixels * sizeof(float));
	cudaMemcpy(&h_img[0], d_img.array, numPixels * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpyFromArray(&h_img[0], d_img.array, 0, 0, numPixels * sizeof(float), cudaMemcpyDeviceToHost);

	// write image to file
	cv::Mat output(d_img.rows, d_img.cols, CV_32FC1, (void*)h_img);
	cv::imwrite(filename, output);

	// free memory from host
	free(h_img);

	if (verbose)
		std::cout << "Successfully wrote '" << filename << "'" << std::endl;
}


// debugging function to read/print some amount of data from device
void readDeviceData(float * dev_ptr, uint start_pos, uint num_elements)
{
	float * output = (float*)malloc(num_elements * sizeof(float));
	cudaMemcpy(output, &dev_ptr[start_pos], num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << std::endl;
	for (uint i = 0; i < num_elements; i++) {
		std::cout << output[i] << " ";
	}
	std::cout << std::endl;

	free(output);
}


// write disparity map to file
void writeDmap(float * d_dmap, int height, int width, int dmax, std::string inputname = "dmap.png")
{
	const int numPixels = height * width;

	// copy image from GPU
	float * h_dmap = (float *)malloc(numPixels * sizeof(float));
	cudaMemcpy(h_dmap, d_dmap, numPixels * sizeof(float), cudaMemcpyDeviceToHost);

	// write image to file
	cv::Mat output(height, width, CV_32FC1, (void*)h_dmap);
	output.convertTo(output, -1, 255.0 / (double)dmax);		// scale appropriately (dmax -> 255)

	// apply 3x3 median filter to reduce noise
	//cv::medianBlur(output, output, 3);	

	// get output filename
	size_t lastindex = inputname.find_last_of(".");
	std::string rawname = inputname.substr(0, lastindex);
	cv::imwrite(rawname + "_dmap.png", output);

	// free memory from host
	free(h_dmap);

	if (verbose)
		std::cout << "\n\nSuccessfully wrote depth map to file" << std::endl;
}

char* cmdOption(char ** begin, char ** end, const std::string & option)
{
	char ** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
		return *itr;
	
	return 0;
}

bool cmdPresent(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

int main(int argc, char **argv)
{
	// parse command line arguments
	if (argc == 1) {
		std::cout << "USAGE: semi_global_matching.exe [-v] [-d dmax] [-w scaled_img_width] left_img right_img" << std::endl;
		return 1;
	}

	if (cmdPresent(argv, argv + argc, "-v")) {
		verbose = true;
	}

	if (cmdPresent(argv, argv + argc, "-d")) {
		dmax = atoi(cmdOption(argv, argv + argc, "-d"));
		if (dmax <= 0) {
			std::cerr << "ERROR : dmax must be greater than zero!" << std::endl;
			return 1;
		}
	}

	if (cmdPresent(argv, argv + argc, "-w")) {
		desired_width = atoi(cmdOption(argv, argv + argc, "-w"));
		if (desired_width <= 0) {
			std::cerr << "ERROR : image width must be greater than zero!" << std::endl;
			return 1;
		}
	}

	// initialize GPU timer for performance testing
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start);	// start timer

	// initialize CPU timer for overall timing
	clock_t cpu_start, cpu_stop;
	double totalTime;
	cpu_start = clock();

	// initialize base image (bImg) and match image (mImg)
	DeviceImage bImg, mImg;

	// read input images
	if (loadImage(argv[argc - 2], &bImg))	return -1;
	if (loadImage(argv[argc - 1], &mImg))	return -1;

	// convert from RGBA to grayscale
	if (verbose)	std::cout << "Converting image to grayscale... ";
	getGrayscaleImage<pxPerThread>(&bImg);
	getGrayscaleImage<pxPerThread>(&mImg);
	if (verbose)	std::cout << "DONE" << std::endl;

	// bind grayscale images to textures
	if (verbose)	std::cout << "Initializing textures... ";
	intializeTextures(bImg, mImg);
	if (verbose)	std::cout << "DONE" << std::endl;

	// initialize cost matrix container
	CostMatrices cm;
	cm.width = bImg.cols;
	cm.height = bImg.rows;
	cm.depth = dmax;

	// get initial costs
	if (verbose)	std::cout << "Getting initial costs... ";
	getInitialCosts(cm);
	if (verbose)	std::cout << "DONE" << std::endl;

	// get penalties for each direction
	if (verbose)	std::cout << "Gathering penalty values... ";
	float * d_P1;
	float * d_P2;
	getPenaltyValues<numDirs>(&d_P1, &d_P2);
	if (verbose)	std::cout << "DONE" << std::endl;

	// do path traversal to get energy matrices
	if (verbose)	std::cout << "Performing path traversal... ";
	doPathTraversal<numDirs>(cm, d_P1, d_P2);
	if (verbose)	std::cout << "DONE" << std::endl;

	// find directions weightings
	if (verbose)	std::cout << "Gathering path weightings... ";
	float * d_weights;
	getDirectionWeightings<numDirs>(&d_weights);
	if (verbose)	std::cout << "DONE" << std::endl;

	// get final cost matrix
	if (verbose)	std::cout << "Calculating final cost matrix... ";
	getFinalCosts<numDirs>(cm, d_weights);
	if (verbose)	std::cout << "DONE" << std::endl;

	// argmin cost matrix over d to find disparities
	if (verbose)	std::cout << "Calculating disparities... ";
	float * d_disparities;
	getDisparities(cm, &d_disparities);
	if (verbose)	std::cout << "DONE" << std::endl;

	// find disparity map for match image (redo all calculations...)
	if (verbose)	std::cout << "Calculating match image's disparity map... ";
	CostMatrices cm_rev = cm;
	getInitialCosts_reverse(cm_rev);
	doPathTraversal<numDirs>(cm_rev, d_P1, d_P2);
	getFinalCosts<numDirs>(cm_rev, d_weights);

	float * d_disparities_rev;
	getDisparities(cm_rev, &d_disparities_rev);
	if (verbose)	std::cout << "DONE" << std::endl;

	// refine base map by detecting occlusions
	if (verbose)	std::cout << "Refining disparity map... ";
	refineDisparityMap<pxPerThread>(d_disparities, d_disparities_rev, cm.height, cm.width);
	if (verbose)	std::cout << "DONE" << std::endl;
	
	// perform median filtering to reduce noise
	if (verbose)	std::cout << "Applying median filter... ";
	doMedianFiltering(&d_disparities, cm.height, cm.width);
	if (verbose)	std::cout << "DONE" << std::endl;

	// write disparity map to file
	writeDmap(d_disparities, cm.height, cm.width, cm.depth, argv[argc - 2]);

	// write grayscale images to file
	//writeImage("output1.png", bImg);	// this is currently broken....
	//writeImage("output2.png", mImg);	//    does cudaArray pointer die after binding texture?

	// free device memory
	cudaFree(bImg.array);
	cudaFree(mImg.array);
	cudaFree(d_disparities);
	cudaFree(d_disparities_rev);
	cudaFree(d_P1);
	cudaFree(d_P2);
	cudaFree(d_weights);

	// record elapsed time
	cudaEventRecord(gpu_stop);
	cudaEventSynchronize(gpu_stop);
	float gpu_milliseconds = 0;
	cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop);
	std::cout << "\nGPU time : " << gpu_milliseconds << "ms" << std::endl;

	cpu_stop = clock();
	totalTime = (cpu_stop - cpu_start) / (double)CLOCKS_PER_SEC;
	std::cout << "Total time : " << totalTime * 1000.0 << "ms" << std::endl;


	return 0;
}






