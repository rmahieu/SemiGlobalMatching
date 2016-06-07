/*
*  GPU kernel functions for semi-global matching algorithm
*/

#ifndef _SEMIGLOBALMATCHING_KERNEL_CU_
#define _SEMIGLOBALMATCHING_KERNEL_CU_


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <math.h>

typedef unsigned char uchar;
typedef unsigned int  uint;

struct DeviceImage
{
	int rows;
	int cols;

	uchar3     *bgr;	// pointer to BGR image on device
	float	   *gray;	// pointer to grayscale image on device
	cudaArray  *array;	// pointer to cudaArray we use for texture
};


struct CostMatrices {
	int width;			// width of matrices (x)
	int height;			// height of matrices (y)
	int depth;			// depth of matrices (z)

	float * C;			// initial costs matrix
	float * E[8];		// energy matrix for each direction
	float * S;			// weighted and summed energy matrices
};


// global texture references for bound image data
texture<float, cudaTextureType2D, cudaReadModeElementType> bTex;
texture<float, cudaTextureType2D, cudaReadModeElementType> mTex;


// check if index is in image bounds
static __device__ __forceinline__ bool in_img(int x, int y, int rows, int cols)
{
	return x >= 0 && x < cols && y >= 0 && y < rows;
}


// convert BGR image to grayscale
template<int pxPerThread>
__global__ void bgr_to_grayscale(DeviceImage img)
{
	// get global index within image
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = pxPerThread * (blockIdx.y*blockDim.y + threadIdx.y);

	// loop over the number of pixels each thread is handling
	for (size_t i = 0; i < pxPerThread; ++i)
	{
		// get BGR pixel values
		uchar3 p;
		if (in_img(x, y + i, img.rows, img.cols))
			p = img.bgr[(y + i) * img.cols + x];
		else
			return;

		// calculate grayscale value
		float g = 0.298839f*(float)p.z + 0.586811f*(float)p.y + 0.114350f*(float)p.x;

		// set grayscale value in image
		if (in_img(x, y + i, img.rows, img.cols))
			img.gray[(y + i) * img.cols + x] = (g >= 255.f ? 255.f : g);
	}
}


// Call grayscale conversion kernel
template<int pxPerThread>
int getGrayscaleImage(DeviceImage * img)
{
	// allocate memory
	img->gray = 0;
	cudaMalloc((void**)&img->gray, img->rows * img->cols * sizeof(float));
	if (img->gray == 0)
	{
		std::cerr << "Failed to allocate memory" << std::endl;
		return -1;
	}

	// define block and grid sizes
	dim3 block_size(32, 8);
	dim3 grid_size(0, 0);
	grid_size.x = (img->cols + block_size.x - 1) / block_size.x;
	grid_size.y = (img->rows + pxPerThread * block_size.y - 1) / (pxPerThread * block_size.y);

	// call kernel
	bgr_to_grayscale<pxPerThread> <<<grid_size, block_size >>>(*img);

	// can now free the BGR device memory
	cudaFree(img->bgr);

	return 0;
}


// Calculate initial costs of the images and store into cost matrix C
__global__ void initial_costs(CostMatrices cm)
{
	const int h = cm.height;
	const int w = cm.width;
	const int dmax = cm.depth;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int d = blockIdx.z * blockDim.z + threadIdx.z;

	// compute normalized coords
	float u = x / (float)w;
	float v = y / (float)h;
	float offset = d / (float)w;

	// fetch values from texture
	float baseVal = tex2D(bTex, u, v);
	float matchVal = tex2D(mTex, u - offset, v);

	// compute C(x,y,d) and write to matrix
	if (x < w && y < h && d < dmax)
		cm.C[(x + (y * w)) * dmax + d] = fabs(baseVal - matchVal);
}


// Bind images on device to textures
void intializeTextures(DeviceImage bImg, DeviceImage mImg)
{	
	// allocate 2D cuda arrays
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&bImg.array, &channelDesc, bImg.cols, bImg.rows);
	cudaMallocArray(&mImg.array, &channelDesc, mImg.cols, mImg.rows);

	// copy images into 2D cuda arrays
	cudaMemcpyToArray(bImg.array, 0, 0, bImg.gray, bImg.rows * bImg.cols * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(mImg.array, 0, 0, mImg.gray, mImg.rows * mImg.cols * sizeof(float), cudaMemcpyDeviceToDevice);

	// set texture parameters
	bTex.normalized = true;                       // access with normalized texture coordinates
	bTex.filterMode = cudaFilterModeLinear;       // linear interpolation
	bTex.addressMode[0] = cudaAddressModeBorder;   // OOB texture calls return clamped edge value
	bTex.addressMode[1] = cudaAddressModeBorder;

	mTex.normalized = true;                      
	mTex.filterMode = cudaFilterModeLinear;
	mTex.addressMode[0] = cudaAddressModeBorder;
	mTex.addressMode[1] = cudaAddressModeBorder;

	// bind arrays to texture
	cudaBindTextureToArray(bTex, bImg.array, channelDesc);
	cudaBindTextureToArray(mTex, mImg.array, channelDesc);

	// can now free original grayscale images from linear memory
	cudaFree(bImg.gray);
	cudaFree(mImg.gray);
}


// Call initial cost calculation kernel
int getInitialCosts(CostMatrices& cm)
{
	const int h = cm.height;
	const int w = cm.width;
	const int dmax = cm.depth;

	// allocate memory on device for the cost matrix
	cm.C = 0;
	cudaMalloc((void**)&cm.C, h * w * dmax * sizeof(float));
	if (cm.C == 0)
	{
		std::cerr << "Failed to allocate memory for initial cost matrix!" << std::endl;
		return -1;
	}

	// define block and grid sizes
	dim3 block_size(4, 4, 32);
	dim3 grid_size(0, 0, 0);
	grid_size.x = (w + block_size.x - 1) / block_size.x;
	grid_size.y = (h + block_size.y - 1) / (block_size.y);
	grid_size.z = (dmax + block_size.z - 1) / (block_size.z);

	// call kernel on GPU
	initial_costs<<<grid_size, block_size>>>(cm);

	return 0;
}


template<int numTestDirections>
int getPenaltyValues(float ** P1, float **  P2)
{
	// use values given in paper
	float h_P1[numTestDirections] = { 22.02, 22.02, 17.75, 17.75, 14.93, 14.93, 10.67, 10.67 };
	float h_P2[numTestDirections] = { 82.79, 82.79, 80.87, 80.87, 23.30, 23.30, 28.80, 28.80 };

	// copy values to device
	*P1 = 0;
	*P2 = 0;
	cudaMalloc((void**)P1, numTestDirections * sizeof(float));
	cudaMalloc((void**)P2, numTestDirections * sizeof(float));

	if (*P1 == 0 || *P2 == 0)
	{
		std::cerr << "Failed to allocate memory for penalty values" << std::endl;
		return 1;
	}
	cudaMemcpy(*P1, h_P1, numTestDirections * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*P2, h_P2, numTestDirections * sizeof(float), cudaMemcpyHostToDevice);

	return 0;
}




// stuff
__global__ void path_traversal(CostMatrices cm, float * P1, float * P2)
{
	const int h = cm.height;
	const int w = cm.width;
	const int dmax = cm.depth;

	const int d = threadIdx.x;	// thread id corresponds to d value
	const int dir = blockIdx.x;

	int dx = 0;
	int dy = 0;

	int x0 = 0;
	int y0 = 0;

	int x_ = 0;
	int y_ = 0;

	int maxItr = 0;

	// each block works on a different direction
	switch (dir) {


	// HORIZONTAL FORWARD DIRECTION 
	case 0:

		// (forward direction)
		dx = 1;
		x0 = 0;

		// do edge case
		for (int y = 0; y < h; y++) {
			cm.E[dir][(x0 + (y * w)) * dmax + d] = cm.C[(x0 + (y * w)) * dmax + d];
		}

		// E(x,y,d)
		for (int x = x0 + dx; x < w; x += dx) {

			// wait for threads to sync up
			__syncthreads();

			for (int y = 0; y < h; y++) {

				float term1 = cm.E[dir][(x - dx + (y * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + (y * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + (y * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + (y * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + (y * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}
		}

		break;

	// HORIZONTAL REVERSE DIRECTION
	case 1:

		// (reverse direction)
		dx = -1;
		x0 = w - 1;

		// do edge case
		for (int y = 0; y < h; y++) {
			cm.E[dir][(x0 + (y * w)) * dmax + d] = cm.C[(x0 + (y * w)) * dmax + d];
		}

		// E(x,y,d)
		for (int x = x0 + dx; x >= 0; x += dx) {

			// wait for threads to sync up
			__syncthreads();

			for (int y = 0; y < h; y++) {

				float term1 = cm.E[dir][(x - dx + (y * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + (y * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + (y * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + (y * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + (y * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}
		}

		break;

	// VERTICAL BOTTOM->TOP
	case 2:

		// (forward direction)
		dy = 1;
		y0 = 0;

		// do edge case
		for (int x = 0; x < w; x++) {
			cm.E[dir][(x + (y0 * w)) * dmax + d] = cm.C[(x + (y0 * w)) * dmax + d];
		}

		// E(x,y,d)
		for (int y = y0 + dy; y < h; y += dy) {

			// wait for threads to sync up
			__syncthreads();

			for (int x = 0; x < w; x++) {

				float term1 = cm.E[dir][(x + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}
		}

		break;

	// VERTICAL TOP->BOTTOM
	case 3:

		// (reverse direction)
		dy = -1;
		y0 = h - 1;

		// do edge case
		for (int x = 0; x < w; x++) {
			cm.E[dir][(x + (y0 * w)) * dmax + d] = cm.C[(x + (y0 * w)) * dmax + d];
		}

		// E(x,y,d)
		for (int y = y0 + dy; y >= 0; y += dy) {

			// wait for threads to sync up
			__syncthreads();

			for (int x = 0; x < w; x++) {

				float term1 = cm.E[dir][(x + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}
		}

		break;

	// DIAGONAL TOPLEFT->BOTTOMRIGHT
	case 4:

		// top left -> bottom right
		dx = 1;
		dy = -1;

		x0 = 0;
		y0 = h - 1;

		// do top row edge case
		for (int x = x0; x < w; x++) {
			cm.E[dir][(x + (y0 * w)) * dmax + d] = cm.C[(x + (y0 * w)) * dmax + d];
		}

		// do first col edge case
		for (int y = y0; y >= 0; y--) {
			cm.E[dir][(x0 + (y * w)) * dmax + d] = cm.C[(x0 + (y * w)) * dmax + d];
		}

		maxItr = (w >= h) ? h : w;
		y_ = y0;
		x_ = x0;

		for (int itr = 1; itr < maxItr; itr++) {

			// wait for threads to sync up
			__syncthreads();

			// incremement starting point
			x_ += dx;
			y_ += dy;

			// iterate over current row
			int y = y_;
			for (int x = x_; x < w; x++) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

			// iterate over current col
			int x = x_;
			for (int y = y_; y >= 0; y--) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

		}

		break;

	// DIAGONAL BOTTOMRIGHT->TOPLEFT
	case 5:

		// bottom right -> top left
		dx = -1;
		dy = 1;

		x0 = w - 1;
		y0 = 0;

		// do bottom row edge case
		for (int x = x0; x >= 0; x--) {
			cm.E[dir][(x + (y0 * w)) * dmax + d] = cm.C[(x + (y0 * w)) * dmax + d];
		}

		// do last col edge case
		for (int y = y0; y < h; y++) {
			cm.E[dir][(x0 + (y * w)) * dmax + d] = cm.C[(x0 + (y * w)) * dmax + d];
		}

		maxItr = (w >= h) ? h : w;
		y_ = y0;
		x_ = x0;

		for (int itr = 1; itr < maxItr; itr++) {

			// wait for threads to sync up
			__syncthreads();

			// incremement starting point
			x_ += dx;
			y_ += dy;

			// iterate over current row
			int y = y_;
			for (int x = x_; x >= 0; x--) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

			// iterate over current col
			int x = x_;
			for (int y = y_; y < h; y++) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

		}

		break;

	// DIAGONAL BOTTOMLEFT->TOPRIGHT
	case 6:

		// bottom left -> top right
		dx = 1;
		dy = 1;

		x0 = 0;
		y0 = 0;

		// do row edge case
		for (int x = x0; x < w; x++) {
			cm.E[dir][(x + (y0 * w)) * dmax + d] = cm.C[(x + (y0 * w)) * dmax + d];
		}

		// do col edge case
		for (int y = y0; y < h; y++) {
			cm.E[dir][(x0 + (y * w)) * dmax + d] = cm.C[(x0 + (y * w)) * dmax + d];
		}

		maxItr = (w >= h) ? h : w;
		y_ = y0;
		x_ = x0;

		for (int itr = 1; itr < maxItr; itr++) {

			// wait for threads to sync up
			__syncthreads();

			// incremement starting point
			x_ += dx;
			y_ += dy;

			// iterate over current row
			int y = y_;
			for (int x = x_; x < w; x++) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

			// iterate over current col
			int x = x_;
			for (int y = y_; y < h; y++) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

		}

		break;


	// DIAGONAL TOPRIGHT->BOTTOMLEFT
	case 7:

		// top right -> bottom left
		dx = -1;
		dy = -1;

		x0 = w - 1;
		y0 = h - 1;

		// do row edge case
		for (int x = x0; x >= 0; x--) {
			cm.E[dir][(x + (y0 * w)) * dmax + d] = cm.C[(x + (y0 * w)) * dmax + d];
		}

		// do col edge case
		for (int y = y0; y >= 0; y--) {
			cm.E[dir][(x0 + (y * w)) * dmax + d] = cm.C[(x0 + (y * w)) * dmax + d];
		}

		maxItr = (w >= h) ? h : w;
		y_ = y0;
		x_ = x0;

		for (int itr = 1; itr < maxItr; itr++) {

			// wait for threads to sync up
			__syncthreads();

			// incremement starting point
			x_ += dx;
			y_ += dy;

			// iterate over current row
			int y = y_;
			for (int x = x_; x >= 0; x--) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

			// iterate over current col
			int x = x_;
			for (int y = y_; y >= 0; y--) {

				float term1 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d];

				// handle d edge cases
				float term2 = (d == 0) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d - 1] + P1[dir];
				float term3 = (d == dmax - 1) ? term1 : cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + d + 1] + P1[dir];

				// get minimum of all last d values
				float term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax] + P2[dir];
				for (int i = 1; i < dmax; i++) {
					float test_term4 = cm.E[dir][(x - dx + ((y - dy) * w)) * dmax + i] + P2[dir];
					if (test_term4 < term4)
						term4 = test_term4;
				}

				// get minimum over mimization terms
				float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

				// calculate E value
				cm.E[dir][(x + (y * w)) * dmax + d] = cm.C[(x + (y * w)) * dmax + d] + minVal;
			}

		}

		break;
	}

}

template<int numTestDirections>
int doPathTraversal(CostMatrices& cm, float * P1, float * P2)
{
	const int h = cm.height;
	const int w = cm.width;
	const int dmax = cm.depth;

	// allocate memory
	for (int i = 0; i < numTestDirections; i++) {
		cm.E[i] = 0;
		cudaMalloc((void**)&cm.E[i], h * w * dmax * sizeof(float));
		if (cm.E[i] == 0) {
			std::cerr << "ERROR: E[" << i << "] failed to allocate memory" << std::endl;
			return 1;
		}
	}

	dim3 block_size(dmax);
	dim3 grid_size(numTestDirections);

	path_traversal <<<grid_size, block_size >>>(cm, P1, P2);


	// can now free initial costs matrix
	cudaFree(cm.C);

	return 0;
}


// get weightings for each direction
template<int numTestDirections>
int getDirectionWeightings(float ** d_weights)
{
	// for now we'll just use 1/numdirections
	
	// set values on host
	//float h_weights[numTestDirections] = { 0.01, 0.01, 0.01, 0.01, 0.9f, 0.9f, 0.9f, 0.9f };
	//float h_weights[numTestDirections] = { 0.96f, 0.96f, 0.98f, 0.98f, 0.06f, 0.06f, 0.27f, 0.27f };

	float h_weights[numTestDirections];
	std::fill_n(h_weights, numTestDirections, 1.f / (float)numTestDirections);

	// copy values to device
	*d_weights = 0;
	cudaMalloc((void**)d_weights, numTestDirections * sizeof(float));
	if (*d_weights == 0)
	{
		std::cerr << "Failed to allocate memory for direction weights" << std::endl;
		return 1;
	}
	cudaMemcpy(*d_weights, h_weights, numTestDirections * sizeof(float), cudaMemcpyHostToDevice);
	
	return 0;
}




__global__ void sum_energy_matrices(const CostMatrices cm, float * weights, int numMatrices = 8)
{
	// each thread performs sum for all d values at a given x,y

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int w = cm.width;
	int h = cm.height;
	int dmax = cm.depth;

	float elementSum;
	if (x < w && y < h){
		for (int d = 0; d < dmax; d++) {
			elementSum = 0;
			for (int i = 0; i < numMatrices; i++) {
				elementSum += weights[i] * cm.E[i][(x + (y * w)) * dmax + d];
			}
			cm.S[(x + (y * w)) * dmax + d] = elementSum;
		}
	}
}

template<int numTestDirections>
int getFinalCosts(CostMatrices& cm, float * d_weights)
{
	int h = cm.height;
	int w = cm.width;
	int dmax = cm.depth;

	// allocate memory
	cm.S = 0;
	cudaMalloc((void**)&cm.S, h * w * dmax * sizeof(float));
	if (cm.S == 0) {
		std::cerr << "ERROR: S[] failed to allocate memory" << std::endl;
		return 1;
	}

	dim3 block_size(32,32);
	dim3 grid_size(0,0);
	grid_size.x = (w + block_size.x - 1) / block_size.x;
	grid_size.y = (h * block_size.y - 1) / block_size.y;

	sum_energy_matrices << <grid_size, block_size >> >(cm, d_weights, numTestDirections);


	// can now free energy matrices
	for (int i = 0; i < numTestDirections; i++) {
		cudaFree(cm.E[i]);
	}

	return 0;
}




__global__ void find_minima(const CostMatrices cm, float * D)
{
	// each thread find minima over all d values at a given x,y

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int w = cm.width;
	int h = cm.height;
	int dmax = cm.depth;

	int mind;
	float mindval;
	if (x < w && y < h) {

		mind = 0;
		mindval = cm.S[(x + (y * w)) * dmax];

		for (int d = 1; d < dmax; d++) {
			float test_dval = cm.S[(x + (y * w)) * dmax + d];
			if (test_dval < mindval) {
				mindval = test_dval;
				mind = d;
			}
		}

		D[x + (y * w)] = mind;

	}
}


int getDisparities(const CostMatrices cm, float ** D)
{
	int h = cm.height;
	int w = cm.width;

	// allocate memory
	*D = 0;
	cudaMalloc((void**)D, h * w * sizeof(float));
	if (*D == 0) {
		std::cerr << "ERROR: D[] failed to allocate memory" << std::endl;
		return 1;
	}

	dim3 block_size(32, 32);
	dim3 grid_size(0, 0);
	grid_size.x = (w + block_size.x - 1) / block_size.x;
	grid_size.y = (h * block_size.y - 1) / block_size.y;

	find_minima << <grid_size, block_size >> >(cm, *D);

	// can now free final cost matrix
	cudaFree(cm.S);

	return 0;
}



// compute initial costs where base and match (left and right) images are reversed
__global__ void initial_costs_reverse(CostMatrices cm)
{
	const int h = cm.height;
	const int w = cm.width;
	const int dmax = cm.depth;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int d = blockIdx.z * blockDim.z + threadIdx.z;

	// compute normalized coords
	float u = x / (float)w;
	float v = y / (float)h;
	float offset = d / (float)w;

	// fetch values from texture
	float baseVal = tex2D(mTex, u, v);
	float matchVal = tex2D(bTex, u + offset, v);

	// compute C(x,y,d) and write to matrix
	if (x < w && y < h && d < dmax)
		cm.C[(x + (y * w)) * dmax + d] = fabs(baseVal - matchVal);
}


int getInitialCosts_reverse(CostMatrices& cm)
{
	const int h = cm.height;
	const int w = cm.width;
	const int dmax = cm.depth;

	// allocate memory on device for the cost matrix
	cm.C = 0;
	cudaMalloc((void**)&cm.C, h * w * dmax * sizeof(float));
	if (cm.C == 0)
	{
		std::cerr << "Failed to allocate memory for initial cost matrix!" << std::endl;
		return -1;
	}

	// define block and grid sizes
	dim3 block_size(4, 4, 32);
	dim3 grid_size(0, 0, 0);
	grid_size.x = (w + block_size.x - 1) / block_size.x;
	grid_size.y = (h + block_size.y - 1) / (block_size.y);
	grid_size.z = (dmax + block_size.z - 1) / (block_size.z);

	// call kernel on GPU
	initial_costs_reverse << <grid_size, block_size >> >(cm);

	return 0;
}



// detect occlusion areas and set to zero
template<int pxPerThread>
__global__ void refine_dmap(float * D_base, const float * D_ref, int h, int w)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = pxPerThread * (blockIdx.y*blockDim.y + threadIdx.y);

	const int tolerance = 3;	// set some tolerance (px)

	for (int i = 0; i < pxPerThread; i++) {
		if (in_img(x, y + i, h, w)) {
			int baseVal = D_base[x + ((y + i) * w)];
			int matchVal;

			if (x + baseVal < w)	matchVal = D_ref[x - baseVal + ((y + i) * w)];
			else					continue;
			
			if (abs(baseVal - matchVal) > tolerance)
				D_base[x + ((y + i) * w)] = 0;
				
		}
	}
}

// refines base disparity map (writes to base map)
template<int pxPerThread>
void refineDisparityMap(float * D_base, float * D_ref, int h, int w)
{
	dim3 block_size(32, 8);
	dim3 grid_size(0, 0);
	grid_size.x = (w + block_size.x - 1) / block_size.x;
	grid_size.y = (h + pxPerThread * block_size.y - 1) / (pxPerThread * block_size.y);

	refine_dmap<pxPerThread><<<grid_size, block_size>>>(D_base, D_ref, h, w);

	// no longer need the match image disparity map
	cudaFree(D_ref);
}




// applies median filter with 3x3 kernel to image
__global__ void median_filter_3x3(float * d_input_img, float * d_output_img, int h, int w)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float window[9];

	if (!in_img(x,y,h,w))
		return;

	// get elements for kernel
	window[0] = (y == 0 || x == 0) ? 0 :		 d_input_img[(y - 1)*h + (x - 1)];
	window[1] = (y == 0) ? 0 :					 d_input_img[(y - 1)*w + x];
	window[2] = (y == 0 || x == w - 1) ? 0 :	 d_input_img[(y - 1)*w + (x + 1)];
	window[3] = (x == 0) ? 0 :					 d_input_img[y*w + (x - 1)];
	window[4] =									 d_input_img[y*w + x];
	window[5] = (x == w - 1) ? 0 :				 d_input_img[y*w + (x + 1)];
	window[6] = (y == h - 1 || x == 0) ? 0 :	 d_input_img[(y + 1)*w + (x - 1)];
	window[7] = (y == h - 1) ? 0 :				 d_input_img[(y + 1)*w + x];
	window[8] = (y == h - 1 || x == w - 1) ? 0 : d_input_img[(y + 1)*w + (x + 1)];

	// order elements
	for (uint j = 0; j<5; ++j)
	{
		// find position of minimum element
		float temp = window[j];
		uint  idx = j;
		for (uint l = j + 1; l<9; ++l)
			if (window[l] < temp){ idx = l; temp = window[l]; }

		// put found minimum element in its place
		window[idx] = window[j];
		window[j] = temp;
	}

	// write median value
	d_output_img[y*w + x] = window[4];
}

int doMedianFiltering(float ** image, int h, int w)
{
	// allocate memory
	float * result = 0;
	cudaMalloc((void**)&result, h * w * sizeof(float));
	if (result == 0) {
		std::cerr << "ERROR: failed to allocate memory for median filtering" << std::endl;
		return 1;
	}

	dim3 block_size(16, 32);
	dim3 grid_size(0, 0);
	grid_size.x = (w + block_size.x - 1) / block_size.x;
	grid_size.y = (h * block_size.y - 1) / block_size.y;

	median_filter_3x3 <<< grid_size, block_size >>> (*image, result, h, w);

	// free original image and return filtered result
	cudaFree(*image);
	*image = result;

	return 0;
}





#endif // #ifndef _SEMIGLOBALMATCHING_KERNEL_CU_
