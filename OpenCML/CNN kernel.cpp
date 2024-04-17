#pragma once

#include <chrono>
using namespace std::chrono;

class CNN_kernel {
private:
	cl_mem weights;
	cl_mem metaData;
	cl_mem dWeights;


	int width;
	int height;

public:

	CNN_kernel() 
	{
		
		cl_int ret;

		dWeights = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
		weights  = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
		metaData = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(3)     * sizeof(float), NULL, &ret);


		this->width = NULL;
		this->height = NULL;

	}
	CNN_kernel(int width, int height, stride strideSize)
	{
		
		if (width == 0 || height == 0)
			return;

		this->width = width;
		this->height= height;

		float* temp = new float[width * height]{ 0 };
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {

				temp[i * height + j] = float(float(rand() % 200) - 100) / 200;
			}
		
		}

		int* meta = new int[3]
		{
			width,
			height,
			strideSize
		};

		cl_int ret;

		weights   = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
		dWeights  = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
		metaData  = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(3)     * sizeof(float), NULL, &ret);


		clEnqueueWriteBuffer(COMMAND_QUEUE, metaData,  CL_TRUE, 0, 3 * sizeof(int),                meta,  0, NULL, NULL);
		clEnqueueWriteBuffer(COMMAND_QUEUE, weights,   CL_TRUE, 0, width * height * sizeof(float), temp,  0, NULL, NULL);
		clEnqueueWriteBuffer(COMMAND_QUEUE, dWeights,  CL_TRUE, 0, width * height * sizeof(float), NULL, 0, NULL, NULL);

		delete[] temp;



	}
	CNN_kernel(float** kernel, int width, int height, stride strideSize) 
	{

		if (kernel == nullptr || width == 0 || height == 0)
			return;


		this->width = width;
		this->height = height;

		float* temp;
		array2dToArray1d(temp, kernel, width, height);

		float* dtemp = new float[width * height]{ 0 };
		

		int* meta = new int[3]
		{
			width,
			height,
			strideSize
		};

		cl_int ret;


		weights  = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
		dWeights = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
		metaData = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(3) * sizeof(float), NULL, &ret);

		clEnqueueWriteBuffer(COMMAND_QUEUE, metaData,  CL_TRUE, 0, 3 * sizeof(int),                meta, 0, NULL, NULL);
		clEnqueueWriteBuffer(COMMAND_QUEUE, weights,   CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);
		clEnqueueWriteBuffer(COMMAND_QUEUE, dWeights,  CL_TRUE, 0, width * height * sizeof(float), dtemp, 0, NULL, NULL);

		delete[] temp;
		delete[] dtemp;

	}


	void randomize() {
	
		float* temp = new float[width * height]{ 0 };

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				temp[i * height + j] = float(float(rand() % 200) - 100) / 200;
			}

		}

		cl_int ret;

		clEnqueueWriteBuffer(COMMAND_QUEUE, weights, CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);

		delete[] temp;
	
	}
	void convolve(buffer input, buffer ret) {


		cl_int r;
		clSetKernelArg(CONVOLVE, 0, sizeof(cl_mem), &weights);
		clSetKernelArg(CONVOLVE, 1, sizeof(cl_mem), &input.data);
		clSetKernelArg(CONVOLVE, 2, sizeof(cl_mem), &ret.data);
		clSetKernelArg(CONVOLVE, 3, sizeof(cl_mem), &input.meta);
		clSetKernelArg(CONVOLVE, 4, sizeof(cl_mem), &metaData);

		
		
		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);
		 

		clFinish(COMMAND_QUEUE);

	}
	void convolve(image input, image ret) {

		if (
				input.getHeight() - ret.getHeight() != height - 1 ||
				input.getWidth()  - ret.getWidth()  != width  - 1
		) 
		{
			throw std::invalid_argument("improper sizing");
		}



		for (int i = 0; i < input.getDimensionNum(); i++) {
			convolve(input.getBuffer(i), ret.buffers[i]);
		}

	}
	void convolve_180(buffer input, buffer ret) {


		cl_int r;
		clSetKernelArg(CONVOLVE_180, 0, sizeof(cl_mem), &weights);
		clSetKernelArg(CONVOLVE_180, 1, sizeof(cl_mem), &input.data);
		clSetKernelArg(CONVOLVE_180, 2, sizeof(cl_mem), &ret.data);
		clSetKernelArg(CONVOLVE_180, 3, sizeof(cl_mem), &input.meta);
		clSetKernelArg(CONVOLVE_180, 4, sizeof(cl_mem), &metaData);

		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_180, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);



		clFinish(COMMAND_QUEUE);

	}
	void convolve_180(image input, image ret) {

		if (
			input.getHeight() - ret.getHeight() != height - 1 ||
			input.getWidth()  - ret.getWidth()  != width  - 1
			)
		{
			throw std::invalid_argument("improper sizing");
		}



		for (int i = 0; i < input.getDimensionNum(); i++) {
			convolve_180(input.getBuffer(i), ret.buffers[i]);
		}

	}
	void getCost(buffer input, buffer cost) {


		cl_int r;
		clSetKernelArg(SOLVE_dKERNELS, 0, sizeof(cl_mem), &cost.data);
		clSetKernelArg(SOLVE_dKERNELS, 1, sizeof(cl_mem), &input.data);
		clSetKernelArg(SOLVE_dKERNELS, 2, sizeof(cl_mem), &dWeights);
		clSetKernelArg(SOLVE_dKERNELS, 3, sizeof(cl_mem), &input.meta);
		clSetKernelArg(SOLVE_dKERNELS, 4, sizeof(cl_mem), &cost.meta);

		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, SOLVE_dKERNELS, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


		clFinish(COMMAND_QUEUE);

	}


	void update() {


		cl_int r;
		clSetKernelArg(SUBTRACT_AND_CLEAR, 0, sizeof(cl_mem), &dWeights);
		clSetKernelArg(SUBTRACT_AND_CLEAR, 1, sizeof(cl_mem), &weights);
		clSetKernelArg(SUBTRACT_AND_CLEAR, 2, sizeof(cl_mem), &metaData);

		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, SUBTRACT_AND_CLEAR, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);


		clFinish(COMMAND_QUEUE);


	}
	float** showBuffer() {



		float* temp = new float[width * height];
		if (temp != nullptr)
			clEnqueueReadBuffer(COMMAND_QUEUE, weights, CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);

		float** RBG_buffer = nullptr;
		
		array1dToArray2d(RBG_buffer, temp, width, height);

		delete[] temp;

		return RBG_buffer;
	}
	float** showdBuffer() {



		float* temp = new float[width * height];
		if (temp != nullptr)
			clEnqueueReadBuffer(COMMAND_QUEUE, dWeights, CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);

		float** RBG_buffer = nullptr;
		
		array1dToArray2d(RBG_buffer, temp, width, height);

		delete[] temp;

		return RBG_buffer;
	}
	void save(std::ofstream* f) {

		if ((*f).is_open() && (*f).good()) {

			float* temp = new float[width * height];
			if (temp != nullptr)
				clEnqueueReadBuffer(COMMAND_QUEUE, weights, CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);

			(*f).write((char*)temp, width * height * sizeof(float));

			delete[] temp;
		}
		else {

			std::cerr << "invalid fstream";

		}

	}
	void load(std::ifstream* f) {

		if ((*f).is_open() && (*f).good()) {

			float* temp = new float[width * height];

			(*f).read((char*)temp, width * height * sizeof(float) / sizeof(char));

			if (temp != nullptr)
				clEnqueueWriteBuffer(COMMAND_QUEUE, weights, CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);

			delete[] temp;
		}
		else {

			std::cerr << "invalid fstream";

		}

	}
};