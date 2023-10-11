class DNN_layer {

public:


private:

	image dvalues;
	image nodeVals;

	image activeNodeVals;

	cl_mem weights;
	cl_mem dweights;

	cl_mem bias;
	cl_mem dBias;

	cl_mem metadata; 


	// this is a very bandaid solution I would like to replace this later but for now it works and im tired
	cl_mem metadata2;

	int inpWidth;
	int outpWidth;
	int inpDimensionNum;

	bool Bsolved = false;


	void clearBuffers() {

		nodeVals.getBuffer(0).clear();
		activeNodeVals.getBuffer(0).clear();
		dvalues.getBuffer(0).clear();

	}

	void activateLayer() {
		
		cl_int r;
		clSetKernelArg(LIGHT_SIGMOID, 0, sizeof(cl_mem), &nodeVals.buffers[0].data);
		clSetKernelArg(LIGHT_SIGMOID, 1, sizeof(cl_mem), &activeNodeVals.buffers[0].data);
		clSetKernelArg(LIGHT_SIGMOID, 2, sizeof(cl_mem), &metadata);


		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, LIGHT_SIGMOID, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL); 

		clFinish(COMMAND_QUEUE);
		
	};

	void processLayer(image input) {
	
		cl_int r;
		clSetKernelArg(DNN_PROPGATE, 0, sizeof(cl_mem), &weights);
		clSetKernelArg(DNN_PROPGATE, 2, sizeof(cl_mem), &nodeVals.buffers[0].data);
		clSetKernelArg(DNN_PROPGATE, 3, sizeof(cl_mem), &metadata2);

		for (int i = 0; i < inpDimensionNum; i++) {
		

			clSetKernelArg(DNN_PROPGATE, 1, sizeof(cl_mem), &input.buffers[i].data);

			clSetKernelArg(DNN_PROPGATE, 4, sizeof(cl_mem), &input.buffers[i].meta);
			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, DNN_PROPGATE, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);
			clFinish(COMMAND_QUEUE);
		} 
	}

	void addBias() {
	
		cl_int r;

		clSetKernelArg(ADD, 0, sizeof(cl_mem), &bias);
		clSetKernelArg(ADD, 1, sizeof(cl_mem), &nodeVals.buffers[0].data);
		clSetKernelArg(ADD, 2, sizeof(cl_mem), &nodeVals.buffers[0].meta);


		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, ADD, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);

		clFinish(COMMAND_QUEUE);
		
	}

	void solveDwieghts(image input) {
		cl_int r;

		for (int i = 0; i < inpDimensionNum; i++) {

			clSetKernelArg(DNN_GETWCOSTS, 0, sizeof(cl_mem), &dweights);
			clSetKernelArg(DNN_GETWCOSTS, 2, sizeof(cl_mem), &dBias);
			clSetKernelArg(DNN_GETWCOSTS, 3, sizeof(cl_mem), &metadata);
			clSetKernelArg(DNN_GETWCOSTS, 1, sizeof(cl_mem), &input.buffers[i].data);

			clSetKernelArg(DNN_GETWCOSTS, 4, sizeof(cl_mem), &input.buffers[i].meta);
			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, DNN_GETWCOSTS, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);
			clFinish(COMMAND_QUEUE);
		} 
	
	}

	void solveDbias() {
		cl_int r;
		clSetKernelArg(DNN_GETBIASCOSTS, 0, sizeof(cl_mem), &nodeVals.buffers[0].data);
		clSetKernelArg(DNN_GETBIASCOSTS, 1, sizeof(cl_mem), &dBias);
		clSetKernelArg(DNN_GETBIASCOSTS, 2, sizeof(cl_mem), &dvalues.buffers[0].data);
		clSetKernelArg(DNN_GETBIASCOSTS, 3, sizeof(cl_mem), &metadata);
		r = clEnqueueNDRangeKernel(COMMAND_QUEUE, DNN_GETBIASCOSTS, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);
		clFinish(COMMAND_QUEUE);

	}

	void solveDprevLayer(image prevLayerCost) {
		cl_int r;
		clSetKernelArg(DNN_BACKPROPGATE, 1, sizeof(cl_mem), &weights);
		clSetKernelArg(DNN_BACKPROPGATE, 2, sizeof(cl_mem), &dBias);
		clSetKernelArg(DNN_BACKPROPGATE, 3, sizeof(cl_mem), &metadata);

		for (int i = 0; i < inpDimensionNum; i++) {


			clSetKernelArg(DNN_BACKPROPGATE, 0, sizeof(cl_mem), &prevLayerCost.buffers[i].data);

			clSetKernelArg(DNN_BACKPROPGATE, 4, sizeof(cl_mem), &prevLayerCost.buffers[i].meta);
			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, DNN_BACKPROPGATE, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);
			clFinish(COMMAND_QUEUE);
		}

		

	}


	public:

	DNN_layer(int inpWidth, int dimensionNum, int outpWidth) {

		 
		this->inpWidth = inpWidth;
		this->outpWidth = outpWidth;
		this->inpDimensionNum = dimensionNum;

		int* mem = new int[2]{inpWidth, outpWidth };

		cl_int ret;

		metadata = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, &ret);

		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, metadata, CL_TRUE, 0, 2 * sizeof(int), mem, 0, NULL, NULL);

		mem = new int[2]{ 1, outpWidth };
		metadata2 = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, &ret);

		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, metadata, CL_TRUE, 0, 2 * sizeof(int), mem, 0, NULL, NULL);

		delete[] mem;

		float* weightmem	= new float[inpWidth * outpWidth * dimensionNum] {0};
		float* biasmem		= new float[outpWidth] { 0 };

		for (int i = 0; i < outpWidth * inpWidth; i++) {

			weightmem[i] = float(float(rand() % 200) - 100) / 200;
			//weightmem[i] = 1;

		}
		for (int i = 0; i < outpWidth; i++) {

			biasmem[i] = float(float(rand() % 200) - 100) / 200;
			//weightmem[i] = 1;

		}

		weights			= clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, inpWidth * outpWidth * dimensionNum * sizeof(float), NULL, &ret);
		dweights		= clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, inpWidth * outpWidth * dimensionNum * sizeof(float), NULL, &ret);
		bias		    = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, outpWidth * sizeof(float), NULL, &ret);
		dBias			= clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, outpWidth * sizeof(float), NULL, &ret);


		nodeVals		= image(outpWidth, 1, 1);
		dvalues			= image(outpWidth, 1, 1);
		activeNodeVals	= image(outpWidth, 1, 1);

		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, weights,  CL_TRUE, 0, inpWidth * outpWidth * dimensionNum * sizeof(float), weightmem,	0, NULL, NULL);
		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, dweights, CL_TRUE, 0, inpWidth * outpWidth * dimensionNum * sizeof(float), NULL,		0, NULL, NULL);

		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, bias,  CL_TRUE, 0, outpWidth * sizeof(float), biasmem, 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, dBias, CL_TRUE, 0, outpWidth * sizeof(float), NULL,  0, NULL, NULL);
		
		delete[] weightmem;
		delete[] biasmem; 


	}


	void update() {
		
		subtractandclear(dweights, weights, metadata);
		subtractandclearB(dBias, bias, metadata);
 		clearBuffers();
	};

	void clearFBuffers() {

		nodeVals.getBuffer(0).clear();
		activeNodeVals.getBuffer(0).clear();
		dvalues.getBuffer(0).clear();

	}
	void getWeightCosts(image input) {

		if (!Bsolved) {

			solveDbias();
			
		}
		solveDwieghts(input);


		Bsolved = false;
	};


	/*
		this implementaiton is a bit problematic because it assumes the layer behind it only 
		has one dimension, im gonna have to fix by chnanging this or add a n-d to 1-d layer between conv layers and dnn layers
	*/
	void forwardPropogate(image input) {
		 
		processLayer(input);
		addBias();
		activateLayer();

	};
	void getBufferCostsLastLayer (image output) {

		activeNodeVals.getBuffer(0).findlastLayerCost(dvalues.getBuffer(0), output.getBuffer(0));

		
	};


	void getBufferCostsPrevLayer (image dOutputPrevLayer) {
		

		if (!Bsolved) {

			solveDbias();

		}
		solveDprevLayer(dOutputPrevLayer);

	};

	image getBuffers() {


	
		return activeNodeVals;
	}

	image getdBuffers() {
	
		return dvalues;

	};

	void save(std::ofstream* f) {


		if ((*f).is_open() && (*f).good()) {
			
			float* temp = new float[inpWidth * outpWidth];
			if (temp != nullptr)
				clEnqueueReadBuffer(COMMAND_QUEUE, weights, CL_TRUE, 0, inpWidth * outpWidth * sizeof(float), temp, 0, NULL, NULL);

			//(*f).write((char*) temp, inpWidth * outpWidth * sizeof(float));
			for (int i = 0; i < inpWidth * outpWidth; i++)
				(*f) << temp[i] << " ";

			delete[] temp;

			temp = new float[outpWidth];
			if (temp != nullptr)
				clEnqueueReadBuffer(COMMAND_QUEUE, bias, CL_TRUE, 0, outpWidth * sizeof(float), temp, 0, NULL, NULL);

			//(*f).write((char*)temp,  outpWidth * sizeof(float));
			for (int i = 0; i < outpWidth; i++)
				(*f) << temp[i] << " ";

			delete[] temp;


		}
		else {
			
			std::cerr << "invalid fstream";
		
		}


	}
	void load(std::ifstream* f) {


		if ((*f).is_open() && (*f).good()) {

			float* temp = new float[inpWidth * outpWidth];

			//(*f).read((char*)temp, inpWidth * outpWidth * sizeof(float) / sizeof(char));
			for (int i = 0; i < inpWidth * outpWidth; i++)
				(*f) >> temp[i];


			if (temp != nullptr)
				clEnqueueWriteBuffer(COMMAND_QUEUE, weights, CL_TRUE, 0, inpWidth * outpWidth * sizeof(float), temp, 0, NULL, NULL);

			

			delete[] temp;

			temp = new float[outpWidth];

			//(*f).read((char*)temp, outpWidth * sizeof(float) / sizeof(char));
			for (int i = 0; i < outpWidth; i++)
				(*f) >> temp[i];


			if (temp != nullptr)
				clEnqueueWriteBuffer(COMMAND_QUEUE, bias, CL_TRUE, 0, outpWidth * sizeof(float) , temp, 0, NULL, NULL);


			delete[] temp;


		}
		else {

			std::cerr << "invalid fstream";

		}


	}
};