class fastCNN_layer {

private:


	image outputs;
	image dOutputs;

	//std::vector<std::vector<CNN_kernel>> kernels;


	cl_mem weightsBuff;

	cl_mem metaData;
	cl_mem dWeights;


	int kW;
	int kH;
	int inpNum;
	int outpNum;

	void initializekernels() {
		
		if (kW == 0 || kH == 0)
			return;

		unsigned int wSize = kW * kH * inpNum * outpNum;
		//initialize random memory for kernel vals
		float* temp = new float[wSize]{ 0 };
		for (int i = 0; i < wSize; i++) {
				temp[i] = float(float(rand() % 200) - 100) / 200;

		}

		int* meta = new int[4]
		{
			kW,
			kH,
			inpNum,
			outpNum
		};

		cl_int ret;

		weightsBuff = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(wSize)	* sizeof(float), NULL, &ret);
		dWeights	= clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(wSize)	* sizeof(float), NULL, &ret);
		metaData	= clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(4)		* sizeof(float), NULL, &ret);


		clEnqueueWriteBuffer(COMMAND_QUEUE, metaData,		CL_TRUE, 0, 2		* sizeof(int),		meta,  0, NULL, NULL);
		clEnqueueWriteBuffer(COMMAND_QUEUE, weightsBuff,	CL_TRUE, 0, wSize	* sizeof(float),	temp,  0, NULL, NULL);
		clEnqueueWriteBuffer(COMMAND_QUEUE, dWeights,		CL_TRUE, 0, wSize	* sizeof(float),	NULL, 0, NULL, NULL);

		delete[] temp;
		delete[] meta;
	}


	void getBufferCosts(image dOutNextLayer, std::vector <std::vector <CNN_kernel> > kernelsNextLayer) {


		for (int i = 0; i < kernelsNextLayer.size(); i++)
			for (int j = 0; j < kernelsNextLayer[i].size(); j++)
				kernelsNextLayer[i][j].convolve_180(dOutNextLayer.getBuffer(j), dOutputs.getBuffer(i));
	}




public:
	fastCNN_layer(int inputBufferNum, int outputBufferNum, int kernelWidth, int kernelHeight, int inpBufferWidth, int inpBufferHeight) {

		kW = kernelWidth;
		kH = kernelHeight;
		inpNum = inputBufferNum;
		outpNum = outputBufferNum;

		initializekernels();
		if (inpBufferWidth - kernelWidth + 1 > 0 && inpBufferHeight - kernelHeight + 1 > 0) {
			outputs = image(inpBufferWidth - kernelWidth + 1, inpBufferHeight - kernelHeight + 1, outputBufferNum);
			dOutputs = image(inpBufferWidth - kernelWidth + 1, inpBufferHeight - kernelHeight + 1, outputBufferNum);
		}
		else {

			std::cerr << "Invalid dimensions";
			exit(0);
		}

	}

	void forwardPropogate(image inputs) {

		


	}

	void clearBuffers() {

		for (int j = 0; j < outpNum; j++) {

			outputs.getBuffer(j).clear();
			dOutputs.getBuffer(j).clear();
		}
	}

	void clearFBuffers() {

		for (int j = 0; j < outpNum; j++) {

			outputs.getBuffer(j).clear();

		}
	}

	buffer getBuffer(int n) {
		return outputs.getBuffer(n);
	}

	image getBuffers() {

		return outputs;
	}
	image getdBuffers() {

		return dOutputs;
	}
	cl_mem getKernels() {
		return weightsBuff;
	}

	void getBufferCostsPrevLayer(image doutPrevLayer) {



		//for (int i = 0; i < inpNum; i++)
			//for (int j = 0; j < outpNum; j++)
				//kernels[i][j].convolve_180(dOutputs.getBuffer(j), doutPrevLayer.getBuffer(i));
	}

	void getBufferCostsLastLayer(image output) {


		for (int i = 0; i < outpNum; i++) {

			outputs.getBuffer(i).findlastLayerCost(dOutputs.getBuffer(i), output.getBuffer(i));
		}

	}

	void getKernelCosts(image input) {

		/*
		for (int i = 0; i < inpNum; i++) {
			for (int j = 0; j < outpNum; j++) {

				kernels[i][j].getCost(input.getBuffer(i), dOutputs.getBuffer(j));

			}
		}
		*/
	}

	void update() {

		/*
		for (int i = 0; i < inpNum; i++) {
			for (int j = 0; j < outpNum; j++) {

				kernels[i][j].update();

			}
		}
		*/

		clearBuffers();
	}

	

	void save(std::ofstream* f) {

		if ((*f).is_open() && (*f).good()) {

			unsigned int wSize = kW * kH * inpNum * outpNum;
			float* temp = new float[wSize];
			if (temp != nullptr)
				clEnqueueReadBuffer(COMMAND_QUEUE, weightsBuff, CL_TRUE, 0, wSize * sizeof(float), temp, 0, NULL, NULL);

			(*f).write((char*)temp, wSize * sizeof(float));

			delete[] temp;
		}
		else {

			std::cerr << "invalid fstream";

		}

	}
	void load(std::ifstream* f) {

		if ((*f).is_open() && (*f).good()) {

			unsigned int wSize = kW * kH * inpNum * outpNum;

			float* temp = new float[wSize];

			(*f).read((char*)temp, wSize * sizeof(float) / sizeof(char));

			if (temp != nullptr)
				clEnqueueWriteBuffer(COMMAND_QUEUE, weightsBuff, CL_TRUE, 0, wSize * sizeof(float), temp, 0, NULL, NULL);

			delete[] temp;
		}
		else {

			std::cerr << "invalid fstream";

		}

	}


};