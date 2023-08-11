class CNN {
private:

	std::vector<layer> layers;

	std::vector<layerType> layerTypes;
	std::vector<int> layerDimensions;
	std::vector<int> kernelSizes;
	int inputWidth, inputHeight;
	
	cl_mem lr;

public:

	CNN(int inpDimensionNum, int inputWidth, int inputHeight) {

		layerDimensions.push_back(inpDimensionNum);
		this->inputHeight = inputHeight;
		this->inputWidth = inputWidth;

		float* l = new float[1]{ 0.1 };

		cl_int ret;

		lr = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE,  sizeof(float), NULL, &ret);
		clEnqueueWriteBuffer(COMMAND_QUEUE, lr, CL_TRUE, 0, sizeof(int), l, 0, NULL, NULL);

		delete[] l;


	}

	void setLearningRate(float f) {
	

		float* l = new float[1]{ f };

		cl_int ret;

		lr = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, sizeof(float), NULL, &ret);
		clEnqueueWriteBuffer(COMMAND_QUEUE, lr, CL_TRUE, 0, sizeof(float), l, 0, NULL, NULL);

		delete[] l;
	}

	void addLayer(layerType t, int dimensionNum, int kernelSize) {

		layerDimensions.push_back(dimensionNum);
		kernelSizes.push_back(kernelSize);
		layerTypes.push_back(t);

	}
	void addLayer(layerType t, int kernelSize) {

		if (t == AVG_POOL_LAYER || t == SCALE_UP_LAYER) {
			layerDimensions.push_back(layerDimensions[layerDimensions.size() - 1]);
			kernelSizes.push_back(kernelSize);
			layerTypes.push_back(t);
		}
		else if (t == DNN) {
			layerDimensions.push_back(1);
			kernelSizes.push_back(kernelSize);
			layerTypes.push_back(t);
		}
		else {

			std::cerr << "no";

		}

	}

	void setup() {

		int w = inputWidth;
		int h = inputHeight;
		for (int i = 1; i < layerDimensions.size(); i++) {

			layer l = layer(layerTypes[i - 1], layerDimensions[i - 1], layerDimensions[i], kernelSizes[i - 1], kernelSizes[i - 1], w, h, layerDimensions[i - 1]);
			layers.push_back(l);
			w = l.getBuffers().getWidth();
			h = l.getBuffers().getHeight();

		}

	};



	void backpropagate(image input, image output) {

		layers[layers.size() - 1].getBufferCostsLastLayer(output);
		for (int i = layers.size() - 1; i >= 1; i--) {
			layers[i].getBufferCostsPrevLayer(layers[i - 1].getdBuffers());
		}

			


		layers[0].getWeightCosts(input);
		for (int i = 1; i < layers.size(); i++) {
			layers[i].getWeightCosts(layers[i - 1].getBuffers());
		}

	}
	void forwardPropagate(image input) {

		layers[0].forwardPropogate(input);

		for (int i = 1; i < layers.size(); i++) {
			layers[i].forwardPropogate(layers[i - 1].getBuffers());
		}
	}

	image getOutputs() {
		return layers[layers.size() - 1].getBuffers();
	}
	image getdOutputs() {
		return layers[layers.size() - 1].getdBuffers();
	}

	image getOutputs(int i) {
		return layers[i].getBuffers();
	}
	image getdOutputs(int i) {
		return layers[i].getdBuffers();
	}

	void update() {

		clSetKernelArg(SUBTRACT_AND_CLEAR, 3, sizeof(cl_mem), &lr);


		for (int i = 0; i < layers.size(); i++) {
			layers[i].update();
		}
	}

	void save(std::string fileName) {
		std::ofstream f;

		f.open( fileName, std::ios::binary);

		f << "FILE_TYPE: DNN FORMAT V1\n";
		f << layerTypes.size() << "\n";
		for (int i = 0; i < layerTypes.size(); i++) {
			f << layerTypes[i] << " " << layerDimensions[i + 1] << " " << kernelSizes[i] << "\n";
		}

		for (int i = 0; i < layerTypes.size(); i++) {
			layers[i].save(&f);
		}

		f.close();
	
	}

	void load(std::string fileName) {
		std::ifstream f;

		char* garbageP = new char[24];
		char  garbage;

		f.open(fileName, std::ios::binary | std::ios::in);



		f.read(garbageP, 24);

		
		int layerSize;
		f >> layerSize;

		
		for (int i = 0; i < layerSize; i++) {
			int layerT;
			int layerDimension;
			int kernelSize;

			f >> layerT >>  layerDimension >> kernelSize;


			layerTypes.push_back(static_cast<layerType>(layerT));
			layerDimensions.push_back(layerDimension);
			kernelSizes.push_back(kernelSize);
		}


		setup();


		for (int i = 0; i < layerTypes.size(); i++) {
			layers[i].load(&f);
		}

		f.close();

		delete[] garbageP;

	}

	void trainOnDataSet() {}

	void resize() {}

};