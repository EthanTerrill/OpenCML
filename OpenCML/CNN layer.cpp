class CNN_layer{

	image outputs;
	image dOutputs;
	std::vector<std::vector<CNN_kernel>> kernels;

	stride strideSize;

private:
	
	int kW;
	int kH;
	int inpNum;
	int outpNum;


	void getBufferCosts(image dOutNextLayer, std::vector <std::vector <CNN_kernel> > kernelsNextLayer) {
		

		for (int i = 0; i < kernelsNextLayer.size(); i++)
			for (int j = 0; j < kernelsNextLayer[i].size(); j++)
				kernelsNextLayer[i][j].convolve_180(dOutNextLayer.getBuffer(j), dOutputs.getBuffer(i));
	}

	


public:
	CNN_layer(int inputBufferNum, int outputBufferNum, int kernelWidth, int kernelHeight, int inpBufferWidth, int inpBufferHeight) {
	
		kW			= kernelWidth;
		kH			= kernelHeight;
		inpNum		= inputBufferNum;
		outpNum		= outputBufferNum;
		strideSize	= STRIDE_2x2;

		for (int i = 0; i <inputBufferNum; i++) {
			std::vector<CNN_kernel> temp;
			for (int j = 0; j < outputBufferNum; j++) {

				temp.push_back(CNN_kernel(kernelWidth, kernelHeight, strideSize));
			}

			kernels.push_back(temp);
		}

		int outputWidth = (inpBufferWidth - kernelWidth + 1) / strideSize;
		int outputHeight = (inpBufferHeight - kernelHeight + 1) / strideSize;

		if (outputWidth > 0 && outputHeight > 0) {


			

			outputs		= image(outputWidth, outputHeight, outputBufferNum);
			dOutputs	= image(outputWidth, outputHeight, outputBufferNum);
		}
		else {
		
			std::cerr << "Invalid dimensions";
			exit( 0);
		}
		
	}
	
	void forwardPropogate(image inputs) {
		
		for (int i = 0; i < inpNum; i++)
			for (int j = 0; j < outpNum; j++)
				kernels[i][j].convolve(inputs.getBuffer(i), outputs.getBuffer(j));

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
	std::vector<std::vector<CNN_kernel>> getKernels() {
		return kernels;
	}

	void getBufferCostsPrevLayer(image doutPrevLayer) {



		for (int i = 0; i < inpNum; i++) 
			for (int j = 0; j < outpNum; j++) 
				kernels[i][j].convolve_180(dOutputs.getBuffer(j), doutPrevLayer.getBuffer(i));
	}

	void getBufferCostsLastLayer(image output) {


		for (int i = 0; i < outpNum; i++) {

			outputs.getBuffer(i).findlastLayerCost(dOutputs.getBuffer(i), output.getBuffer(i));
		}

	}

	void getKernelCosts(image input) {

		for (int i = 0; i < inpNum; i++) {
			for (int j = 0; j < outpNum; j++) {

				kernels[i][j].getCost(input.getBuffer(i), dOutputs.getBuffer(j));

			}
		}
	}

	void update() {
		
		for (int i = 0; i < inpNum; i++) {
			for (int j = 0; j < outpNum; j++) {

				kernels[i][j].update();

			} 
		}
		clearBuffers();
	}


	void kernelstoBmp(int d, std::string fileName) {

		float*** bmpBuffer = new float** [3];


		for (int i = 0; i < 3; i++) {
			bmpBuffer[i] = kernels[d][i].showBuffer();
		}

		writeBufferToBmp(fileName, kW, kH, bmpBuffer);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j <kW; j++) {
				delete bmpBuffer[i][j];
			}
			delete[] bmpBuffer[i];
		}
		delete[] bmpBuffer;
	}

	void save(std::ofstream* f) {
	
		for (int i = 0; i < inpNum; i++) {
			for (int j = 0; j < outpNum; j++) {
				kernels[i][j].save(f);
			}
		}
	}

	void load(std::ifstream* f) {

		for (int i = 0; i < inpNum; i++) {
			for (int j = 0; j < outpNum; j++) {
				kernels[i][j].load(f);
			}
		}

	}

};