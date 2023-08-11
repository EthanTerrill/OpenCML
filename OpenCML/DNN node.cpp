

//this entire fucking thing is depricated pleas ingore its existance


class DNN_node {

	int id;
	int inpSize;
	int outputSize;
public:

	cl_mem metadata;
	DNN_node() {



	}

	DNN_node(int id, int inpSize, int outputSize) {

		this->id			= id;
		this->inpSize		= inpSize;
		this->outputSize	= outputSize;


		int* mem = new int[3] {id, inpSize, outputSize};

		cl_int ret;

		metadata = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, 3 * sizeof(int), NULL, &ret);

		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, metadata, CL_TRUE, 0, 3 * sizeof(int), mem, 0, NULL, NULL);


		delete[] mem;

	
	}


	void forwardPropogate(buffer input, buffer output, cl_mem weights) {

		




	};
	void getBufferCostsLastLayer(image output) {

	};


	void getBufferCostsPrevLayer(image dOutputPrevLayer) {

	};



};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      