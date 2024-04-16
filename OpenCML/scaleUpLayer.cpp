class scaleUpLayer {

private:
	int height, width, size;
	stride s;
	image outputs;
	image dOutputs;
	cl_mem meta;


public:

	scaleUpLayer(int inputWidth, int inputHeight, int dimensionNum, int ksize, stride strideSize) {

		s			= strideSize;
		outputs		= image(inputWidth * ksize, inputHeight * ksize, dimensionNum);
		dOutputs	= image(inputWidth * ksize, inputHeight * ksize, dimensionNum);
		height		= inputHeight * ksize;
		width		= inputWidth * ksize;
		size		= ksize;

		int* metadata = new int[5]{ width, height, ksize, dimensionNum, s };

		cl_int ret;

		meta = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, 5 * sizeof(int), NULL, &ret);
		ret = clEnqueueWriteBuffer(COMMAND_QUEUE, meta, CL_TRUE, 0, 5 * sizeof(int), metadata, 0, NULL, NULL);


	}



	image getBuffers() {

		return outputs;
	}
	image getdBuffers() {
		return dOutputs;
	}

	void forwardPropogate(image input) {

		for (int i = 0; i < outputs.getDimensionNum(); i++) {
			cl_int r;

			clSetKernelArg(SCALE_UP, 0, sizeof(cl_mem), &outputs.buffers[i].data);
			clSetKernelArg(SCALE_UP, 1, sizeof(cl_mem), &input.buffers[i].data);
			clSetKernelArg(SCALE_UP, 2, sizeof(cl_mem), &meta);

			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, SCALE_UP, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);
			 
			clFinish(COMMAND_QUEUE);

		}
	}

	void getBufferCostsPrevLayer(image doutputLastLayer) {

		for (int i = 0; i < outputs.getDimensionNum(); i++) {

			cl_int r;

			clSetKernelArg(SCALE_UP_COST, 0, sizeof(cl_mem), &doutputLastLayer.buffers[i].data);
			clSetKernelArg(SCALE_UP_COST, 1, sizeof(cl_mem), &dOutputs.buffers[i].data);
			clSetKernelArg(SCALE_UP_COST, 2, sizeof(cl_mem), &meta);

			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, SCALE_UP_COST, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


			clFinish(COMMAND_QUEUE);

		}
	}
	void getBufferCostsLastLayer(image output) {


		for (int i = 0; i < outputs.getDimensionNum(); i++) {

			outputs.getBuffer(i).findlastLayerCost(dOutputs.getBuffer(i), output.getBuffer(i));
		}


	}
	void clearBuffers() {

		for (int j = 0; j < outputs.getDimensionNum(); j++) {

			outputs.getBuffer(j).clear();
			dOutputs.getBuffer(j).clear();
		}
	}
	void clearFBuffers() {

		for (int j = 0; j < outputs.getDimensionNum(); j++) {

			outputs.getBuffer(j).clear();
			dOutputs.getBuffer(j).clear();
		}
	}


};