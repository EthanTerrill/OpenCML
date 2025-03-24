#include "image.hpp"

class avgPoolLayer {
private:
	int height, width, size, strideS;
	image outputs;
	image dOutputs;
	cl_mem meta;


public:

	avgPoolLayer(int inputWidth, int inputHeight, int dimensionNum, int ksize);
	avgPoolLayer(int inputWidth, int inputHeight, int dimensionNum, int ksize, stride strideSize);


	image getBuffers();
	image getdBuffers();

	void forwardPropogate(image input) {


		clSetKernelArg(AVERAGE_POOL, 2, sizeof(cl_mem), &meta);

		for (int i = 0; i < outputs.getDimensionNum(); i++) {
			cl_int r;
			clSetKernelArg(AVERAGE_POOL, 0, sizeof(cl_mem), &input.buffers[i].data);
			clSetKernelArg(AVERAGE_POOL, 1, sizeof(cl_mem), &outputs.buffers[i].data);

			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, AVERAGE_POOL, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


			clFinish(COMMAND_QUEUE);

		}
	}

	void getBufferCostsPrevLayer (image doutputLastLayer) {
		

		clSetKernelArg(AVERAGE_POOL_COST, 2, sizeof(cl_mem), &meta);

		for (int i = 0; i < outputs.getDimensionNum(); i++) {

			cl_int r;

			clSetKernelArg(AVERAGE_POOL_COST, 0, sizeof(cl_mem), &doutputLastLayer.buffers[i].data);
			clSetKernelArg(AVERAGE_POOL_COST, 1, sizeof(cl_mem), &dOutputs.buffers[i].data);

			r = clEnqueueNDRangeKernel(COMMAND_QUEUE, AVERAGE_POOL_COST, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


			clFinish(COMMAND_QUEUE);

		}
	}
	void getBufferCostsLastLayer (image output) {


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
