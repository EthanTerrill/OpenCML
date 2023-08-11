class layer {

private:

	/*
		The layer class is meant as an abstraction which can represent any type of layer
		accessed by the network class, the main idea behind this was that all of the public functions of
		the different layer types would have the same inputs and outputs meaning a more flexible solution could be made.


		Here we have a series of pointers to the different layer types but we only initialize one of them so as to not waste memory.
		We also have a series of pointers to member functions which we then use to point to the member functions which simply call
		the non member funcitons (try saying that 5 times fast)



		The advantages and disadvanteges of this pointer setup are as follows

		advantages:

			- looks cool
			- allows us to store the layer types in an array or vector without losing funtionality
			- black magic

		disadvanteges:
			- kind of a clunky solution
			- pointer function headaches
			- black magic


	*/


	layerType T;

	image(layer::* getBuffersP)  ();
	image(layer::* getdBuffersP) ();

	void (layer::* updateP)						();
	void (layer::* getWeightCostsP)				(image input);
	void (layer::* forwardPropogateP)			(image input);
	void (layer::* getBufferCostsLastLayerP)	(image output);
	void (layer::* getBufferCostsPrevLayerP)	(image dOutputPrevLayer);

	void (layer::* saveP)						(std::ofstream* f);
	void (layer::* loadP)						(std::ifstream* f);




	avgPoolLayer* avgP;
	scaleUpLayer* scale;
	DNN_layer* d;
	CNN_layer* c;

	//yes this is really the easiest way to do this



	inline void nullfunc() {}
	inline void nullfunc(image a) {}
	inline void nullfunc(std::ofstream* f) {}
	inline void nullfunc(std::ifstream* f) {}



	//cnn
	inline image cnnGetBuffers() { return c[0].getBuffers(); }
	inline image cnnGetdBuffers() { return c[0].getdBuffers(); }

	inline void cnnUpdate					()							{ c[0].update(); }
	inline void cnnGetWeightCosts			(image input)				{ c[0].getKernelCosts(input); }
	inline void cnnForwardPropogate			(image input)				{ c[0].forwardPropogate(input); }
	inline void cnnGetBufferCostsLastLayer	(image output)				{ c[0].getBufferCostsLastLayer(output); }
	inline void cnnGetBufferCostsPrevLayer	(image dOutputPrevLayer)	{ c[0].getBufferCostsPrevLayer(dOutputPrevLayer); }
	inline void cnnSave						(std::ofstream* f)			{ c[0].save(f); }
	inline void cnnLoad						(std::ifstream* f)			{ c[0].load(f); }


	//dnn

	inline image dnnGetBuffers() { return d[0].getBuffers(); }
	inline image dnnGetdBuffers() { return d[0].getdBuffers(); }

	inline void dnnUpdate					()							{ d[0].update(); }
	inline void dnnGetWeightCosts			(image input)				{ d[0].getWeightCosts(input); }
	inline void dnnForwardPropogate			(image input)				{ d[0].forwardPropogate(input); }
	inline void dnnGetBufferCostsLastLayer	(image output)				{ d[0].getBufferCostsLastLayer(output); }
	inline void dnnGetBufferCostsPrevLayer	(image dOutputPrevLayer)	{ d[0].getBufferCostsPrevLayer(dOutputPrevLayer); }
	inline void dnnSave						(std::ofstream* f)			{ d[0].save(f); }
	inline void dnnLoad						(std::ifstream* f)			{ d[0].load(f); }




	//avg Pool
	inline image avgPoolGetBuffers() { return avgP[0].getBuffers(); }
	inline image avgPoolGetdBuffers() { return avgP[0].getdBuffers(); }


	inline void avgPoolUpdate() { avgP[0].clearBuffers(); }
	inline void avgPoolForwardPropogate(image input) { avgP[0].forwardPropogate(input); }
	inline void avgPoolGetBufferCostsLastLayer(image output) { avgP[0].getBufferCostsLastLayer(output); }
	inline void avgPoolGetBufferCostsPrevLayer(image dOutputPrevLayer) { avgP[0].getBufferCostsPrevLayer(dOutputPrevLayer); }

	inline image scaleGetBuffers() { return scale[0].getBuffers(); }
	inline image scaleGetdBuffers() { return scale[0].getdBuffers(); }


	inline void scaleUpdate() { scale[0].clearBuffers(); }
	inline void scaleForwardPropogate(image input) { scale[0].forwardPropogate(input); }
	inline void scaleGetBufferCostsLastLayer(image output) { scale[0].getBufferCostsLastLayer(output); }
	inline void scaleGetBufferCostsPrevLayer(image dOutputPrevLayer) { scale[0].getBufferCostsPrevLayer(dOutputPrevLayer); }




public:

	layer(

		layerType t, 
		int inputBufferNum, 
		int outputBufferNum, 
		int kernelWidth, 
		int kernelHeight, 
		int inpBufferWidth, 
		int inpBufferHeight, 
		int inpDimsenion
	) {


		switch (t) {

		case CONVOLUTIONAL_LAYER:
			T = t;
			c = new CNN_layer(inputBufferNum, outputBufferNum, kernelWidth, kernelHeight, inpBufferWidth, inpBufferHeight);

			updateP						= &layer::cnnUpdate;
			saveP						= &layer::cnnSave;
			loadP						= &layer::cnnLoad;

			getBuffersP					= &layer::cnnGetBuffers;
			getdBuffersP				= &layer::cnnGetdBuffers;
			getWeightCostsP				= &layer::cnnGetWeightCosts;
			forwardPropogateP			= &layer::cnnForwardPropogate;
			getBufferCostsLastLayerP	= &layer::cnnGetBufferCostsLastLayer;
			getBufferCostsPrevLayerP	= &layer::cnnGetBufferCostsPrevLayer;
			
			break;

		case DNN:

			T = t;
			d = new DNN_layer(inpBufferWidth * inpBufferHeight, inpDimsenion, kernelWidth);

			updateP = &layer::dnnUpdate;
			saveP	= &layer::dnnSave;
			loadP	= &layer::dnnLoad;

			getBuffersP					= &layer::dnnGetBuffers;
			getdBuffersP				= &layer::dnnGetdBuffers;
			getWeightCostsP				= &layer::dnnGetWeightCosts;
			forwardPropogateP			= &layer::dnnForwardPropogate;
			getBufferCostsLastLayerP	= &layer::dnnGetBufferCostsLastLayer;
			getBufferCostsPrevLayerP	= &layer::dnnGetBufferCostsPrevLayer;

			break;
		case AVG_POOL_LAYER:

			T = t;
			avgP = new avgPoolLayer(inpBufferWidth, inpBufferHeight, inputBufferNum, kernelWidth);

			updateP = &layer::avgPoolUpdate;
			getWeightCostsP = &layer::nullfunc;
			saveP = &layer::nullfunc;
			loadP = &layer::nullfunc;



			getBuffersP					= &layer::avgPoolGetBuffers;
			getdBuffersP				= &layer::avgPoolGetdBuffers;
			forwardPropogateP			= &layer::avgPoolForwardPropogate;
			getBufferCostsLastLayerP	= &layer::avgPoolGetBufferCostsLastLayer;
			getBufferCostsPrevLayerP	= &layer::avgPoolGetBufferCostsPrevLayer;

			break;
		case MAX_POOL_LAYER:
			break;
		case MIN_POOL_LAYER:

			break;
		case SCALE_UP_LAYER:


			T = t;
			scale = new scaleUpLayer(inpBufferWidth, inpBufferHeight, inputBufferNum, kernelWidth);

			updateP			= &layer::scaleUpdate;
			getWeightCostsP = &layer::nullfunc;
			saveP			= &layer::nullfunc;
			loadP			= &layer::nullfunc;





			getBuffersP					= &layer::scaleGetBuffers;
			getdBuffersP				= &layer::scaleGetdBuffers;
			forwardPropogateP			= &layer::scaleForwardPropogate;
			getBufferCostsLastLayerP	= &layer::scaleGetBufferCostsLastLayer;
			getBufferCostsPrevLayerP	= &layer::scaleGetBufferCostsPrevLayer;

			break;

		}




	}

	image getBuffers()	{ return (this->*getBuffersP)	(); };
	image getdBuffers() { return (this->*getdBuffersP)	(); };

	void update					()							{ (this->*updateP)					(); }
	void save					(std::ofstream* f)			{ (this->*saveP)					(f); }
	void load					(std::ifstream* f)			{ (this->*loadP)					(f); }
	void getWeightCosts			(image input)				{ (this->*getWeightCostsP)			(input); }
	void forwardPropogate		(image input)				{ (this->*forwardPropogateP)		(input); }
	void getBufferCostsLastLayer(image output)				{ (this->*getBufferCostsLastLayerP) (output); }
	void getBufferCostsPrevLayer(image dOutputPrevLayer)	{ (this->*getBufferCostsPrevLayerP) (dOutputPrevLayer); }



};