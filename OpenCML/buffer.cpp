#pragma once

struct buffer {
private:
    unsigned int width, height ;

public:
    cl_mem data;
    cl_mem meta;


    buffer() {

        width = height = 0;
        data = meta = NULL;

    }


    buffer(int width, int height, int dimension, int dimensionNum) {




        cl_int ret = 0;

        this->width = width;
        this->height = height;

        float* temp = new float[width * height];

        if (temp != NULL)
            for (int x = 0; x < width * height; x++)
                    if (x < height * width)    // if statement to prevent overflow (intellisense can be a little annoying)
                        temp[x] = 0;
            


        data = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
        ret = clEnqueueWriteBuffer(COMMAND_QUEUE, data, CL_FALSE, 0, size_t(width * height) * sizeof(float), temp, 0, NULL, NULL);

        

        int* meta = new int[4];

        if (meta != nullptr) {
            meta[0] = width;
            meta[1] = height;
            meta[2] = dimensionNum;
            meta[3] = dimension;
        }
        this->meta = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_ONLY, 4 * sizeof(int), NULL, &ret);

        ret = clEnqueueWriteBuffer(COMMAND_QUEUE, this->meta, CL_TRUE, 0, 4 * sizeof(int), meta, 0, NULL, NULL);
        delete[] temp;
        delete[] meta;


    }

    buffer(int width, int height, int dimension, int dimensionNum, float** buffer) {
               
        cl_int ret = 0;

        this->width = width;
        this->height = height;


        float* temp;
        array2dToArray1d(temp, buffer, width, height);

        //create buffer 
        this->data = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, size_t(width * height) * sizeof(float), NULL, &ret);
        //assign memory to buffer
        clEnqueueWriteBuffer(COMMAND_QUEUE, this->data, CL_FALSE, 0, size_t(width * height) * sizeof(float), temp, 0, NULL, NULL);

        //create "meta" data (just the width and height)
        int* meta = new int[4];

        if (meta != nullptr) {
            meta[0] = width;
            meta[1] = height;
            meta[2] = dimensionNum; 
            meta[3] = dimension;
        }
        this->meta = clCreateBuffer(CONTEXT_CL, CL_MEM_READ_WRITE, 4 * sizeof(int), NULL, &ret);

        ret = clEnqueueWriteBuffer(COMMAND_QUEUE, this->meta, CL_TRUE, 0, 4 * sizeof(int), meta, 0, NULL, NULL);




        delete[] temp;
        delete[] meta;

    }


    void clear() {

        cl_int ret;
        clSetKernelArg(CLEAR_BUFFER_KERNEL, 0, sizeof(cl_mem), &data);
        clSetKernelArg(CLEAR_BUFFER_KERNEL, 1, sizeof(cl_mem), &meta);



        ret = clEnqueueNDRangeKernel(COMMAND_QUEUE, CLEAR_BUFFER_KERNEL, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);
        
        clFinish(COMMAND_QUEUE);
    }
    void threshold(float threshold) {

        
        clSetKernelArg(THRESHOLD, 0, sizeof(cl_mem), &data);
        clSetKernelArg(THRESHOLD, 1, sizeof(cl_mem), &meta);
        //clSetKernelArg(THRESHOLD, 2, sizeof(cl_mem), &threshold);

        clEnqueueNDRangeKernel(COMMAND_QUEUE, THRESHOLD, 2, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);
        clFinish(COMMAND_QUEUE);
    }

    void wipe() {
        clReleaseMemObject(data);
        clReleaseMemObject(meta);
    }
    void copy(buffer b) {
    
        clSetKernelArg(COPY, 0, sizeof(cl_mem), &this->data);
        clSetKernelArg(COPY, 1, sizeof(cl_mem), &b.data);
        clSetKernelArg(COPY, 2, sizeof(cl_mem), &this->meta);

        clEnqueueNDRangeKernel(COMMAND_QUEUE, COPY, 2, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);
        clFinish(COMMAND_QUEUE);
    }


    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }

    float** showBuffer() {

        

        float* temp = new float[width * height];
        if (temp != nullptr)
            clEnqueueReadBuffer(COMMAND_QUEUE, data, CL_TRUE, 0, width * height * sizeof(float), temp, 0, NULL, NULL);
        
        float** RBG_buffer = nullptr;
        
        array1dToArray2d(RBG_buffer, temp, width, height);

        delete[] temp;

        return RBG_buffer;
    }
    void open(std::string fileName, int index) {
        cl_int ret;



        float*** buffer = nullptr;
        
        create3dArray(buffer, 3, width, height);



        // get the size of the required padding
        int padSize = width % 4;

        //empty char to write to
        char* c = new char[4];

        //read past the header file 
        uint8_t headerbytes[54] = {};

        std::ifstream ifs;
        ifs.open(fileName, std::ios::binary);

        if (ifs.good()) {

            ifs.read((char*)headerbytes, sizeof(headerbytes));




            for (unsigned int row = 0; row < height; row++) {
                for (unsigned int col = 0; col < width; col++) {
                    buffer[0][col][row] = float(ifs.get()) / 256;
                    buffer[1][col][row] = float(ifs.get()) / 256;
                    buffer[2][col][row] = float(ifs.get()) / 256;
                }

                ifs.read(c, padSize);
            }
            ifs.close();

            float* cast;
            array2dToArray1d(cast, buffer[index], width, height);
            ret = clEnqueueWriteBuffer(COMMAND_QUEUE, this->data, CL_TRUE, 0, size_t(width * height) * sizeof(float), cast, 0, NULL, NULL);


            
            delete[] cast;
        }
        delete[] buffer;

    }
    void findlastLayerCost(buffer cost, buffer trainingData) {


        cl_int r;
        clSetKernelArg(LAST_LAYER_PROPOGATE, 0, sizeof(cl_mem), &cost.data);
        clSetKernelArg(LAST_LAYER_PROPOGATE, 1, sizeof(cl_mem), &data);
        clSetKernelArg(LAST_LAYER_PROPOGATE, 2, sizeof(cl_mem), &trainingData.data);
        clSetKernelArg(LAST_LAYER_PROPOGATE, 3, sizeof(cl_mem), &meta);

        r = clEnqueueNDRangeKernel(COMMAND_QUEUE, LAST_LAYER_PROPOGATE, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);


        clFinish(COMMAND_QUEUE);

    }
    void findlastLayerCost(cl_mem cost, buffer trainingData) {


        cl_int r;
        clSetKernelArg(LAST_LAYER_PROPOGATE, 0, sizeof(cl_mem), &cost);
        clSetKernelArg(LAST_LAYER_PROPOGATE, 1, sizeof(cl_mem), &data);
        clSetKernelArg(LAST_LAYER_PROPOGATE, 2, sizeof(cl_mem), &trainingData.data);
        clSetKernelArg(LAST_LAYER_PROPOGATE, 3, sizeof(cl_mem), &meta);

        r = clEnqueueNDRangeKernel(COMMAND_QUEUE, LAST_LAYER_PROPOGATE, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


        clFinish(COMMAND_QUEUE);

    }
    void add(buffer input) {


        cl_int r;
        clSetKernelArg(ADD, 0, sizeof(cl_mem), &input.data);
        clSetKernelArg(ADD, 1, sizeof(cl_mem), &data);
        clSetKernelArg(ADD, 2, sizeof(cl_mem), &input.meta);

        r = clEnqueueNDRangeKernel(COMMAND_QUEUE, ADD, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


        clFinish(COMMAND_QUEUE);

    }
    void lightSigmoid(buffer input) {


        cl_int r;
        clSetKernelArg(LIGHT_SIGMOID, 0, sizeof(cl_mem), &input.data);
        clSetKernelArg(LIGHT_SIGMOID, 1, sizeof(cl_mem), &data);
        clSetKernelArg(LIGHT_SIGMOID, 2, sizeof(cl_mem), &input.meta);

        r = clEnqueueNDRangeKernel(COMMAND_QUEUE, LIGHT_SIGMOID, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


        clFinish(COMMAND_QUEUE);

    }
    void dLightSigmoid(buffer input) {


        cl_int r;
        clSetKernelArg(dLIGHT_SIGMOID, 0, sizeof(cl_mem), &input.data);
        clSetKernelArg(dLIGHT_SIGMOID, 1, sizeof(cl_mem), &data);
        clSetKernelArg(dLIGHT_SIGMOID, 2, sizeof(cl_mem), &input.meta);

        r = clEnqueueNDRangeKernel(COMMAND_QUEUE, dLIGHT_SIGMOID, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


        clFinish(COMMAND_QUEUE);

    }

    void subtract(buffer input) {


        cl_int r;
        clSetKernelArg(SUBTRACT, 0, sizeof(cl_mem), &input.data);
        clSetKernelArg(SUBTRACT, 1, sizeof(cl_mem), &data);
        clSetKernelArg(SUBTRACT, 2, sizeof(cl_mem), &input.meta);

        r = clEnqueueNDRangeKernel(COMMAND_QUEUE, SUBTRACT, 2, NULL, GLOBAL_ITEM_SIZE2d, LOCAL_ITEM_SIZE2d, 0, NULL, NULL);


        clFinish(COMMAND_QUEUE);

    }




};