#pragma once



//This entire thing is fucking cursed it needs to be rewritten ASAP



uint32_t getBmpWidth(std::string bitmap) {


    assert(bitmap != "");


    std::ifstream ifs;
    ifs.open(bitmap, std::ios::binary);

    //read the header file 
    uint8_t headerbytes[54] = {};
    ifs.read((char*)headerbytes, sizeof(headerbytes));

    ifs.close();

    uint32_t width = *(uint32_t*)(headerbytes + 18);

    return width;
}

uint32_t getBmpHeight(std::string bitmap) {
    

    assert(bitmap != "");

    std::ifstream ifs;
    ifs.open(bitmap, std::ios::binary);

    //read the header file 
    uint8_t headerbytes[54] = {};
    ifs.read((char*)headerbytes, sizeof(headerbytes));

    ifs.close();
    uint32_t height = *(uint32_t*)(headerbytes + 22);

    return height;
}

void create2dArray(float** &arr, int width, int height) {

    assert(width > 0);
    assert(height > 0);


    arr = new float* [width];
    for (int i = 0; i < width; i++)
        arr[i] = new float[height];
};


//  NOTE THIS IS A VERY DANGEROUS FUNCTION TO USE
//  USE ONLY WHEN ABSOLUTLY NECASSARY

void create3dArray(float*** &arr, int dimensionNum,  int width, int height) {


    assert(width > 0);
    assert(height > 0);
    assert(dimensionNum > 0);


    arr = new float** [dimensionNum];

    for (int i = 0; i < dimensionNum; i++)
        create2dArray(arr[i], width, height);


};

void copyArray(float* &copy, float* arr, int length) {

    assert(arr != nullptr);
    assert(length > 0);
    
    copy = new float [length];

    for (int i = 0; i < length; i++)
        copy[i] = arr[i];

};
void copy2dArray(float** &copy, float** arr, int width, int height) {
    
    assert(arr != nullptr);
    assert(width > 0);
    assert(height > 0);

    copy = new float* [width];

    for (int i = 0; i < width; i++)
        copyArray(copy[i], arr[i], height);
};

void copy3dArray(float*** copy, float*** arr, int dimensionNum, int width, int height) {


    assert(arr != nullptr);
    assert(dimensionNum > 0);
    assert(width > 0);
    assert(height > 0);



    copy = new float** [dimensionNum];

    for (int i = 0; i < dimensionNum; i++)
         copy2dArray(copy[i], arr[i], width, height);


};

void array2dToArray1d(float*  &arr1d, float** arr2d, int width, int height) {

    assert(arr2d != nullptr);
    assert(width > 0);
    assert(height > 0);


    arr1d = new float [width * height];

    for (int i = 0; i < width; i++)
        for(int j = 0; j < height; j++)
            arr1d[i * height + j] = arr2d[i][j];

};

void array1dToArray2d(float** & arr2d, float* arr1d, int width, int height) {

    assert(arr1d != nullptr);


    arr2d = nullptr;
    
    create2dArray(arr2d, width, height);

    for (int i = 0; i < width; i++)
        for (int j = 0; j < height; j++)
              arr2d[i][j] = arr1d[i * height + j];


};


void writeBufferToBmp(std::string fileName, int width, int height, float*** buffer) {


    int padSize = width % 4;


    char* c = new char[4]{ 0, 0 ,0 ,0 };

    std::ofstream ofs;
    ofs.open(fileName, std::ios::binary);


    writeHeader(ofs, width, height);
    float pixel = 0;


    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int i = 0; i < 3; i++) {
                pixel = buffer[i][col][row];
                pixel = (pixel < 0) * 0 + (pixel >= 0) * pixel;
                pixel = (pixel < 1) * pixel + (pixel >= 1) * 1;
                ofs << unsigned char(pixel * 255);
            }
        }
        ofs.write(c, padSize);
    }
    ofs.close();
    
}

void subtractandclear(cl_mem input, cl_mem output, cl_mem metadata) {


    cl_int r;
    clSetKernelArg(SUBTRACT_AND_CLEAR, 0, sizeof(cl_mem), &input);
    clSetKernelArg(SUBTRACT_AND_CLEAR, 1, sizeof(cl_mem), &output);
    clSetKernelArg(SUBTRACT_AND_CLEAR, 2, sizeof(cl_mem), &metadata);

    r = clEnqueueNDRangeKernel(COMMAND_QUEUE, SUBTRACT_AND_CLEAR, 1, NULL, &GLOBAL_ITEM_SIZE, &LOCAL_ITEM_SIZE, 0, NULL, NULL);


    clFinish(COMMAND_QUEUE);


}