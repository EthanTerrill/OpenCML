#pragma once
#pragma comment(lib, "OpenCL.lib")

//intellisense marks opencl with a bunch of warnings
//that I cant really do anything about and its really annoying 
// so i have this here to clear the errors
#pragma warning(disable: C26439 )
#include <CL/OpenCl.hpp>
#pragma warning(default: C26439 )

//required libraries
#include <fstream>
#include <filesystem>
#include <cassert>

//OS specific stuff for interacting with file systems and whatever the fuck else
#ifdef _WIN32
    #include<windows.h>

    void writeHeader(std::ostream& out, int width, int height) {

        BITMAPFILEHEADER tWBFH = *new BITMAPFILEHEADER();
        tWBFH.bfType = 0x4d42;
        tWBFH.bfSize = 14 + 40 + (width * height * 3);
        tWBFH.bfReserved1 = 0;
        tWBFH.bfReserved2 = 0;
        tWBFH.bfOffBits = 14 + 40;

        BITMAPINFOHEADER tW2BH;
        memset(&tW2BH, 0, 40);
        tW2BH.biSize = 40;
        tW2BH.biWidth = width;
        tW2BH.biHeight = height;
        tW2BH.biPlanes = 1;
        tW2BH.biBitCount = 24;
        tW2BH.biCompression = 0;

        out.write((char*)(&tWBFH), 14);
        out.write((char*)(&tW2BH), 40);
    }
#else ifdef __LINUX__
#endif


enum layerType{

    DNN,
    CONVOLUTIONAL_LAYER,
    AVG_POOL_LAYER,
    MAX_POOL_LAYER,
    MIN_POOL_LAYER,
    SCALE_UP_LAYER,
    SIGMOID_LAYER

};



#include "OpenCLstuff.cpp"
#include "functions.cpp"
#include "buffer.cpp"
#include "image.cpp"
#include "DataSet.cpp"



#include "CNN kernel.cpp"
#include "CNN layer.cpp"
//#include "DNN node.cpp"
#include "DNN layer.cpp"

#include "scaleUpLayer.cpp"
#include "avgPoolLayer.cpp"


#include "layer.cpp"
#include "CNN.cpp"


