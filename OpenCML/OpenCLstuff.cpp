#pragma once

cl_kernel CLEAR_BUFFER_KERNEL;
cl_kernel CONVOLVE;
cl_kernel CONVOLVE_180;
cl_kernel COPY;
cl_kernel LAST_LAYER_PROPOGATE;
cl_kernel ADD;
cl_kernel SUBTRACT;
cl_kernel SUBTRACT_AND_CLEAR_B;
cl_kernel SUBTRACT_AND_CLEAR;
cl_kernel THRESHOLD;
cl_kernel LIGHT_SIGMOID;
cl_kernel dLIGHT_SIGMOID;
cl_kernel SOLVE_dKERNELS;
cl_kernel AVERAGE_POOL;
cl_kernel SCALE_UP;
cl_kernel AVERAGE_POOL_COST;
cl_kernel SCALE_UP_COST;
cl_kernel DNN_PROPGATE;
cl_kernel DNN_BACKPROPGATE;
cl_kernel DNN_GETWCOSTS;
cl_kernel DNN_GETBIASCOSTS;




cl_context CONTEXT_CL;
cl_command_queue COMMAND_QUEUE;

/*
    the Global ItemSize determines the number
    of threads in each work Item and the Local
    Item size determines the number of Items in 
    each group.


    The definitions here are of 2d and 1d fucntions
    respectively

*/
const size_t GLOBAL_ITEM_SIZE2d[2]  = { 256, 256 };
const size_t LOCAL_ITEM_SIZE2d[2]   = { 8, 8 };

const size_t GLOBAL_ITEM_SIZE       = 256 * 256;
const size_t LOCAL_ITEM_SIZE        = 64;




/*
    setGPUFunctions is responsible for intializing all
    kernels responsible for the 
*/


void buildKernels(cl_program program) {

    ////////////////////////////////////////////////////////////////
   //                  Build Individual kernels
   /////////////////////////////////////////////////////////////
    cl_int          ret = 0;
    
    assert(program != nullptr);

    CONVOLVE                = clCreateKernel(program, "convolve",           &ret); assert(ret == 0);
    CLEAR_BUFFER_KERNEL     = clCreateKernel(program, "clearBuffer",        &ret); assert(ret == 0);
    CONVOLVE_180            = clCreateKernel(program, "convolve_180",       &ret); assert(ret == 0);
    LAST_LAYER_PROPOGATE    = clCreateKernel(program, "lastLayerPropogate", &ret); assert(ret == 0);
    ADD                     = clCreateKernel(program, "add",                &ret); assert(ret == 0);
    SUBTRACT                = clCreateKernel(program, "subtract",           &ret); assert(ret == 0);
    SUBTRACT_AND_CLEAR      = clCreateKernel(program, "subtractAndClear",   &ret); assert(ret == 0);
    THRESHOLD               = clCreateKernel(program, "threshold",          &ret); assert(ret == 0);
    LIGHT_SIGMOID           = clCreateKernel(program, "lightSigmoid",       &ret); assert(ret == 0);
    dLIGHT_SIGMOID          = clCreateKernel(program, "dLightSigmoid",      &ret); assert(ret == 0);
    SOLVE_dKERNELS          = clCreateKernel(program, "solvedKernels",      &ret); assert(ret == 0);

    AVERAGE_POOL            = clCreateKernel(program, "avgPool",            &ret); assert(ret == 0);
    SCALE_UP                = clCreateKernel(program, "scaleUp",            &ret); assert(ret == 0);
    AVERAGE_POOL_COST       = clCreateKernel(program, "avgPoolCost",        &ret); assert(ret == 0);
    SCALE_UP_COST           = clCreateKernel(program, "scaleUpCost",        &ret); assert(ret == 0);

    DNN_PROPGATE            = clCreateKernel(program, "DNN_propogate",      &ret); assert(ret == 0);
    DNN_BACKPROPGATE        = clCreateKernel(program, "DNN_getprevCosts",   &ret); assert(ret == 0);
    DNN_GETWCOSTS           = clCreateKernel(program, "DNN_getWeightCosts", &ret); assert(ret == 0);
    DNN_GETBIASCOSTS        = clCreateKernel(program, "DNN_getBiasCosts",   &ret); assert(ret == 0);

}

void readKernelFile(FILE* f, size_t  source_size, char* source_str, const int max_source_size) {

    //////////////////////////////////////////////////////////
    //             Read Kernel Data
    ///////////////////////////////////////////////////////

    errno_t error = fopen_s(&f, "SkyNet\\kernels.cl", "r");

    assert(f);
    assert(source_str != NULL);
    assert(!error);
    source_size = fread(source_str, 1, max_source_size, f);
    fclose(f);
    assert(source_size > 0);
}

void setGPUfunctions() {


   //////////////////////////////////////////////////////////
   //             Constants
   ///////////////////////////////////////////////////////

    const int max_source_size(0x100000);

    //char pointer bad not like
    char* source_str = (char*)malloc(max_source_size);


    cl_device_id    deviceId            = NULL;
    cl_int          ret                 = 0;
    cl_uint         ret_num_devices     = NULL;
    cl_uint         ret_num_platforms;
    size_t          source_size;

    //I dont like the file pointer, Speeed isnt an issue here so I think
    //Im going to replace it with an std::string sometime
    FILE*           f;



    //////////////////////////////////////////////////////////
    //             Read Kernel Data
    ///////////////////////////////////////////////////////

    errno_t error = fopen_s(&f, "SkyNet\\kernels.cl", "r");

    assert(f);
    assert(source_str != NULL);
    assert(!error);
    source_size = fread(source_str, 1, max_source_size, f);
    fclose(f);
    assert(source_size > 0);

    //////////////////////////////////////////////////////////
    //    Get platform data and create  enviorenment
    ///////////////////////////////////////////////////////

    cl_platform_id* platforms = (cl_platform_id*)malloc(ret_num_devices * sizeof(cl_platform_id));

    clGetPlatformIDs(0, NULL, &ret_num_platforms);
    clGetPlatformIDs(ret_num_platforms, platforms, NULL);

    if (platforms != NULL)
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &deviceId, &ret_num_devices);


    CONTEXT_CL = clCreateContext(NULL, ret_num_devices, &deviceId, NULL, NULL, &ret);
    COMMAND_QUEUE = clCreateCommandQueue(CONTEXT_CL, deviceId, 0, &ret);


    cl_program program = clCreateProgramWithSource(CONTEXT_CL, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
    assert(ret == 0);
    ret = clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);




    //////////////////////////////////////////////////////////
    //             Check for Build Errors
    ///////////////////////////////////////////////////////


    if (ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;

        //get build errors
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        //display errors
        std::cerr << log;

        MessageBox(NULL, (LPCWSTR)L"Error: could not compile CL function...", (LPCWSTR)L"Fatal Program error", MB_ICONSTOP);
    }


    
    assert(ret == 0);
    buildKernels(program);
   
    free(source_str);


}

