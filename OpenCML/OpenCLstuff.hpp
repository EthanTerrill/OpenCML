#include <CL/opencl.hpp>
#include <assert.h>
#include <cerrno>
#include <stdio.h>



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




cl_context        CONTEXT_CL;
cl_command_queue  COMMAND_QUEUE;

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




void buildKernels(cl_program program);
void readKernelFile(FILE* f, size_t  source_size, char* source_str, const int max_source_size);
void setGPUfunctions();
