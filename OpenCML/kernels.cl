__kernel void clearBuffer(__global float* buffer, __constant int* meta) {


   // printf("%d, %d", meta[0], meta[1]);


    int width  = meta[0];
    int height = meta[1];   
    for(int i = get_global_id(0); i < width * height; i += get_global_size(0))
    {
        buffer[ i ] = 0; 
    }

    //rintf("%d, %d", meta[0], meta[1]);

}

__kernel void convolve
(
    const __global float* Kernel,
    const __global float* inputBuffer, 
    __global float* outputBuffer, 
    const __global int* inpBufferMetaData,
    const __global int* kernelMetaData
)
{
    int kWidth  = kernelMetaData[0];
    int kHeight = kernelMetaData[1];

    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int d  = inpBufferMetaData[3];  

    int outputWidth  = (width - kWidth + 1);
    int outputHeight = (height - kHeight + 1); 
    
    if(outputWidth * outputHeight > kWidth * kHeight){
        
        for(int i = get_global_id(0); i < outputWidth; i += get_global_size(0))
        {
        
            for(int j = get_global_id(1); j < outputHeight; j += get_global_size(1))
            {
                float pixel = 0;
                for(int x = 0; x < kWidth; x++)
                {
                    for(int y = 0; y < kHeight; y++)
                    {
                        pixel += inputBuffer[ (i + x) * (height) + ( j + y) ] * Kernel[y * kWidth + x];
                    }
                }
                outputBuffer[i * outputHeight + j] += pixel;
        
            }
        
        }
    }
    else
    {
        for(int i = 0; i < outputWidth; i += 1)
        {
        
            for(int j = 0; j < outputHeight; j += 1)
            {
                float pixel = 0;
                for(int x = get_global_id(0); x < kWidth; x+= get_global_size(0))
                {
                    for(int y = get_global_id(1); y < kHeight; y+= get_global_size(1))
                    {
                        
                        pixel += inputBuffer[ (i + x) * (height) + ( j + y) ] * Kernel[y * kWidth + x];
                    }
                }

                outputBuffer[i * outputHeight + j] += pixel;
        
            }
        
        }        
    }


}

__kernel void convolve_180
(
    const __global float* Kernel, 
    const __global float* inputBuffer, 
          __global float* outputBuffer, 
    const __global int* inpBufferMetaData,
    const __global int* kernelMetaData
)
{
    int kWidth  = kernelMetaData[0];
    int kHeight = kernelMetaData[1];

    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int d      = inpBufferMetaData[2];


    int outputWidth  = (width + kWidth - 1);
    int outputHeight = (height + kHeight - 1); 
    
    if(outputWidth * outputHeight > kWidth * kHeight || true){

        for(int i = get_global_id(0); i < outputWidth; i += get_global_size(0))
        {
        
            for(int j = get_global_id(1); j < outputHeight; j += get_global_size(1))
            {
                float pixel = 0;
                for(int x = 0; x < kWidth; x++)
                {
                    for(int y = 0; y < kHeight; y++)
                    {
                        int k = i + x - kWidth  + 1;
                        int l = j + y - kHeight + 1;

                        if( k >= 0 && k < width && l >= 0 && l <height)
                        pixel += inputBuffer[ (k) * (height) + (l) ] * Kernel[(kWidth - y - 1) * kWidth + (kWidth - x - 1)];

                       
                    }
                }
                outputBuffer[i * outputHeight + j] += pixel/(d );
        
            }
        
        }
    }
    else
    {
        for(int i = 0; i <outputWidth; i += 1)
        {
        
            for(int j = 0; j < outputHeight; j += 1)
            {
                float pixel = 0;
                for(int x = get_global_id(0); x < kWidth; x+= get_global_size(0))
                {
                    for(int y = get_global_id(1); y < kHeight; y+= get_global_size(1))
                    {
                        int k = i + x - kWidth  + 1;
                        int l = j + y - kHeight + 1;

                        if( k >= 0 && k < width && l >= 0 && l <height)
                        pixel += inputBuffer[ (k) * (height) + (l) ] * Kernel[(kWidth - x - 1) * kWidth + (kWidth - y - 1)];
                    }
                }

                outputBuffer[i * outputHeight + j] += pixel/(d);
            }
        
        }        
    }
    
    



}

__kernel void add
(
    __global float* inpBuffer,
    __global float* outBuffer,
    __constant int* inpBufferMetaData
)
{


    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    for(int i = get_global_id(0); i < width * height; i += get_global_size(0))
    {
        outBuffer[i] += inpBuffer[i];

        
    }

}
__kernel void subtract
(
    __global float* inpBuffer,
    __global float* outBuffer,
    __global int* inpBufferMetaData
)
{


    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];

    for(int i = get_global_id(0); i < width; i += get_global_size(0))
    {
        
        for(int j = get_global_id(1); j < height; j += get_global_size(1))
        {
            
            outBuffer[i * height + j] -= outBuffer[i * height + j];
        
        }
        
    }

}
__kernel void subtractAndClear
(
    __global float* inpBuffer,
    __global float* outBuffer,
    __constant int* inpBufferMetaData,
    __constant float* learningRate
)
{
    float lr = learningRate[0];
    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];

    for(int i = get_global_id(0); i < width * height ; i += get_global_size(0))
    {
        
        outBuffer[i] -= inpBuffer[i ] *  lr;
        inpBuffer[i] = 0;
    }

}
__kernel void subtractBias
(
    __global float* inpBuffer,
    __global float* outBuffer,
    __constant int* inpBufferMetaData,
    __constant float* learningRate
)
{
    float lr = learningRate[0];
    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];

    for(int i = get_global_id(0); i < width * height; i += get_global_size(0))
    {
        
        outBuffer[i] -= inpBuffer[i ] *  lr;
        inpBuffer[i] = 0;
    }

}

__kernel void lastLayerPropogate
(
    __global float* costBuffer,
    __constant float* y,
    __constant float* z,
    __constant int* inpBufferMetaData
)
{


    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];

    for(int i = get_global_id(0); i < width * height; i += get_global_size(0))
    { 
        costBuffer[i] += 2 * (y[i] - z[i]); 
    }

}
__kernel void threshold
(
    __global float* buffer,
    __global int* metaData
)
{


    int width  = metaData[0];
    int height = metaData[1];

    for(int i = get_global_id(0); i < width; i += get_global_size(0))
    {
        
        for(int j = get_global_id(1); j < height; j += get_global_size(1))
        {
            
            if(buffer[i * height + j] < 0.1){

                buffer[i * height + j] = 0;
            } 
            else{
                buffer[i * height + j] = 1;
            }
        }
        
    }

}

 __kernel void lightSigmoid
(
    __constant float* inpBuffer,
    __global float*   outBuffer,
    __constant int*     inpBufferMetaData
)
{

    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1]; 

    for(int i = get_global_id(0); i < height; i += get_global_size(0))
    {
        
        
            private float val = inpBuffer[i];
            private float abs = (val >= 0) * val + (val < 0) * -val;
            outBuffer[i] = (val/(abs+1) + 1) * 0.5; 
        
        
    }

}

__kernel void dLightSigmoid
(
    __constant float* inpBuffer,
    __global float* outBuffer,
    __constant int* inpBufferMetaData
)
{
    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1]; 

    for(int i = get_global_id(0); i < height; i += get_global_size(0))
    {
        
        
            private float val = inpBuffer[i];
            private float abs = (val >= 0) * val + (val < 0) * -val;
            private float add = abs + 1;
            outBuffer[i] = 1/(2 * add * add); 
         
    }
    
}


//vecXvec = vec
__kernel void DNN_getBiasCosts
(
    __constant float* inpBuffer,
    __global float* outBuffer,
    __constant float* activationCost,
    __constant int* inpBufferMetaData
)
{
    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1]; 

    //get f'(x) and assign it to outBuffer
    for(int i = get_global_id(0); i < height; i += get_global_size(0))
    {
        
        
        private float val = inpBuffer[i];
        private float abs = (val >= 0) * val + (val < 0) * -val;
        private float add = abs + 1;
        outBuffer[i] = 1.0/(2 * add * add); 
         
    }

    for(int i = get_global_id(0); i < height; i += get_global_size(0))
    {  
        outBuffer[i] = activationCost[i] * outBuffer[i];
    } 
}

//vecXvec = mat 
__kernel void DNN_getWeightCosts
(
    __global float* dweights,
    __constant float* inputBuffer, 
    __constant float* dBias, 
    __constant int* metaData,
    __constant int* inpMetaData
)
{

    int inpWidth   = metaData[0];
    int outpWidth  = metaData[1];
    int d          = inpMetaData[3];  
    for(int i = get_global_id(0); i < outpWidth; i += get_global_size(0))
    { 
        for(int j = get_global_id(1); j < inpWidth; j += get_global_size(1) )
        {

            dweights[ (j) * (outpWidth) + (i) + d * inpWidth * outpWidth] += inputBuffer[ j ] * dBias[i];
           // dweights[ (j) * (outpWidth) + (i) + d * inpWidth * outpWidth] = 0; 

        }
    }
}  
 
__kernel void DNN_getprevCosts
(
    __global float* prevLayerCosts,
    __constant float* weights, 
    __constant float* dBias, 
    __constant int* metaData,
    __constant int* inpMetaData
)
{

    int inpWidth   = metaData[0];
    int outpWidth  = metaData[1];
    int d          = inpMetaData[3];
   
    for(int i = get_global_id(0); i < outpWidth; i += get_global_size(0))
    {
        float pixel = 0;
        for(int j = get_global_id(1); j < inpWidth; j += get_global_size(1) )
        {

            prevLayerCosts[j] = weights[ (j) * (outpWidth) + (i) + d * inpWidth * outpWidth] * dBias[i];

        }
    }
    
               
}  
__kernel void solvedKernels
(
    const __global float* Kernel,
    const __global float* inputBuffer, 
    __global float* outputBuffer, 
    const __global int* inpBufferMetaData,
    const __global int* kernelMetaData
)
{
    int kWidth  = kernelMetaData[0];
    int kHeight = kernelMetaData[1];
    


    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int dNum   = inpBufferMetaData[2];


    int outputWidth  = (width - kWidth + 1);
    int outputHeight = (height - kHeight + 1);
    int scale        = width * height;


    if(outputWidth * outputHeight > kWidth * kHeight){
        for(int i = get_global_id(0); i < outputWidth; i += get_global_size(0))
        {
        
            for(int j = get_global_id(1); j < outputHeight; j += get_global_size(1))
            {
                float pixel = 0;
                for(int x = 0; x < kWidth; x++)
                {
                    for(int y = 0; y < kHeight; y++)
                    {
                        pixel += inputBuffer[ (i + x) * (height) + ( j + y) ] * Kernel[y * kWidth + x];
                    }
                }
                outputBuffer[i * outputHeight + j] += pixel/(scale * dNum);
                
            }
        
        }
    }
    else
    {
        for(int i = 0; i < outputWidth; i += 1)
        {
        
            for(int j = 0; j < outputHeight; j += 1)
            {
                float pixel = 0;
                for(int x = get_global_id(0); x < kWidth; x+= get_global_size(0))
                {
                    for(int y = get_global_id(1); y < kHeight; y+= get_global_size(1))
                    {
                        
                        pixel += inputBuffer[ (i + x) * (height) + ( j + y) ] * Kernel[y * kWidth + x];
                    }
                }
                outputBuffer[i * outputHeight + j] += pixel/(scale * dNum);
        
            }
        
        }        
    }


}

__kernel void avgPool
(
    __constant float* inpBuffer,
    __global float* outBuffer,
    __constant int* inpBufferMetaData
)
{


    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int size   = inpBufferMetaData[2];
    int outpHeight = height/size;

    float scale = 1.0/(size * size * 3);

    for(int i = get_global_id(0); i < width/size; i += get_global_size(0))
    {
        
        for(int j = get_global_id(1); j < height/size; j += get_global_size(1))
        {
            
            for(int x = 0; x < size; x++)
            {
                for(int y = 0; y < size; y++)
                {
                    outBuffer[i * outpHeight + j] += inpBuffer[(i * size + x) * height + size * j + y]*(scale);
                    
                }
            }
        
        }
        
    }

}
__kernel void avgPoolCost
(
    __global float* outBuffer,
    __constant float* inpBuffer,
    __constant int* inpBufferMetaData
)
{
    
    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int size   = inpBufferMetaData[2];
    int dim    = inpBufferMetaData[3];
    int inpHeight = height/size;


    for(int i = get_global_id(0); i < width/size; i += get_global_size(0))
    {
        
        for(int j = get_global_id(1); j < height/size; j += get_global_size(1))
        {
            
            for(int x = 0; x < size; x++)
            {
                for(int y = 0; y < size; y++)
                {
                    outBuffer[(i * size + x) * height + size * j + y]  += inpBuffer[i * inpHeight + j]/dim;
                    
                }
            }
        
        }
        
    }
    

}
__kernel void scaleUp
(
    __global float* outBuffer,
    __constant float* inpBuffer,
    __constant int* inpBufferMetaData
)
{
    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int size   = inpBufferMetaData[2];
    int inpHeight = height/size;


    for(int i = get_global_id(0); i < width/size; i += get_global_size(0))
    {
        
        for(int j = get_global_id(1); j < height/size; j += get_global_size(1))
        {
            
            for(int x = 0; x < size; x++)
            {
                for(int y = 0; y < size; y++)
                {
                    outBuffer[(i * size + x) * height + size * j + y]  += inpBuffer[i * inpHeight + j];
                   
                    //printf("%f", outBuffer[(i * size + x) * height + size * j + y]);
                }
            }
        
        }
        
    }
    

}
__kernel void scaleUpCost
(
    __constant float* inpBuffer,
    __global float* outBuffer,
    __constant int* inpBufferMetaData
)
{


    int width  = inpBufferMetaData[0];
    int height = inpBufferMetaData[1];
    int size   = inpBufferMetaData[2];
    int dim    = inpBufferMetaData[3];

    int outpHeight = height/size;

    int scale = size * size;

    for(int i = get_global_id(0); i < width/size; i += get_global_size(0))
    {
        
        for(int j = get_global_id(1); j < height/size; j += get_global_size(1))
        {
            
            for(int x = 0; x < size; x++)
            {
                for(int y = 0; y < size; y++)
                {
                    outBuffer[i * outpHeight + j] += inpBuffer[(i * size + x) * height + size * j + y]/(scale * dim);
                    
                }
            }
        
        }
        
    }

}

__kernel void DNN_propogate
(
    __constant float* weights,
    __constant float* inputBuffer, 
    __global float* outputBuffer, 
    __constant int* metaData,
    __constant int* inpMetaData
)
{

    int inpWidth   = inpMetaData[0] *  inpMetaData[1];
    int outpWidth  = metaData[0] * metaData[1];
    int d          = inpMetaData[3];


    int place = d * inpWidth * outpWidth;
   
    
    for(int i = get_global_id(0); i < outpWidth; i += get_global_size(0))
    {
        float pixel = 0;
        for(int j = get_global_id(1); j < inpWidth; j+= get_global_size(1))
        {

            pixel += inputBuffer[ j ] * weights[ (j) * (outpWidth) + (i) + place];

        }
        outputBuffer[i] = pixel;
    }
}  
 