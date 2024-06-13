#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 1

__global__ void sobel(const unsigned char *imageIn, unsigned char *imageOut, const int width, const int height)
{
    int x_ix = blockIdx.x * blockDim.x + threadIdx.x;
    int y_ix = blockIdx.y * blockDim.y + threadIdx.y;
    
    int s[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

    for (int i = -1; i <= 1; i++) {
        
        int curr_x_ix = x_ix + i;
        if (curr_x_ix < 0 || curr_x_ix >= width) {
            continue;
        }

        for (int j = -1; j <= 1; j++) {
            
            int curr_y_ix = y_ix + j;
            if (curr_y_ix < 0 || curr_y_ix >= height) {
                continue;
            }


            int curr_image_index = curr_y_ix * width + curr_x_ix;

            int currVal = (int) imageIn[curr_image_index];
            s[j+1][i+1] = currVal;
        }

    }

    int xDer = -s[0][0] -2*s[1][0] -s[2][0] +s[0][2] +2*s[1][2] +s[2][2];
    int yDer = s[0][0] +2*s[0][1] + s[0][2] -s[2][0] -2*s[2][1] -s[2][2];
    
    float d = sqrt(xDer * xDer + yDer * yDer);
    int dFloored = (int) d;
    if (dFloored > 255) {
        dFloored = 255;
    }

    int curr_image_index = y_ix * width + x_ix;
    imageOut[curr_image_index] = dFloored; //255;

}


// __global__ void vectorDistance(float *c, const float *a, const float *b, int len)
// {
// 	// računanje razlike
// 	int gid = blockIdx.x * blockDim.x + threadIdx.x;
// 	while (gid < len) {
// 		c[gid] = a[gid] - b[gid];
// 		gid += gridDim.x * blockDim.x;
// 	}
// }







int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        printf("USAGE: sample input_image output_image\n");
        exit(EXIT_FAILURE);
    }
    
    char szImage_in_name[255];
    char szImage_out_name[255];

    snprintf(szImage_in_name, 255, "%s", argv[1]);
    snprintf(szImage_out_name, 255, "%s", argv[2]);

    // Load image from file and allocate space for the output image
    int width, height, cpp;
    unsigned char *h_imageIn = stbi_load(szImage_in_name, &width, &height, &cpp, COLOR_CHANNELS);
    cpp = COLOR_CHANNELS;

    


    // printf("Trying to crash this: %c", h_imageIn[0]);





    if (h_imageIn == NULL)
    {
        printf("Error reading loading image %s!\n", szImage_in_name);
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", szImage_in_name, width, height);
    const size_t datasize = width * height * cpp * sizeof(unsigned char);
    unsigned char *h_imageOut = (unsigned char *)malloc(datasize);
    
    printf("width, height, cpp: %d, %d, %d.\n", width, height, cpp);
    printf("datasize: %d.\n", datasize);


    // Kot preizkus samo kopiramo vhodno sliko v izhodno
    // memcpy(h_imageOut,h_imageIn,datasize);

    // Nastavimo organizacijo niti v 2D
    int sizeOfBlockOneDim = 16; // 256

    dim3 blockSize(sizeOfBlockOneDim, sizeOfBlockOneDim);
    dim3 gridSize((width / sizeOfBlockOneDim) +1, (height / sizeOfBlockOneDim) +1);

    unsigned char *d_imageIn;
    unsigned char *d_imageOut;

    // Rezervacija pomnilnika na napravi
    checkCudaErrors(cudaMalloc(&d_imageIn, datasize));
    checkCudaErrors(cudaMalloc(&d_imageOut, datasize));

    // prenesemo sliko iz gostitelja na napravo
	checkCudaErrors(cudaMemcpy(d_imageIn, h_imageIn, datasize, cudaMemcpyHostToDevice));

    // Uporabimo dogodke CUDA za merjenje casa
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Zazenemo scepec
    cudaEventRecord(start);
    sobel<<<gridSize, blockSize>>>(d_imageIn, d_imageOut, width, height);
    getLastCudaError("do_nothing() execution failed\n");
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    // pocakamo, da vse niti na napravi zakljucijo
	checkCudaErrors(cudaDeviceSynchronize());

    
    // printf("sizeof(h_imageOut) before: %d.\n", sizeof(h_imageOut));


	checkCudaErrors(cudaMemcpy(h_imageOut, d_imageOut, datasize, cudaMemcpyDeviceToHost));

    // printf("sizeof(h_imageOut): %d.\n", sizeof(h_imageOut));


    // Izpisemo cas
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution time is: %0.3f milliseconds \n", milliseconds);

    // Zapisemo izhodno sliko v datoteko
    char szImage_out_name_temp[255];
    strncpy(szImage_out_name_temp, szImage_out_name, 255);
    char *token = strtok(szImage_out_name_temp, ".");
    char *FileType = NULL;
    while (token != NULL)
    {
        FileType = token;
        token = strtok(NULL, ".");
    }

    if (!strcmp(FileType, "png"))
        stbi_write_png(szImage_out_name, width, height, cpp, h_imageOut, width * cpp);
    else if (!strcmp(FileType, "jpg"))
        stbi_write_jpg(szImage_out_name, width, height, cpp, h_imageOut, 100);
    else if (!strcmp(FileType, "bmp"))
        stbi_write_bmp(szImage_out_name, width, height, cpp, h_imageOut);
    else
        printf("Error: Unknown image format %s! Only png, bmp, or bmp supported.\n", FileType);

    // Sprostimo pomnilnik na napravi
    checkCudaErrors(cudaFree(d_imageIn));
    checkCudaErrors(cudaFree(d_imageOut));

    // Pocistimo dogodke
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
    // Sprostimo pomnilnik na gostitelju
    free(h_imageIn);
    free(h_imageOut);

    return 0;
}
