#include <iostream>
#include <vector>

#include <cuda.h>
#include <vector_types.h>

#include "bitmap_image.hpp"

using namespace std;

__global__ void color_to_grey(uchar3 *input_image, uchar3 *output_image, int width, int height)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col < width && row < height)
    {
        int pos = row * width + col;
        output_image[pos].x = static_cast<unsigned char>(input_image[pos].x * 0.2126f + input_image[pos].y * 0.7125f + input_image[pos].z * 0.0722f);
    }
}


int main()
{
    bitmap_image bmp("lenna.bmp");

    if(!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    
    cout << "image dimensions" << endl;
    cout << "height " << height << " width " << width << endl;

    //Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(width / 16), ceil(height / 16), 1);
    dim3 dimBlock(16, 16, 1);

    color_to_grey<<< dimGrid , dimBlock >>> (d_in, d_out, width, height);

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    
    //Set updated pixels
    for(int x = 0; x < width; x++)
    {
        for(int y = 0; y < height; y++)
        {
            int pos = x * width + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }
    
    bmp.save_image("./grey_scaled.bmp");

    cudaFree(d_in);
    cudaFree(d_out);
}