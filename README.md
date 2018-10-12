# Grayscale Images


Convert photos from RGB to black and white. 

Currently only supports bitmap images.

| Color  | Grayscale |
| ------------- | ------------- |
| ![color image](lenna.bmp)  | ![grayscaled image](grayscaled.bmp)  |


## How to Run

```
nvcc --std=c++11 grayscale.cu -o grayscale
./grayscale
```
