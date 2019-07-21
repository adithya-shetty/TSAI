# Assignment 1B

### What are Channels and Kernels (according to EVA)?
> According to my understand, `Channel` is an entity we get by splitting an image. For e.g., We can split a color image into 3 channels of Red, Green and Blue. For printing process, images are formed by combination of 4 channels(CMYK). A grayscale image contains only single channel which contains different shades of gray. There can also be an alpha channel which can be used for transperency control. So each channel has some characteristic feature of the image. So for the formation of the images, all these characteristics must come together. A `Kernel` can be considered as a function which when **convolves** over an image has the ability to change is properties or highlight some features of the image. Usually represented as a matrix, it has many names like convolution matrix, mask, feature extracter, etc. It is of odd dimensions.

### Why should we only (well mostly) use 3x3 Kernels?
- First of all we cannot use even shaped kernels. The reason for that is that they do not have a center element. Hence they do not have symmetry which is needed for most of the feature extraction.
- We can we higher odd shaped kernels like 5x5 and 7X7, but why 3X3 is preffered is because we can build higher kernel functionality by using multiple 3X3 kernels. For e.g. To extract features similar to 5X5 kernel, we can use two 3X3 kernels. Instead of 7X7 kernel we can use three 3X3 kernels and so on.

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)
> Every convolution of a 3X3 kernel over an image reduces the image dimensions by 2. For example, In the first step, when we convolve 3X3 kernel over the 199X199 image, we will get 197X197. This is continue till 1X1 is reached. So we can easy derive that there should be `199//2 => 99` steps to reach the final result.

**Hope this is helpful**
