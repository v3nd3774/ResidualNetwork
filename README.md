# ResidualNetwork
Repo to store work on RNNs: [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

The following graph shows AUC by by the one-vs-all strategy. The y-axis represents the AUC achieved for a distinct class,
while the x-axis represents the number of batches of size 128 used to train the model.

![Residual Network Results](/comparison.png)

The top and bottom axes contain the plain and resnet neural networks.
The difference between them is that the resnet neural network contains the identity and `1x1` convolution w/ stride `2`
projection shortcuts described in the paper.

The time taken for training was <time taken here> on:
  
    - Processor: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (12 CPUs), ~3.7GHz
    - Memory: 16384MB RAM
    - GPU: GTX 1080
    
The hardware didn't look fully saturated.
