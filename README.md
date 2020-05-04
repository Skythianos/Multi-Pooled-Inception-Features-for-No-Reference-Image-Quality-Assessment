# Multi-Pooled-Inception-Features-for-No-Reference-Image-Quality-Assessment
Image quality assessment (IQA) is an important element of a broad spectrum of applications ranging from automatic video streaming to display technology. Furthermore, the measurement of image quality requires a balanced investigation of image content and features. Our proposed approach extracts visual features by attaching global average pooling (GAP) layers to multiple Inception modules of on an ImageNet database pretrained convolutional neural network (CNN). In contrast to previous methods, we do not take patches from the input image. Instead, the input image is treated as a whole and is run through a pretrained CNN body to extract resolution-independent, multi-level deep features. As a consequence, our method can be easily generalized to any input image size and pretrained CNNs. Thus, we present a detailed parameter study with respect to the CNN base architectures and the effectiveness of different deep features. We demonstrate that our best proposal—called MultiGAP-NRIQA—is able to outperform the state-of-the-art on three benchmark IQA databases. Furthermore, these results were also confirmed in a cross database test using the LIVE In the Wild Image Quality Challenge database.

In this repository, the code belonging to Multi-pooled Inception Features for No-reference Image Quality Assessment can be found. If you use this code, please cite the following paper:

@article{varga2020multi,
  title={Multi-Pooled Inception Features for No-Reference Image Quality Assessment},
  author={Varga, Domonkos},
  journal={Applied Sciences},
  volume={10},
  number={6},
  pages={2186},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}

