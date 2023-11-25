### Robust Few-Shot Classification via Bilevel Knowledge Distillation ([Korean Paper](https://github.com/Jeong-Bin/MS-Thesis-Paper/files/13464189/paper.pdf))

You can create **corrupt images** with ```./data/custom_create_FSL_C.py``` for Mini-ImageNer-C, CUB-200-C, FGVC-Aircraft-C or ```custom_create_cifar_C.py``` for CIFAR-FS-C. <br/>
I recommend adding the above codes to the directory in ```ImageNet-C/create_c``` of this [Github](https://github.com/hendrycks/robustness/tree/master). <br/>
And original datasets should be prepared.

<br/>

**[Download link of original datasets]**
- Mini-ImageNet : [Github](https://github.com/yaoyao-liu/mini-imagenet-tools) or [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet).
- CUB-200-2011 : [Github](https://github.com/pytorch/vision/issues/2992) or [Caltech](https://data.caltech.edu/records/65de6-vp158).
- CIFAR-FS : [Github](https://github.com/bertinetto/r2d2) or [Drive](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view).
- FGVC-Aircraft : [Site](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
