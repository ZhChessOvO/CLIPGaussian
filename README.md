# CLIPGaussian
Code in “Enhancing 3D Gaussian Splatting for Low-Quality Images: Semantically Guided Training and Unsupervised Quality Assessment”

### How to setup (recommended in Conda)

Our experiments are done in Windows11, Visual Studio 2019 for Windows (C++ compiler), CUDA SDK 11.8, just for reference.

1. Follow the steps in https://github.com/graphdeco-inria/gaussian-splatting to build the environment

2. run

   ```
   pip install clip
   ```

### How to train

run

```
python train2.py -s <PATH-to-dataset>
```

It is worth stating that these codes are the versions we used to conduct our experiments. To avoid problems such as long training times, we recommend that you change the number on line 142 in train2.py to the number of images in your training set or half the number of images. 

Of course, it will also work fine without any modifications. 

If the paper is fortunate enough to be accepted, we will update a version of the code that allows you to set these parameters on the command line.

### How to view

run

```
cd .\viewers\bin
.\SIBR_gaussianViewer_app.exe -m <PATH-to-output>
```

### How to get the pretrained models and the low-quality datasets (which are mentioned in our paper)

pretrained models: download in Releases

low-quality datasets: use *blur2.py*

### Others

**If you have any concerns or questions, please contact us at chesszh@foxmail.com**

If the paper is fortunate enough to be accepted, we will optimize the code into a more mature and stable version.
