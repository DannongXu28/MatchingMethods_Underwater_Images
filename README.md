# CTNet for Underwater Images

This is the deep learning algorithm developed for image matching in underwater environments.



## Architecture of CTNet
![CTNet](https://github.com/DannongXu28/MatchingMethods_Underwater_Images/blob/main/Architecture_CTNet/CTNet.png?raw=true)

## Environment

 - **Python 3.8**
   
 - **PyTorch 1.13.0**
 
 - **TensorFlow 2.11.0**

## Dataset

The underwater images are from the dataset [Tasmania Coral Point Count](https://marine.acfr.usyd.edu.au/datasets/) published by ACFR. 

After downloading the Tasmania Coral Point Count dataset, reconstruct it as structure below:

```
Dataset
	- Group 1
		- Scene 1 Left
		- Scene 1 Right
	- Group 2
		...
	...
...  
```

## How to train

1. Edit Hyper Parameter in ```option.py``` 

2. Run ```train.py```

3.  All data will be saved under ```logs```

## How to test

1. Load weights in ```option.py```

2. Run ```predict.py```

3. Insert the first picture

4. Insert the second picture

5. Obtain the similarity score

## Citation
If you want to use this open source code for CTNet, please cite this [github link](https://github.com/DannongXu28/MatchingMethods_Underwater_Images/tree/main).

##  Acknowledgement

**Thanks for these open source publishers !!!**

The code for CTNet is based on:

[Siamese-PyTorch](https://github.com/bubbliiiing/Siamese-pytorch)

[ACT](https://github.com/jinsuyoo/act)

The code for Siamese Network is based on:

[Siamese-PyTorch](https://github.com/bubbliiiing/Siamese-pytorch)

The code for MAML is based on:

[MAML-PyTorch](https://github.com/dragen1860/MAML-Pytorch)

The code for Reptile is based on:

[Reptile-PyTorch](https://github.com/dragen1860/Reptile-Pytorch)
