# What does this tool do?
1. Detects and highlights a face using your device camera
2. Detects if the face has a mask on or not
3. Displays feedback as shown below

![Demo](https://github.com/helich0pper/DetectFaceMask/blob/master/demo.png) <br>

Original can be found [here](https://github.com/chandrikadeb7/Face-Mask-Detection)

# How does it work?
All **pre-trained** deep neural network models used are in the ```models``` folder. <br>
The datasets used to train the modules can be found [here](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG). <br>
Dataset details:
* 2165 images **with** a mask
* 1930 images **without** a mask

# How to use?
1. ```git clone https://github.com/helich0pper/DetectFaceMask.git```
2. ```cd DetectFaceMask```
3. ```pip install -r requirements.txt```
4. ```python detect_mask.py```
5. Always wear a mask in public. Stay safe!

