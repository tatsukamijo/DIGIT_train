# DIGIT_train

This is a repository including a sample code for creating a custom touch detection model 
for a [DIGIT](https://digit.ml/#:~:text=What%20is%20DIGIT%3F,by%20playing%20the%20video%20below.) sensor.  
See also: [PyTouch](https://github.com/facebookresearch/PyTouch)

# Requirements
It requires `pytouch`, `digit-interface`, `torch`, `torchvision` to be installed in your python.  
Install with `pip install` if not.

# Usage
## 1. Make a dataset
1. Change the directory name `Dxxxx` to your DIGIT id in `DIGIT_train/Dxxxx`   
2. Change `DIGIT_ID` and the path to yours in `make_data.py`
3. Run `make_data.py` to make your dataset in `./Dxxxx/touch` and `./Dxxxx/notouch`  
Each frame will be saved to the specified directory while running.  
Make sure to create various touching situations.
To achieve this, I changed the touching situation for each set of 200 or 300 frames.
## 2. Train a model
Change `DIGIT_ID` in `model.py` and run it. 
`.pth` file will be saved under the `./weight` directory.
## 3. Test the model
Change `DIGIT_ID` in `realtime_touchdetect.py` and run it. 
