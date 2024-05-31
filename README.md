# LocationPredictionContextQ

## Set up
a) Create a working directory to hold the contents of this ```LocationPredictionContextQ``` repository.

b) Add executorch to the working directory following the instructions in
https://pytorch.org/executorch/stable/getting-started-setup.html

c) In the executorch directory, run the following commands to install packages necessary in order to train the model
- pip install wheel
- python3 setup.py bdist_wheel
- pip install -U pip setuptools ruamel-yaml pyyaml PyYaml
- pip install matplotlib scikit-learn scipy pydantic
- pip install geopandas pygeos transformers gensim
- pip install easydict
- pip install trackintel

## Running the Training Loop and Executorch Model Export
1. If there is a folder called ```temp``` in <working directory>/data delete it.
2. In the working directory, execute the ```main.py config/geolife/transformer.yml``` script. 

## Notes - Please Read
i) I am new to Python. Please bear in this mind as you will probably find some (many) of the things I did amateurish/beginner.

ii) ```train-minimum.py``` is (very) messy. I have been using it as a sandbox. There are many lines which are commented out. They are either instructions, notes, or code which I tried and disabled because it did not work, or code I used to test some ideas which I abandoned. I am sorry as this will make navigating/understanding train-minimum.py (more) difficult to navigate.

iii) Although they should work, I did not test the instructions in this ```README.md``` file. However I did run ```train-minimum.py``` on my system in a working direcotry set up as described above producing the execution failure as described in
https://github.com/pytorch/executorch/issues/120219
  
iv) The executorch related code starts here
https://github.com/adonnini/LocationPredictionContextQ/blob/c43859507144ed5596323af2831d01c8c9e7ab6c/utils/train.py#L254
and ends here
https://github.com/adonnini/LocationPredictionContextQ/blob/c43859507144ed5596323af2831d01c8c9e7ab6c/utils/train.py#L518
