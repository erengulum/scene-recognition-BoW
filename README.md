
# Hough-transform for Plate Recognition

The scene-recognition.py file has been prepared for scene recognition. The code is written in a single file and there is no other code file on which it depends.
Most of the code consists of functions. The functions are kept as simple as possible and tried to be made in accordance with the singleton principle.

## Usage

The code consists of a single .py file. 
To run the code, open it in a Python interpreter and execute it. SceneDataset has to be in same folder with scenerecognition.py file.
Execution time is 603 seconds (depends on dataset and enviroment). After the execution is finished, train-validation-test subfolders will be created in the same folder.
In addition, 4 different Confusion Matrix will be created in the same folder in png format.

Warning: If you are going to use dataset containing different categories than current one, don't forget to update the "categories" variable in the code.

## License
[MIT](https://choosealicense.com/licenses/mit/)
