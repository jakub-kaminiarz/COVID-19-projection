Jakub Kaminiarz
Covid-19 Project


CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Configuration
 * How to run
 * Tests




INTRODUCTION
------------
In this project different methods of disease and fatality prediction will be considered.




REQUIREMENTS
------------
Program to run correctly requires Python 3.7 or higher. 
To run the program correctly, following libraries should be installed:
- pandas
- geopandas
- matplotlib
- seaborn
- matplotlib
- numpy
- datetime
- sklearn
- warnings
- keras
- statistics

If you are using Jupyter Notebook, all you have to do is import all libraries. 
Installation of geopandas library requires a command:
pip install geopandas




CONFIGURATION
-------------
Zip folder contains two type of executable files (.py and .ipynb). 

---- .py ----
File is ready to run from terminal. Python 3.7 or higher must be installed.
To run program .csv file must be in the same folder as the executable file. 

---- .ipynb ----
This file is prepare for ease run using jupyter notebook.




HOW TO RUN
----------
---- .py ----
Move to a folder containing all the necessary files.
To run program from terminal simply use this command:
python covid19_kaminiarz_program.py

---- .ipynb ----
To run program properly open Jupyter Notebook and simply run all the cell. Remember that .csv file must be in the same directory as executable file.




TESTS
-----
Zip. Folder contains file RMSE_test.py and RMSE_results.txt. The tests program has been prepared in order to find the most suitable parameters of the custom MLP model. 
The results contained in the .txt file include the average RMSE error from 20 iterations of the learning process for each case, number of epochs in range [10, 50, 100, 200, 500, 1000], batch size in the range [1, 5, 10, 15, 30, 50], for the original dataset and smaller set (from 2020-04-01).
The whole testing process took about 100 hours.