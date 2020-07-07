# Driver Behaviour Study/Analysis with DL

This repository is dedicated to data analysis on time-series provided by CAN data from a vehicle on the road.

## Data
Data comes from: https://github.com/Eromera/uah_driveset_reader  
Some information about the column names is avaialble: 
* https://github.com/Eromera/uah_driveset_reader/blob/master/driveset_reader.py
* https://www.researchgate.net/publication/307628008_Need_Data_for_Driver_Behaviour_Analysis_Presenting_the_Public_UAH-DriveSet

## Analysis
The repo is organized as follows:
* __lib__:
    a library with all useful functions for this study but could be re-used for more
* __notebooks__:
    the notebooks used to investigate the data and experiment. They are organized by their prefix "X_". If there is no prefix it is either old or something out of ordinary

## Inspiration and help
This is heavily inspired (with copy/paste) from the following article:
[Data Exploration with Adversarial Autoencoders](https://towardsdatascience.com/data-exploration-with-adversarial-autoencoders-311a4e1f271b)