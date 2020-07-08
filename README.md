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

## To do(s)
I will come back to it to:
- check yaw rate values
- improve GAN clustering
- test with more features such as **OpenStreetMap** or **PROC VEHICLE DETECTION**

## Conclusion

### Notebook:
[Notebook link](https://nbviewer.jupyter.org/github/sqrx-mckl/driver_behaviour/blob/8da8cf5bcd19aaa6cb031f5cd160df9afa580deb/notebooks/4_clustering_gan.ipynb)

[Notebook Github link](https://github.com/sqrx-mckl/driver_behaviour/blob/master/notebooks/4_clustering_gan.ipynb)

### Plots:
* [GAN output link](https://nbviewer.jupyter.org/github/sqrx-mckl/driver_behaviour/blob/8da8cf5bcd19aaa6cb031f5cd160df9afa580deb/notebooks/4_clustering_gan.ipynb#Example-of-GAN-output)
![Gan output](.img/gan_output.png?raw=true "Gan output")
* [Line plots per cluster link](https://nbviewer.jupyter.org/github/sqrx-mckl/driver_behaviour/blob/8da8cf5bcd19aaa6cb031f5cd160df9afa580deb/notebooks/4_clustering_gan.ipynb#Line-plots)
![lineplot](.img/lineplot.png?raw=true "Lineplot")
* [Trajectories per cluster link](https://nbviewer.jupyter.org/github/sqrx-mckl/driver_behaviour/blob/8da8cf5bcd19aaa6cb031f5cd160df9afa580deb/notebooks/4_clustering_gan.ipynb#Ghost-plot-with-trajectories)
![trajectories](.img/trajectories.png?raw=true "Trajectories")
* [T-SNE link](https://nbviewer.jupyter.org/github/sqrx-mckl/driver_behaviour/blob/8da8cf5bcd19aaa6cb031f5cd160df9afa580deb/notebooks/4_clustering_gan.ipynb#T-SNE)
![tsne](.img/tsne.png?raw=true "T-SNE")