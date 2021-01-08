

# inCCsight
inCCsight is a web-based software for processing, exploring e visualizing data for Corpus Callosum analysis using Diffusion Tensor Images (DTI) implemented in Python/Dash/Plotly. 

![Software screenshot](https://github.com/thaiscaldeira/ccinsight/blob/master/assets/inccsight_screenshot.png)

***

## How to build the docker repo

To build the docker repository you must use the Dockerfile available in this folder. To do so you must place it in the main inCCsight folder and run the command:

```
sudo docker build -t thaiscaldera/inccsight
```

## How to run the docker file

To run the program using docker you must run:
```
sudo docker run -p 8000:_port_ -v _sharedFolderPath_:/p/ thaiscaldeira/inccsight -p /p/
```



