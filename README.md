

# inCCsight
inCCsight is a web-based software for processing, exploring e visualizing data for Corpus Callosum analysis using Diffusion Tensor Images (DTI) implemented in Python/Dash/Plotly. 

The software is open source,
portable and interactive for analysis of the corpus callosum in DTI individually or in groups,
implementing different techniques available for segmentation and installment and proposing
relevant metrics for comparing and evaluating the quality of these procedures in
a web application.

![Software screenshot](https://github.com/thaiscaldeira/ccinsight/blob/master/assets/inccsight_screenshot.png)

***

## Considerations
This software was developed through the [MICLab](https://miclab.fee.unicamp.br/), check out our [Github](https://github.com/MICLab-Unicamp)!

**Article:** The article that explains the development and operation of the tool was published by [Computer & Graphics](https://www.journals.elsevier.com/computers-and-graphics). 
You can check out this article [here](https://www.sciencedirect.com/science/article/abs/pii/S0097849321001436).
In case of using the software, please cite this article: 

*Caldeira, Thais, et al. "inCCsight: A software for exploration and visualization of DT-MRI data of the Corpus Callosum." Computers & Graphics 99 (2021): 259-271.*

**Data**: In case you want to test the tool and do not have the data (DTI), check the [OASIS](https://www.oasis-brains.org/#access). This is a data center with medical images available for studies and collaboration of science community. 

If you use data from Oasis, check out the Oasis3 notebook in this repository, it performs a pre-processing of data collected from the data center.

For an overview of the tool, we have a video showing the process and use: [InCCsight](https://www.youtube.com/watch?v=9Y8s8H3X2ow&list=PLCZ64jtDHDO0fBxdyRM5jtukD3U_ZxME_&index=3)

## How to install

### Building from source

**Docker**: The main installation method is via Docker, both on Windows and for Linux.

We suggest you to create a separate virtual environment running Python 3 for this app, and install all of the required dependencies there. The installation steps below include the creation of the virtual environment.

You will need `Python3`, `pip`, `git` and `virtualenv` installed in your machine in order to install this app, please refer to https://docs.python.org/3/using/unix.html and https://git-scm.com/book/en/v2/Getting-Started-Installing-Git if you need help installing these tools. If you already have `Python3` and `pip` installed, you can install `virtualenv` by typing:
```
pip install virtualenv
```

Run in Terminal. Clone the git repository:
```
git clone https://github.com/thaiscaldeira/ccinsight/
cd ccinsight
```
Create a new virtual environment:
```
python3 -m virtualenv venv
source venv/bin/activate
```
Update pip and install app requirements
```
python -m pip install -U pip
pip install -r requirements.txt
```
Install siamxt (https://github.com/rmsouza01/siamxt)

**Warning for Windows users:** We failed to install one of the required libraries (siamxt) on Windows, which chould be built from source. If you're a Windows user we suggest you use the Docker image provided, as explained in the next section.

```
cd siamxt
python setup.py install
```

### Using a Docker image

Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. You can find more information, as well as download links and installation steps here: https://www.docker.com/

To use the docker image you'll only have to run it normally with the commands explained in the next sections. The first time you run it, it will automatically download the image. 

If you want to download the image without running it do:
```
docker pull thaiscaldeira/inccsight
```
Using docker will require, however, that you use some docker flags. inCCsight runs on localhost port 8000 by default, so if you wish to change its port while using the docker image you can do it by mapping the port like in the example below where it will run on https://localhost:8888:
```
docker run -p 8888:8000 thaiscaldeira/inccsight [INCCSIGHT FLAGS]
```
Alongside, to allow the container to read and save information from your disk we have to map the folders we wanna work with. For example, if I have folders organised as showed below we can map the volume `ALL_DATA` to a folder /f/ by using the flag `-v`followed by the mapping configuration `./ALL_DATA/:/f/` and use it with with the `--parent` flag from inCCsight. We will include more examples in the following 'How to use inCCsight' section of how to use it with docker.
```
|- ALL_DATA
   |- HEALTH_CONTROLS
      |- SUBJECT_000001
      |- SUBJECT_000002
      |- SUBJECT_000003
      |- ...
   |- CONDITION_X
      |- SUBJECT_000014
      |- SUBJECT_000015
      |- SUBJECT_000016
      |- ...
   |- CONDITION_Y
      |- SUBJECT_000029
      |- SUBJECT_000030
      |- SUBJECT_000031
      |- ...
```
```
docker run -v ./ALL_DATA:/f/ thaiscaldeira/inccsight --parent /f/HEALTH_CONTROLS/ /f/CONDITION_X/ /f/CONDITION_Y/
```

***

## How to use inCCsight

inCCsight processes DTI data, segmenting the Corpus Callosum and parcellating it using most proeminent segmentation an parcellation techniques available and reproducible in literature. More information about the techniques available please refer to the Segmentation and Parcellation documents. 

### Input data 

We input data by importing DTI eigenvectors and eigenvalues in the FSL format as outputted by [FSL DTIFit](https://users.fmrib.ox.ac.uk/~behrens/fdt_docs/fdt_dtifit.html), where you can inform the used basename by using the -b flag (default is dti):

* `basename_V1.nii.gz` : 1st eigenvector
* `basename_V2.nii.gz` : 2nd eigenvector
* `basename_V3.nii.gz` : 3rd eigenvector
* `basename_L1.nii.gz` : 1st eigenvalue
* `basename_L2.nii.gz` : 2nd eigenvalue
* `basename_L3.nii.gz` : 3rd eigenvalue

For each subject there must be a folder with the indicated files. The name of each folder will be used in the program as reference to the subject analysed. 

You can indicate the folders one by one (subject by subject) or the parent(s) folders that contain folders for each subject in a group. 

For instance, using the flag `-f` you can indicate each subject folder, subch as `SUBJ_00001` and `SUBJ_00002` in the example below:

```
|- SUBJECT_000001
   |- dti_V1.nii.gz
   |- dti_V2.nii.gz
   |- dti_V3.nii.gz
   |- dti_L1.nii.gz
   |- dti_L2.nii.gz
   |- dti_L3.nii.gz
|- SUBJECT_000002
   |- dti_V1.nii.gz
   |- dti_V2.nii.gz
   |- dti_V3.nii.gz
   |- dti_L1.nii.gz
   |- dti_L2.nii.gz
   |- dti_L3.nii.gz
```

When working with several subjects and/or different categories of individual (such as *Health Control* x *Condition* or *Male* x *Female*) it is more practical to indicate the parent folders of each group, such as `HEALTH_CONTROLS`, `CONDITION_X` and `CONDITION_Y` in the example below, where all `SUBJECT` folders contain the eigenvectors and eigenvalue files discussed previously:
```
|- HEALTH_CONTROLS
   |- SUBJECT_000001
   |- SUBJECT_000002
   |- SUBJECT_000003
   |- ...
|- CONDITION_X
   |- SUBJECT_000014
   |- SUBJECT_000015
   |- SUBJECT_000016
   |- ...
|- CONDITION_Y
   |- SUBJECT_000029
   |- SUBJECT_000030
   |- SUBJECT_000031
   |- ...
```

### Run commands

To use inCCsight we simply have to call the path to `app.py` on the Terminal/Command Prompt, indicating the subject(s) folder(s) paths (using the flag `--folders` or `-f`) or parent folder(s) paths (using the flag `--parents` or `-p`) to be analysed, such as shown below.
```
python app.py -f ./SUBJECT_000001 ./SUBJECT_000002 ./SUBJECT_000034 ...
```
```
python app.py -p ./HEALTH_CONTROLS ./CONDITION_X ./CONDITION_Y ...
```
We can also indicate auxiliary flags:
* `-b`, or `--basename` : string indicating the basename used in the eigenvectors/eigenvalues files (default is `'dti'`). See the Input Data section for more information;
* `-s`, or `--segm` : segmentation methods to be performed on the data (default is all available: _ROQS_ and _Watershed_);
* `--staple` : if used will create a consensus using the STAPLE method between the segmentations available (including imported masks);
* `-d`, or `--extra-data` : path to sheet file (.xls, .xlsm or .csv) with additional information to be imported and visualized. See the Importing External Data section for more information;
* `-m`, or `--maskname` : string contained in the file name of imported masks. See the Importing Masks section for more information;

After calling the initial command, data will be processed and your default browser will open, showing the interactive dashboard for data exploration and visualization. While the Terminal or Command Prompt where the program is running is kept open, you can access the dashboard on `http://127.0.0.1:5050/`.

### Importing external data

We often would like to cross categorical data (such as sex or race), or numerical data (such as age), with information extracted through DTI processing. Using **inCCsight** we can import such types of data and visualize relations between them and the processed data by importing a sheet file, such as .xlsm, .xls or .csv, that must have a column called 'Subjects' with the names of the imported Subject folders.

Columns with categorical data will be listed as a View Category that, when selected, will make all graphs compare the groups in this category. For example, if you select the category 'Sex' and your columns divided subjects between 'M' and 'F', all graphs will compare these two groups.

To import external data you can use the flag `-d` or `--extra-data`, as shown:
```
python app.py -p ./HEALTH_CONTROLS ./CONDITION_X ./CONDITION_Y -d ./subjects_informations.xls
```

### Next Steps and Updates

- [ ] Bug fixes.
- [ ] Data (DTI) available for testing.
