

# inCCsight
inCCsight is a web-based software for processing, exploring e visualizing data for Corpus Callosum analysis using Diffusion Tensor Images (DTI) implemented in Python/Dash/Plotly. 



***

## How to install

We suggest you to create a separate virtual environment running Python 3 for this app, and install all of the required dependencies there. 

### In Windows:
You will need *Python3* and *git* installed in your machine in order to install this app, please refer to https://docs.python.org/3/using/windows.html and https://git-scm.com/book/en/v2/Getting-Started-Installing-Git if you need help installing these tools.

Run in Command Prompt. Clone the git repository:
```
git clone https://github.com/thaiscaldeira/ccinsight/
cd ccinsight
```
Create a new virtual environment:
```
python3 -m virtualenv venv
venv\Scripts\activate
```
Update pip and install app requirements
```
py -m pip install -U pip
pip install -r requirements.txt
```


### In UNIX-based systems:
You will need *Python3* and *git* installed in your machine in order to install this app, please refer to https://docs.python.org/3/using/unix.html and https://git-scm.com/book/en/v2/Getting-Started-Installing-Git if you need help installing these tools.

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
* `-m`, or `--maskname` : string contained in the file name of imported masks. See the Importing Masks section for more information;
* `-d`, or `--extra-data` : path to sheet file (.xls, .xlsm or .csv) with additional information to be imported and visualized. See the Importing External Data section for more information;

After calling the initial command, data will be processed and your default browser will open, showing the interactive dashboard for data exploration and visualization. While the Terminal or Command Prompt where the program is running is kept open, you can access the dashboard on `http://127.0.0.1:5050/`.

