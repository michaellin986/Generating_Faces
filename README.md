
Setup: install requirements in requirements.txt. Make sure the correct version of torch is installed

1) Downloading the dataset

Automatic option:
    Run the "download_dataset.py" script

Manual option: 
    Download UKT dataset from https://www.kaggle.com/datasets/jangedoo/utkface-new
    (download link: https://www.kaggle.com/datasets/jangedoo/utkface-new/download?datasetVersionNumber=1)
    make a directory inside the project root directory called "UTKFace"
    Unzip the downloaded file, and place a UTKFace directory containing jpg photos with no subdirectories (check unzipped folder subdirectories) 
        inside the UTKFace directory you just corrected. Final directory structure should be:
        ./UTKFace/UTKFace/***.jpg's

2) Running the program

run main.py
