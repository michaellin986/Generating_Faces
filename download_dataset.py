import os
import tarfile

if __name__ == '__main__':
    # requirements
    os.system('pip3 install gdown')
    
    # download dataset
    os.system('gdown https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk')
    
    # extract tar
    file = tarfile.open('UTKFace.tar.gz')
    file.extractall('./UTKFace')
    file.close()

    # remove tar
    os.remove('UTKFace.tar.gz')
    print('done.')
