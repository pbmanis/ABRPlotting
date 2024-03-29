import os
import subprocess


def getcomputer():
    if os.name == "nt":
        computer_name = os.environ["COMPUTERNAME"]
    else:
        computer_name = subprocess.getoutput("scutil --get ComputerName")
    print(computer_name)
    if computer_name == "Lytle":
        basedir = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data"
    elif computer_name == "Tamalpais":
        # basedir = '/Users/pbmanis/Desktop/data/ABR_DATA'
        basedir = "/Volumes/Backup2B/ABR_DATA"
    elif computer_name == "RIG5":
        basedir = "C:/Users/experimenters/Desktop/ABR_Data"
    else:
        raise ValueError("Need valid computer name to set base path to data")
    return basedir, computer_name


if __name__ == "__main__":
    b, c = getcomputer()
    print(b, c)
