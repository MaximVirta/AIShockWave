import numpy as np
import os
import sys

def has_nan(array):
    '''
    checks if given array has a NaN inside
    returns: True if given array has at least one NaN, False otherwise
    '''
    return np.isnan(array).any()


def main():
    directory = '/scratch/project_2003154/MachineLearning/FirstImages5TeV/test_set'
    howmanyNaNs = 0
    file_list = os.listdir(directory)
    n_files = len(file_list)
    i = 0
    for filename in file_list:
        i += 1
        filepath = os.path.join(directory, filename)
        compressed = np.load(filepath, allow_pickle=False)
        flowdata = compressed['flow_data']
        if has_nan(flowdata):
            howmanyNaNs += 1
        if(i%100 == 0):
            sys.stdout.write("\r{0}".format((float(i)/n)*100))
            sys.stdout.flush()
    print('files containing NaNs: ', howmanyNaNs, ' /', n_files)
    
if __name__ == '__main__':
    main()
