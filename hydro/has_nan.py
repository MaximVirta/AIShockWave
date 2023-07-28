import numpy as np
import os

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
    for filename in file_list:
        filepath = os.path.join(directory, filename)
        compressed = np.load(filepath, allow_pickle=False)
        flowdata = compressed['flow_data']
        if has_nan(flowdata):
            howmanyNaNs += 1
        print('files containing NaNs: ', howmanyNaNs, ' /', n_files)
    
if __name__ == '__main__':
    main()
