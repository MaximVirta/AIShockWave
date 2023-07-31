import numpy as np
import os
import sys

def removeNans(flowdata, images):
    """
    removes flowdata with NaNs and corresponding images
    args: arrays to remove from
    returns: same arrays with columns containing NaNs removed (if any)
    """
    # Find indices of rows in flowdata that do not contain NaN
    valid_indices = ~np.isnan(flowdata).any(axis=1)

    # Return copies of flowdata and images with rows at valid_indices
    return flowdata[valid_indices].copy(), images[valid_indices].copy()


def main(batchnum):
    directory = '/scratch/project_2003154/MachineLearning/FirstImages5TeV/'
    directory += batchnum+'/'
    print(directory)
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath): continue
        compressed = np.load(filepath, allow_pickle=False)
        flowdata, images = compressed['flow_data'], compressed['images']
        #print('data before', flowdata.shape)
        #print('images before', images.shape)
        flowdata2, images2 = removeNans(flowdata, images)
        #print('data after', flowdata2.shape)
        #print('images after', images2.shape)
        np.savez_compressed(filepath, images = images2, flow_data = flowdata2)

if __name__ == '__main__':
    batchnum = sys.argv[1]
    main(batchnum)
