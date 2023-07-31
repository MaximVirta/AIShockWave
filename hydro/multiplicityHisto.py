# script to plot histogram of multiplicities of samples (each sample interpreted as its own event, so same hydro sim samples can be - in theory - put into different centrality bins)
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def getMultiplicities(directory):
    multiplicities = []
    for i in range(100):
        file_list = os.listdir(directory + str(i)+'/')
        for filename in file_list:
            filepath = os.path.join(directory, filename)
            if not os.path.isfile(filepath): continue
            compressed = np.load(filepath, allow_pickle=False)
            M = compressed['flow_data'][2]
            multiplicities.append(M)
    return np.array(multiplicities)

    np.append(multiplicities, M)
    return multiplicities 

def plotHisto(values):#, etaMin = -0.8, etaMax = 0.8):
    plt.plot(np.histogram(values, bins=1000))
    plt.title("Pb$-$Pb $\\sqrt{s_{\\mathrm{NN}}}$ = 5.02 TeV")
    # doesn't work -> 
    #annotation = """${T\\raisebox{-.5ex}{R}ENTo}+VISH(2+1)+UrQMD$""" 
    annotation = "$T_RENTo+VISH(2+1)+UrQMD$"
    plt.annotate(annotation, (0,10), xycoords='axes points')
    plt.xlabel('multiplicity')
    plt.ylabel('N of events')
    #plt.legend(title = 'Event #')
    #plt.show()
    filename = "multiplictyHistogram.png"
    plt.savefig(filename)
    #plt.close()

def main():
    directory = '/scratch/project_2003154/MachineLearning/FirstImages5TeV/'
    multiplicities = getMultiplicities(directory)
    plotHisto(multiplicities)

if __name__ == '__main__':
    main()