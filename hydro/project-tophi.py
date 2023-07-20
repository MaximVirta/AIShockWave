# script to plot projections of images created by v2fromHDF.py
import numpy as np
import matplotlib.pyplot as plt

def projectToPhi(imgName="particles_PbPb_50evt.hdf.npz"):#, etaMin = -0.8, etaMax = 0.8):
    event_n = 1
    #sample_n = 1
    j = (event_n-1)*10#+sample_n-1 # 10 images per event, so event_n=1 images are at j = 0...9, etc. 
    compressed = np.load(imgName)
    print(compressed['images'].shape)
    #projection = np.sum(compressed["images"][j][3][:,:], axis=0) #etaMin:etaMax
    
    for k in range(7): # take 7 events
        for i in range(10): # sum over all 10 samples
            sum = 0
            projection = np.sum(compressed["images"][j+k*10+i][3][:,:], axis=0) #etaMin:etaMax
            sum += projection
        plt.plot(np.linspace(-np.pi, np.pi, 32), sum, label = "{}".format(j+k+1))
    plt.title("Pb$-$Pb $\\sqrt{s_{\\mathrm{NN}}}$ = 5.02 TeV")
    # doesn't work -> 
    #annotation = """${T\\raisebox{-.5ex}{R}ENTo}+VISH(2+1)+UrQMD$""" 
    annotation = "$T_RENTo+VISH(2+1)+UrQMD$"
    plt.annotate(annotation, (0,10), xycoords='axes points')
    plt.xlabel('$\\phi$')
    plt.ylabel('$dN_{ch}/d\\phi$')
    plt.legend(title = 'Event #')
    #plt.show()
    filename = "projection-evts-1-to-{}.png".format(k+1)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
	projectToPhi()
