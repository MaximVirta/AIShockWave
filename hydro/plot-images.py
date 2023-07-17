# script to plot images created by v2fromHDF.py
import numpy as np
import matplotlib.pyplot as plt
import sys

imgName = sys.argv[1]; 
compressed = np.load(imgName)

j = 0
for i in range(3):
    fig, axes = plt.subplots(1,3, figsize=(12,3))#(1,2, figsize=(12,3))
    for iax, ax in enumerate(axes):
        im = ax.imshow(compressed["images"][j][0], interpolation = 'none') #extent = [-0.8, 0.8, 0, 2*np.pi], 
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('eta')
        ax.set_xticks([0,7,15,23,31], np.round(np.linspace(-0.8, 0.8, 5), 2))
        ax.set_ylabel('phi')
        ax.set_yticks([0,7,15,23,31], np.round(np.linspace(-np.pi, np.pi, 5), 2))
        j +=1
		#ax.set(title='{} jet: $p_T=${:.0f} GeV'.format(['QCD','top'][iax], [jetpep0,jetpep][iax][idx][0][0]))
    plt.subplots_adjust(wspace=0.5)
    fig.suptitle("Particle images")
    axes[0].set_title("sample {}".format(j-2))
    axes[1].set_title("sample {}".format(j-1))
    axes[2].set_title("sample {}".format(j))
#plt.show()
    filename = "images-sample{}-to-{}.png".format(j-2, j)
    plt.savefig(filename)
    plt.close(fig)
