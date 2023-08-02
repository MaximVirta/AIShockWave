# script to plot images created by v2fromHDF.py
import numpy as np
import matplotlib.pyplot as plt
import sys



def plot_images(imgName="particles_PbPb_50evt.hdf.npz"):
    event_n = 2
    sample_n = 1
    j = (event_n-1)*10+sample_n-1 # 10 images per event, so event_n=1 images are at j = 0...9, etc. 
    compressed = np.load(imgName)
    for i in range(3):
        fig, axes = plt.subplots(1,3, figsize=(12,3))#(1,2, figsize=(12,3))
        for iax, ax in enumerate(axes):
            #print(compressed['images'].shape)
            im = ax.imshow(compressed["images"][j][0], interpolation = 'none') #extent = [-0.8, 0.8, 0, 2*np.pi], 
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('$\\eta$')
            ax.set_xticks([0,7,15,23,31], np.round(np.linspace(-0.8, 0.8, 5), 2))
            ax.set_ylabel('$\\phi$')
            ax.set_yticks([0,7,15,23,31], np.round(np.linspace(-np.pi, np.pi, 5), 2))
            j +=1
    		#ax.set(title='{} jet: $p_T=${:.0f} GeV'.format(['QCD','top'][iax], [jetpep0,jetpep][iax][idx][0][0]))
        plt.subplots_adjust(wspace=0.5)
        fig.suptitle("Event {}".format(event_n))
        axes[0].set_title("sample {}".format(sample_n))
        axes[1].set_title("sample {}".format(sample_n+1))
        axes[2].set_title("sample {}".format(sample_n+2))
    #plt.show()
        filename = "images-sample{}-to-{}.png".format(sample_n, sample_n+2)
        plt.savefig(filename)
        plt.close(fig)
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        image_path=sys.argv[1]
    plot_images(image_path)