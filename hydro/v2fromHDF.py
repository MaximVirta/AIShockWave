# This script creates particle images and calculates v2 from hdf files using 2-particle correlations from Q vectors
# Based on https://arxiv.org/pdf/1010.0233.pdf
# Images are created for each sample
# Flow coefficients for each event are calculated using all samples
# For simplicity assume there is uniform acceptance in the detector
# TODO: add comments to functions

import h5py
import numpy as np
import matplotlib.pyplot as plt


parts_dtype = [
	('sample', int),
	('ID', int),
	('charge', int),
	('pT', float),
	('ET', float),
	('mT', float),
	('phi', float),
	('y', float),
	('eta', float)
]

# skipped_images=np.array([])
# skipped_v2s= np.array([])
# skipped_Ms =np.array([])

def QVector(phis, n):
    return sum(np.exp(n*1j*phis))
# by eq (16)


def SingleEvtAvgTwoParticleCorr(Qns, Ms):
    return (np.abs(Qns)*np.abs(Qns) - Ms)#/(Ms*Ms-Ms)


def make_image_sample(sample, energy=5.02):
    histoE, xedgesE, yedgesE = np.histogram2d(sample['eta'], sample['phi'], weights=np.log(energy)*np.ones(len(sample)), bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    histomT, xedgesmT, yedgesmT = np.histogram2d(sample['eta'], sample['phi'], weights=sample['mT'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    histopT, xedgespT, yedgespT = np.histogram2d(sample['eta'], sample['phi'], weights=sample['pT'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    # unweighted
    histo, xedges, yedges = np.histogram2d(sample['eta'], sample['phi'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    return np.array([np.array([histoE, histomT, histopT, histo])])


def select_particles_from_sample(particles, N_samples, Q2s,Q3s, Ms, images):
    
    for n in range (N_samples):  
      sample=[]
      for particle in particles:
            if particle['sample']!=n+1 or particle['charge']==0 or np.abs(particle['eta']) >= 0.8: continue; 
            sample.append(particle)
      sample = np.array(sample) 
      Ms=np.append(Ms, len(sample)) 
      Q2s = np.append(Q2s, QVector(sample[:]['phi'], 2))
      Q3s = np.append(Q3s, QVector(sample[:]['phi'], 3))
      sample_images=make_image_sample(sample)
      images=np.append(images, sample_images, axis=0)


    return [ Ms, Q2s,  Q3s, images]


 
def calculate_vn_per_event( Ms,Q2s, Q3s,v2s,v3s, N_samples=10):
            skipped = False
            weights = Ms*Ms-Ms
            sum_of_weights = sum(weights)
            single_event_avgs_2 = SingleEvtAvgTwoParticleCorr(Q2s, Ms)
            single_event_avgs_3 = SingleEvtAvgTwoParticleCorr(Q3s, Ms)
           
            if sum(single_event_avgs_3) > 0: # only positive flow coefficients usable
                v2s = np.append(v2s, np.sqrt(sum(single_event_avgs_2)/sum_of_weights)*np.ones(N_samples)) # N_samples images per event
                v3s = np.append(v3s, np.sqrt(sum(single_event_avgs_3)/sum_of_weights)*np.ones(N_samples))
               # Ms_event = np.append(Ms_event, np.sum(Ms)*np.ones(N_samples)) # charged particle multiplicity per event
            else:
                skipped = True
               
            return [v2s,v3s, skipped]

    
def  main():
   #define initial parameters
    energy=5.02
    v2s = np.array([])
    v3s = np.array([])

    fn="particles_PbPb_50evt.hdf"
  
    images = np.empty((0,4,32,32))


    N_samples=10   

    with h5py.File(fn,"r") as f:

       # Ms_event = np.array([])
        for evt in f.values():
           
            Ms=np.array([])
            Q2s = np.array([]) # Q-vectors per sample
            Q3s = np.array([])
            particles = np.sort(np.array(evt[:], dtype=parts_dtype), order=['sample','charge'])
            Ms, Q2s,  Q3s, images=select_particles_from_sample(particles, N_samples,Q2s,Q3s, Ms,images)


            v2s,v3s, skipped=calculate_vn_per_event(Ms,Q2s,Q3s,v2s,v3s)
            if skipped==True:              
                images = images[:-N_samples] # throw away images of last event

    Ms_image=np.sum(images,axis=(1,2,3))
    # print(v2s.shape)
    # print(v3s.shape)
    # print(Ms_image.shape) #should be 180 for particles_PbPb_50evt.hdf

    flowdata = np.stack((v2s, v3s, Ms_image), axis=-1)
    np.savez_compressed('{}.npz'.format(fn),images=images,flow_data = flowdata)

 

if __name__=="__main__":
     main()



