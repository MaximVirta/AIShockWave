# author: pyjorunk
# Arguments: CM energy, 
# This script creates particle images and calculates v2 from hdf files using 2-particle correlations from Q vectors
# Calculating with Q-vectors avoids having to consider each particle pair 
# This technique is insensitive to non-flow effects and interference between different harmonics
# Based on https://arxiv.org/pdf/1010.0233.pdf

# Images are created for each sample
# Flow coefficients for each event are calculated using all samples
# For simplicity assume there is uniform acceptance in the detector


import sys
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

# by eq (4)
def QVector(phis, n):
    return sum(np.exp(n*1j*phis))
# by eq (16)
def SingleEvtAvgTwoParticleCorr(Qns, Ms):
    return (np.abs(Qns)*np.abs(Qns) - Ms)#/(Ms*Ms-Ms)

# TODO edit this to make the two other layers
def make_image_sample(sample, energy=5.02):
    histoE, xedgesE, yedgesE = np.histogram2d(sample['eta'], sample['phi'], weights=np.log(energy)*np.ones(len(sample)), bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    histomT, xedgesmT, yedgesmT = np.histogram2d(sample['eta'], sample['phi'], weights=sample['mT'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    histopT, xedgespT, yedgespT = np.histogram2d(sample['eta'], sample['phi'], weights=sample['pT'], bins=(32,32), 
                                           range=[[-0.8,0.8],[-np.pi,np.pi]])
    return np.array([np.array([histoE, histomT, histopT])])



#print(sys.argv[2:]) # when running, give as arguments the files with data
#TODO: give as first arg the beam energy, second arg bool of hdf or MC
#for fn in sys.argv[1:]:
def createImages(fn="particles_PbPb_50evt.hdf"):

    energy = 5.02 #sys.argv[1]
    v2s = np.array([])
    v3s = np.array([])
    Ms_event = np.array([])
    # images = np.empty(shape=(0,3,32,32))

    # for statistics
    skipped_v2s = np.array([])
    skipped_Ms = np.array([])
    skipped_images = np.array([])
    sample_diffs = np.array([])
    image_diffs = np.array([])
    sample_sizes = np.array([])
    nch_inimage_check = 0
    total_samples = 0
    total_particles = 0

    images = np.empty((0,3,32,32))
    with h5py.File(fn,"r") as f:
        event_n = 1

        ievt = 0
        for evt in f.values(): # loop over events in file
            Ms = np.array([]) # mulitplicities per event
            Q2s = np.array([]) # Q-vectors per event
            Q3s = np.array([])
            particles = np.sort(np.array(evt[:], dtype=parts_dtype), order=['sample','charge']) # list of particles in event
            sample_start = 0
            sample_n = particles[0]['sample']
            current_particle = 0
            n_particles = 0
            ch_indices = np.array([])
            N_samples = 10 #particles.max(axis=1);
            for s in range(N_samples):
                sample = []
                for particle in particles:
                    if particle['sample']!=s+1 or particle['charge']==0 or np.abs(particle['eta']) >= 0.8: continue; 
                    sample.append(particle);

                sample = np.array(sample)
                Ms = np.append(Ms, len(sample)) # charged multiplicity of sample
                Q2s = np.append(Q2s, QVector(sample[:]['phi'], 2))
                Q3s = np.append(Q3s, QVector(sample[:]['phi'], 3))
                sample_images = make_image_sample(sample, energy=energy)
                images = np.append(images, sample_images, axis=0)
                image_diffs = np.append(image_diffs, len(sample)-np.sum(images[-1, 0], axis=(0,1)))
                #print(s, images.shape, sample.size)

            # calculate vn per event using all samples, weights from eq (9) of source paper
            weights = Ms*Ms-Ms
            sum_of_weights = sum(weights)
            single_event_avgs_2 = SingleEvtAvgTwoParticleCorr(Q2s, Ms)
            single_event_avgs_3 = SingleEvtAvgTwoParticleCorr(Q3s, Ms)
            # weighting already done with skipping the division by Ms*Ms-Ms in the function above
            #weighted_sum_2 = np.dot(weights, single_event_avgs_2)
            #weighted_sum_3 = np.dot(weights, single_event_avgs_3)

            if sum(single_event_avgs_3) > 0: # only positive flow coefficients usable
                v2s = np.append(v2s, np.sqrt(sum(single_event_avgs_2)/sum_of_weights)*np.ones(N_samples)) # sample_n images per event
                v3s = np.append(v3s, np.sqrt(sum(single_event_avgs_3)/sum_of_weights)*np.ones(N_samples))
                Ms_event = np.append(Ms_event, np.sum(Ms)*np.ones(sample_n)) # charged particle multiplicity per event
            else:
                skipped_images = np.append(skipped_images, images[-sample_n:])
                images = images[:-N_samples] # throw away images of last event
                skipped_v2s = np.append(skipped_v2s, np.sqrt(sum(single_event_avgs_2)/sum_of_weights))
                skipped_Ms = np.append(skipped_Ms, np.sum(Ms))

            # print("finished event {}".format(event_n))
            total_samples += sample_n
            total_particles += current_particle
            event_n += 1
    
    Ms_image = np.sum(images, axis=(1,2,3))  # array of multiplicities of each energy layer: images = [[histoE1, histomT1, histopT1], [histoE2, histomT2, histopT2], ...]

    flowdata = np.stack((v2s, v3s, Ms_image), axis=-1)
    np.savez_compressed('{}.npz'.format(fn),images=images,flow_data = flowdata)


if __name__ == '__main__':
    createImages()
    # print to check things
    # should be 937 252 particles
    # print('events', event_n-1)
    # print('samples', total_samples)
    # print('particles', total_particles)

    # print('total nch in images {nch} and skipped {skipped}'.format(nch = np.sum(Ms_event), skipped = np.sum(skipped_Ms)))
    # print('skipped {n} events with nch avg {avg}, v2s {v2s}'.format(n = len(skipped_v2s), avg = np.average(skipped_Ms), v2s = skipped_v2s))

    # print('v2s =', np.round(v2s, 4))
    # print('v3s = ', np.round(v3s, 4))

    # print('image check = {}'.format(np.sum(np.abs(image_diffs))))
    # print('charged particles in image')
    # TODO get the correct values for avg min max
    # print(images.shape)
    # print('avg {avg} min {min} max {max}'.format(avg=np.average(Ms_image),min=np.min(Ms_image), max=np.max(Ms_image)))

    # print(v2s.shape, v3s.shape, Ms_image.shape)
    # TODO: group image with corresponding vn and multiplicity
 