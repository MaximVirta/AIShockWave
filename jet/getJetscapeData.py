import numpy as np
import sys
#from particle import Particle


def getJetscapeData(jetscape_hadrons):
    print('loading hadron list')
    hadrons_in = np.loadtxt(jetscape_hadrons)
    print('loading complete')
    # output as a numpy array first dimension has events, second dimension particles, third dimension coordinates (pT,mass,eta,phi)
    hadrons_out = np.empty((0,9,4))

    # get values as arrays, one line contains 0:PSN 1:PID 2:?? 3:E 4:px 5:py 6:pz 7:eta 8:phi
    #Energy = hadrons_in[:,3]
    px = hadrons_in[:,4]
    py = hadrons_in[:,5]
    #pz = hadrons_in[:,6]
    
    eventNum = getEventNum(hadrons_in) # works

    pT = getPT(px, py)
    #mass = getMass(Energy,px,py,pz, hadrons_in)
    #pdg = Particle.from_pdgid(hadrons_in[:,1])
    #print(pdg)
    eta = hadrons_in[:,7]
    phi = hadrons_in[:,8]
    #print('phi {}'.format(phi))
    #print(np.nonzero(phi<0)[0])
    
    # couldn't get mass
    hadrons_out = np.array([np.column_stack((eventNum, pT, eta, phi))])[0]
    print('hadrons shape {}'.format(hadrons_out.shape))
    #np.savez_compressed('jetscape_particles_3big', hadrons_out)
    makeImage(hadrons_out)



# based on https://stackoverflow.com/questions/34027288/cumulative-counts-in-numpy-without-iteration
# zeroth element in each row is "PSN" and PSN = 0 means start of new event
def getEventNum(hadrons):
    eventNum = (hadrons[:,0] == 0).cumsum()
    return eventNum

# calculate pT from px py
def getPT(px, py):
    return np.sqrt(px*px + py*py)

# calculate mass from E px py pz
# E² = m²c⁴ + p²c² -> m² = E² - p²
def getMass(E, px, py, pz, hadrons_in):
    mass2 = E*E - px*px - py*py - pz*pz 
    if np.any(m2 < 0 for m2 in mass2):
        photons = np.nonzero(hadrons_in[:,1] == 22)[0]
        #print(photons)
        print(mass2[photons])
        sys.exit()
        indices = np.nonzero(mass2 < 0)[0] # indices where mass² < 0
        print('negative mass² count {}'.format(np.shape(indices)))
        print('total # of particles {}'.format(np.shape(hadrons_in)))
        print('negative m²s {}'.format(mass2[indices]))
        print('(hadrons_in[np.where(mass2 < 0)][1] {}'.format(hadrons_in[indices][0]))
        print('most neg mass2 {}'.format(np.min(mass2[indices])))
        negmassind = np.argmin(mass2[indices])
        print(negmassind)
        print(hadrons_in[negmassind])
        print(E[negmassind], px[negmassind], py[negmassind], pz[negmassind])
        print(E[negmassind]*E[negmassind] - px[negmassind]*px[negmassind]- py[negmassind]*py[negmassind] -pz[negmassind]*pz[negmassind])
        sys.exit()
        return E
    else:
        return np.sqrt(mass2)

def makeImage(particles):
    print('starting image creation')
    # particles is an array of shape (N_particles, 4), second axis contains n_event, pT, eta, phi
    N_events = int(particles[-1,0]) #event number of last particle

    for i in range(1, N_events+1):
        sample = particles[np.nonzero(particles[:,0] == i)[0]]
    
        if len(sample[np.nonzero(np.abs(sample[:,2]) <= 0.8)]) < 10: # skip if there are less than 10 particles within the image range (eta € [-0.8,0.8])
            continue
    
        eta = sample[:,2]
        phi = sample[:,3]
        histoE, xedges, yedges = np.histogram2d(eta, phi, bins=(32,32), range=[[-0.8,0.8],[0,2*np.pi]], weights=np.log(5.02)*np.ones(sample.shape[0]))
        histoPt, xedges, yedges = np.histogram2d(eta, phi, bins=(32,32), range=[[-0.8,0.8],[0,2*np.pi]], weights=sample[:,1])
    
        if i%1000 == 1: # to make np.concatenate work, create the first event of batch seperately, and overwrite histos already saved
            histos = np.array([[histoE, histoPt]])
            print('cleared images already on disk')
    
        else: # add to current the image list
            temp_histos = np.array([[histoE, histoPt]])
            histos = np.concatenate((histos, temp_histos), axis = 0)
    
        if i%1000 == 0: # save batch of 1000 images
            np.save('/projappl/project_2003154/MachineLearning/AIShockWave/jet/100kJetscapeImages/JetscapeImages3-{}'.format(i/1000), histos)
            print('saved image batch number {}'.format(i/1000))

if __name__=='__main__':
    hadronfile = sys.argv[1]
    getJetscapeData(hadronfile)
