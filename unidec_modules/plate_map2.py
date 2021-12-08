import plate_map as pm 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unidec_modules import unidectools as ud

rheaders = {"Reaction":str, "Species":str, "Concentration":float,
           "Units":str, "Mass":float, "Reagent Type":str, "Sequence":str}

pheaders = {"Well ID":{'dtype':str, 'long':True, 'short_row': False, 'short_col':False}, 
            "Type":{'dtype':str, 'long':True, 'short_row': True, 'short_col':True}, 
            "Reaction":{'dtype':str, 'long':True, 'short_row': True, 'short_col':True}, 
            "Time":{'dtype':str, 'long':True, 'short_row': True, 'short_col':True}}



def read_in_long(map_path, sheet_names = ['plate map', 'species map']):
    """

    Note - plate map must come first in sheet_names. 

    """
    rmap = pd.read_excel(map_path, 
                        dtype = rheaders,
                        sheet_name = sheet_names[1])

    pdata_types = {i[0]: i[1]['dtype'] for i in pheaders.items()} 
    pmap = pd.read_excel(map_path, dtype = pdata_types, index_col = "Well ID", skiprows = 1)

    # add valid column
    pheaders2 = [x for x in pheaders.keys() if pheaders[x]['long']]
    pmapdf = pm.empty_map(size = 6, header_names = pheaders)
    pmapdf.update(pmap)

    return rmap, pmapdf

# TODO: short map 

def process_data(rmap, pmapdf, spectra = None, groupby = 'Time', maptogroupby = 'plate_map'):
    """Collects metadata and data for each reaction on plate map into single dictionary of dictionaries
    
    :param groupby: what to sort each peak in TIC by, default = 'Time'
    :dtype groupby: str
    
    """
    # filter by reactions 
    reactions = {key:val for key, val in rmap.groupby('Reaction')}
    wells = {key:val for key, val in pmapdf.groupby('Reaction')} 
    
    # first iterate through reactions, then species to unpack plate/species map
    reactions2 = {}
    for rkey, rval in reactions.items():
        # convert each species dataframe row to dictionary containing metadata
        species = {skey:sval.to_dict('records')[0] for skey, sval in rval.groupby('Species')}
        reactions2[rkey] = species

    # next update each species with groupby parameter that defines each TIC peak (normally 'Time')
    for rkey, rdict in reactions2.items():
        # again, iterate through reactions then species for each reaction (even though there's normally just 1 reaction per plate)
        for species, metadict in rdict.items():
            metadict[groupby] = (wells[rkey][groupby])
            # create copy in which to house data from TIC and peak info (indexed by well ID?)]
            emptywellsdict = metadict[groupby].to_dict()
            metadict['spectra'] = emptywellsdict
            metadict['peaks'] = emptywellsdict.copy()
            # metadict['specattrs'] = emptywellsdict.copy()

    # length is used to check number of peaks in TIC matches groupby variable
    if maptogroupby == 'plate_map': 
        length = len(pmapdf[groupby])
    else:
        length = len(rmap[groupby])
    # returns dictionary of dictionaries containing each reaction on well plate and metadata for each species, 
    # as well as a list of the groupy parameter (normally time)
    # TODO: add check that reaction names appear in both reagent/reaction map and plate map

    return reactions2, length

def integrate_all(eng, int_range = None):
    """Creates new spectra attribute and stores areas for each peak there. (eventually put this in UniChrom)"""
    spectra = eng.data.spectra
    
    if int_range == None:
        lb, ub = -eng.config.peakwindow, eng.config.peakwindow
    elif type(int_range) == float:
        lb, ub = int_range[0], int_range[1]
    else:
        lb, ub = -int_range, int_range
        

    for s in spectra:
        peak_ints = []

        for p in s.pks.peaks:
                p.integralrange = [p.mass+lb, p.mass+ub]
                ints = (ud.integrate(s.massdat, p.integralrange[0], p.integralrange[1]))
                peak_ints.append((ints))
        
        s.integrals = peak_ints

    return eng

def update_vars(eng, pmap, skip_empty = False, groupby = 'Time'):
    """Updates vars for each spectra with well ID (starting from top) and stpulated 'groupby' parameter (usually 'Time')"""
    spectra = eng.data.spectra

    # filter pmap 
    if skip_empty == True:
        pmap = pmap[pmap['Type'] != 'empty']
    if len(spectra) == len(pmap):
        
        for i, s in enumerate(spectra):
            
            well_id = pmap.index[i]
            groupbyvar = pmap[groupby].iloc[i]
            s.attrs['Variable 1'] = well_id
            s.var1 = well_id
            s.attrs['Variable 2'] = groupbyvar
            s.var2 = groupbyvar

    else:
        raise Exception("Check plate map and TIC - are lengths the same?")
    return eng


def plot_all(eng, show_ints = True, xlim = []):
    """Plots each spectra stored in Unichrom class"""

    spectra = eng.data.spectra

    for s in spectra:
        plt.figure()
        plt.plot(s.massdat[:, 0], s.massdat[:, 1])

        for i, p in enumerate(s.pks.peaks):
            plt.scatter(p.mass, p.height, marker = 'x', color = p.color)
            if xlim != []:
                plt.xlim(xlim[0], xlim[1])
            if show_ints == True:
                ints = s.integrals[i][1]
                plt.fill_between(ints[:, 0], ints[:, 1], color = p.color)
                
    plt.show()


def export_spectra(eng, data_dict):
    """Updates data_dict with spectra"""
    # label spectra in dictionary
    spectra_dict = {s.var1:s for s in eng.data.spectra}
    # unpack metadata dictionary (reactions->species in each reaction->time point/groupby param [done in process_data])
    for rkey, rval in data_dict.items():
        for skey, sval in rval.items():
            for speckey, spectra in spectra_dict.items():
                sval['spectra'][speckey]=spectra
                # sval['specattrs'][speckey] = spectra.attrs
    return data_dict

def match_peaks(eng, data_dict, window = None, show_unlabelled = False):
    """Updates data_dict with peak info for each species (if mass is provided in species map)
    """
    if window == None:
        window = eng.config.peakwindow

    lb, ub = -window, window

    # unpack dictionary
    for rkey, rdict in data_dict.items():
        for spkey, spdict in rdict.items(): # where sp is species 
            # first clean the peaks dict
            for pkey, pval in spdict['peaks'].items():
                pval = 0

            masslb, massub = spdict['Mass']+lb, spdict['Mass']+ub

            for speckey, s in spdict['spectra'].items():
                for p in s.pks.peaks:
                    # print(p.mass)
                    if p.mass >= masslb and p.mass <= massub:
                        print("Mass {:.2f} Da matched to Species {} with mass {} Da".format(p.mass, 
                        spdict['Species'], spdict['Mass']))
                        spdict["peaks"][speckey] = p

                        # label peak with name of species
                        p.label = spdict['Reaction Name']

    
    return data_dict, eng

def match_peaks2(eng, data_dict2):
    pass


def plot_data(eng, data_dict):
    pass
