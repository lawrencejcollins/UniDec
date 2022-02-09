from numpy.lib.index_tricks import diag_indices_from
from pandas.io.stata import value_label_mismatch_doc
from unidec_modules.ChromEng import *
import plate_map as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from unidec_modules import unidectools as ud
from copy import deepcopy
import os
from datetime import datetime
import scipy
import threading
from multiprocessing import Pool

def michaelis(x, Vm, Km):
    return Vm*x/(x + Km)

def colorcodeclass(lst, cmap = 'rainbow'):

    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, len(lst)))
    for i, s in enumerate(lst):
        s.color = colors[i]
    return lst

class Species:
    def __init__(self, dictionary, name, speciesmap = None, pmap = None, peak = None, integral = []):
        self.__dict__.update(dictionary)
        self.__name__ = name
        self.peak = peak
        self.speciesmap = speciesmap
        self.pmap = pmap
        self.integral = []
        self.color = 'black'
        self.cmap = 'viridis'

    def __repr__(self):
        keys = [key for key, val in self.__dict__.items()]
        vals = [val for key, val in self.__dict__.items()]
        return "<"+self.__name__+ "("+", ".join("{} = {}".format(*t) for t in zip(keys, vals))+")>"

    # def __getattr__(self, attr):
    #     if attr.startswith("__"):
    #         raise AttributeError(attr)
    #     return self[attr]

    # def __deepcopy__(self, memo):
    #     return Species(deepcopy(dict(self)))

class Time:
    """Class that contains the data sorted by a stipulated groupby variable (normally time). """
    def __init__(self, time, species, coord = None, name = 'Time', speciesmap = None, pmap = None):

        self.species = species # list of species in each time point
        self.time = time
        self.coord = coord
        self.__name__= name
        self.spectra = None
        self.thresh = 1
        self.speciesmap = speciesmap
        self.pmap = pmap
        self.cmap = 'viridis'


        if self.species is not None:
            try:
                for s in self.species:
                    s.Time = self.time
            except:
                self.species.Time = self.time

    def __repr__(self):
        species_str = "species = (" + ", ".join(s.__name__ for s in self.species)+")"
        if self.spectra == None:
            return self.__name__+ "(time = {}, coord = {}, ".format(self.time, self.coord) + species_str + ")"
        else:
            return self.__name__+ "(time = {}, coord = {}, spectra = {}, ".format(self.time, self.coord, self.spectra)+ species_str + ")"

    def extract_masses(self):
        self.theory_masses = np.array([sp.Mass for sp in self.species])
        self.species_name = np.array([sp.__name__ for sp in self.species], dtype = str)
        self.data_masses= np.array([p.mass for p in self.spectra.pks.peaks])
        self.pks = np.array([p for p in self.spectra.pks.peaks])

# TODO: take out edited funcitons in ChromEng and put here
class SeqChrom(ChromEngine):

    def __init__(self):
        ChromEngine.__init__(self)

        self.rheaders = {"Reaction":str, "Species":str, "Concentration":float,
                        "Units":str, "Mass":float, "Reagent Type":str, "Sequence":str}

        self.pheaders = {"Well ID":{'dtype':str, 'long':True, 'short_row': False, 'short_col':False},
                        "Type":{'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                        "Reaction":{'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                        "Time":{'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                        "Substrate Conc":{'dtype':float, 'long':True, 'short_row':True, 'short_col':True},
                        "Protein Conc":{'dtype':float, 'long':True, 'short_row':True, 'short_col':True},
                        "Catalyst Conc":{'dtype':float, 'long':True, 'short_row':True, 'short_col':True},
                        "Units":{'dtype':str, 'long':True, 'short_row':True, 'short_col':True},
                        "Filename":{'dtype':str, 'long':True, 'short_row':True, 'short_col':True}}
                        # Substrate conc required if doing michaelis-menten analysis on same plate.
    def load_mzml(self, path, load_hdf5=True, clear_hdf5 = True, *args, **kwargs):


        self.path = path
        name = os.path.splitext(path)[0]
        if name[-5:].lower() == ".mzml":
            name = name[:-5]
        self.outpath = name + ".hdf5"
        self.setup_filenames(self.outpath)
        self.data.filename = self.outpath
        hdf5 = False
        self.clear()

        if clear_hdf5:
            try:
                os.remove(self.outpath)
            except:
                print("hdf5 does not exist")

        if os.path.isfile(self.outpath) and load_hdf5:
            print('Opening HDF5 File:', self.outpath)
            try:
                self.open(self.outpath)
                hdf5 = True
            except Exception as e:
                print("Error opening prior hdf5 file:", e)
        if not os.path.isfile(self.outpath):
            self.data.new_file(self.outpath)
            self.open(self.outpath)

        self.update_history()

        self.chromdat = ud.get_importer(path)
        self.tic = self.chromdat.get_tic()
        self.ticdat = np.array(self.tic)

        return hdf5

    def get_chrom_peaks(self, window=None, lb = None, ub = None): # LJC Edit
        # Cleanup TIC Data
        ticdat = deepcopy(self.ticdat)
        ticdat = ud.gsmooth(ticdat, 2)
        ticdat[:, 1] -= np.nanmin(ticdat[:, 1])
        # ticdat = ud.gaussian_backgroud_subtract(ticdat, 100)
        maxval = np.amax(ticdat[:, 1])
        ticdat[:, 1] /= maxval
        maxt = np.nanmax(ticdat[:, 0])
        mint = np.nanmin(ticdat[:, 0])

        # Set Window
        if window is None:
            window = self.config.chrom_peak_width

        # Set Threshold
        noise = ud.noise_level2(ticdat, percent=0.50)
        print("Noise Level:", noise, "Window:", window)

        # Detect Peaks
        peaks = ud.peakdetect_nonlinear(ticdat, window=window, threshold=noise)
        # peaks = ud.peakdetect(ticdat, window=window, threshold=noise)
        # Filter Peaks
        goodpeaks = []
        tranges = []
        diffs = np.diff(ticdat[:, 0])
        for p in peaks:
            fwhm, range = ud.calc_FWHM(p[0], ticdat)
            index = ud.nearest(ticdat[:, 0], p[0])
            if index >= len(diffs):
                index = len(diffs) - 1
            localdiff = diffs[index]
            if p[0] - fwhm / 2. < mint or p[0] + fwhm / 2. > maxt or fwhm > 4 * window or fwhm < localdiff * 2 or range[
                0] == p[0] or range[1] == p[0]:
                print("Bad Peak", p, fwhm, range)
                pass
            else:
                print(p[0], fwhm)
                goodpeaks.append(np.array(p))
                tranges.append(range)

        tranges = np.array(tranges)
        goodpeaks = np.array(goodpeaks)
        goodpeaks = goodpeaks[np.where(goodpeaks[:, 0] > lb)]
        self.chrompeaks = goodpeaks
        self.chrompeaks_tranges = tranges
        if lb != None:
            tranges = tranges[np.all(tranges > lb, axis = 1)]
            self.chrompeaks_tranges = tranges
        if ub != None:
            tranges = tranges[np.all(tranges < ub, axis = 1)]
            self.chrompeaks_tranges = tranges

        return goodpeaks, tranges

    def add_chrom_peaks2(self): # LJC Edit
        # self.get_chrom_peaks()
        times = self.chrompeaks_tranges
        self.data.clear()
        for i, t in enumerate(times):
            data = self.get_data_from_times(t[0], t[1])
            self.data.add_data(data, name=str(t[0]), attrs=self.attrs, export=False)

        self.data.export_hdf5()
    def get_files(self, directory, filetype):
        paths = []
        for dname, dirs, files in os.walk(directory):
            for fname in files:
                if fname[-len(filetype):] == filetype:

                    path = os.path.join(dname, fname)
                    paths.append(path)
        return paths

    def load_multi_single(self, directory, t0 = 1.9, t1 = 2.1, load_hdf5=True,
                        clear_hdf5=True):
        paths = self.get_files(directory, filetype="mzML")

        get_data_from_times = t0, t1
        # TODO: if HDF5 exists skip loading step
        self.load_multi_mzml(paths, load_hdf5=load_hdf5, clear_hdf5=clear_hdf5,
                            get_data_from_times=get_data_from_times)


    def load_multi_mzml(self, paths, chom_peak_width = 0.2,
                        chrom_lb = 1, chrom_ub = 5.5, plot = False,
                        name = "", load_hdf5 = True, clear_hdf5 = False,
                        get_data_from_times = None):
        """[summary]

        Args:
            paths ([type]): [description]
            id (list, optional): [description]. Defaults to [].
            chom_peak_width (float, optional): [description]. Defaults to 0.2.
            chrom_lb (int, optional): [description]. Defaults to 1.
            chrom_ub (float, optional): [description]. Defaults to 5.5.
            plot (bool, optional): [description]. Defaults to False.
            name (str, optional): [description]. Defaults to "".
            load_hdf5 (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        # Prep HDF5 - taken from ChromEng.load_mzml()
        self.paths = paths
        self.data.spectra = []


        self.data.clear()


        if name == "":
            path = os.path.realpath(paths[0])
            dir = os.path.dirname(path)
            self.name = dir

        self.outpath = name + ".hdf5"
        self.setup_filenames(self.outpath)
        self.data.filename = self.outpath
        hdf5 = False
        self.clear()

        if clear_hdf5:
            try:
                os.remove(self.outpath)
            except:
                print("hdf5 does not exist")

        if os.path.isfile(self.outpath) and load_hdf5:
            print('Opening HDF5 File:', self.outpath)

            try:
                self.open(self.outpath)
                hdf5 = True
            except Exception as e:
                print("Error opening prior hdf5 file:", e)

        if not os.path.isfile(self.outpath):
            self.data.new_file(self.outpath)
            self.open(self.outpath)

        self.update_history()
        ################################
        # open mzMLs



        if hdf5 == False:
            for i, p in enumerate(paths):
                eng = SeqChrom() # create new instance of UniChrom to pick peaks in separate TICs
                print(eng)
                eng.load_mzml(p)
                eng.config.chrom_peak_width = chom_peak_width
                goodpeaks, tranges = eng.get_chrom_peaks(lb = chrom_lb, ub = chrom_ub)
                times = eng.chrompeaks_tranges
                eng.data.clear()
                if get_data_from_times == None:
                    for i, t in enumerate(times):
                        data = eng.get_data_from_times(t[0], t[1])
                        self.data.add_data(data, name="{}_{}".format(str(t[0]), p),
                        attrs=eng.attrs, export=False) # append to main class object
                else:
                    try:
                        data = eng.get_data_from_times(get_data_from_times[0], get_data_from_times[1])
                        self.data.add_data(data, name="{}".format(p),
                        attrs=eng.attrs, export=False) # append to main class object
                    except Exception("check times are correct, moving to default 1.9 to 2.1"):
                        get_data_from_times = 2.1, 2.5
                        data = eng.get_data_from_times(get_data_from_times[0], get_data_from_times[1])
                        self.data.add_data(data, name="{}".format(p),
                        attrs=eng.attrs, export=False) # append to main class object



                if plot:
                    eng.plot_tic(peak_windows=True)


                print(i)
                print(len(self.data.spectra))

        ########################################


        # self.data.len = len(self.data.spectra)
        self.data.export_hdf5()
        print(len(self.data.spectra))
        return hdf5
        # TO DO: Check appending lists in correct order -> ID with filenames to order files???

        # note - clear option - keep track of indexes from TICs to access spectra?

    def _read_long(self, map_path,
    sheet_names = ['plate map', 'species map'], wellplatesize = None):
        """Plate map must come first in sheet names"""

        speciesmap = pd.read_excel(map_path,
                        dtype = self.rheaders,
                        sheet_name = sheet_names[1])

        pdata_types = {i[0]: i[1]['dtype'] for i in self.pheaders.items()}
        pmap = pd.read_excel(map_path, dtype = pdata_types, index_col = "Well ID", skiprows = 1)
        pmap[['Species', 'Valid']]= np.nan, True # column to list species objects, validation column
        # pmap['Time'].astype(float)
        # add valid column
        pheaders2 = [x for x in self.pheaders.keys() if self.pheaders[x]['long']]
        if wellplatesize != None:
            pmapdf = pm.empty_map(size = wellplatesize,
            header_names = self.pheaders)
            pmapdf.update(pmap)
            return speciesmap, pmapdf
        else:

        #     pmapdf = pd.DataFrame(columns=self.pheaders)
        #     pmapdf['Valid'] = True
        #     if 'Type' in pmapdf.columns:
        #         pmapdf['Type'] = 'empty'
        #     pmapdf.update(pmap)

            return speciesmap, pmap

    def _read_short(self,
                    map_path, sheet_names = ['plate map', 'species map']): # maybe del this? unless repeats?
        pass
    # TODO: short map

    def upload_map(self, map_path,
                    sheet_names = ['plate map', 'species map'],
                    mtype = 'long'):
        """[summary]

        Args:
            map_path ([type]): [Map type (mtype) can either be 'long' or 'short']
            sheet_names (list, optional): [description]. Defaults to ['plate map', 'species map'].
            mtype (str, optional): [description]. Defaults to 'long'.

        Returns:
            [type]: [description]
        """
        if mtype == 'long':
            speciesmap, pmap = self._read_long(map_path, sheet_names)

        if mtype == 'short':
            speciesmap, pmap = self._read_short(map_path, sheet_names)

        self.speciesmap = speciesmap
        self.pmap = pmap
        return self.speciesmap, self.pmap

    def get_auto_peak_width2(self, set=True):
        try:
            fwhm, psfun, mid = ud.auto_peak_width(self.data.data2)
            self.config.automzsig = fwhm
            self.config.autopsfun = psfun
            if set:
                self.config.psfun = psfun
                self.config.mzsig = fwhm
            print("Automatic Peak Width:", fwhm)
        except Exception as e:
            print("Failed Automatic Peak Width:", e)
            print(self.data.data2)


    def update_config(self, minmz = "", maxmz = "",
                        massub = 10000, masslb = 100000, peakthresh = 0.01,
                        subtype = 2, subbuff = 100, datanorm = 0,
                        numit = 100, massbins = 1, mzsig = 0, startz = 10,
                        endz = 100, zzsig = 1, psig = 1, beta = 0,
                        psfun = 0, peaknorm = 0, peakwindow = 10,
                        exnorm = 0,numz = 50, mtabsig = 0.0, molig = 0.0,
                        mzbins = 0.0, msig=0.0, smooth = 0, reductionpercent = 0.0,
                        aggressive = 0, rawflag = 0, nativezub = 1000.0, nativezlb=-1000.0,
                        poolflag = 2, noiseflag = 0, linflag = 2, isotopemode=0,
                        baselineflag = 1, orbimode=0, peakplotthresh = 0.1,
                        adductmass = 1.007276467, intthresh = 0):
        """[summary]

        Args:
            minmz (str, optional): [description]. Defaults to "".
            maxmz (str, optional): [description]. Defaults to "".
            massub (int, optional): [description]. Defaults to 10000.
            masslb (int, optional): [description]. Defaults to 100000.
            peakthresh (float, optional): [description]. Defaults to 0.01.
            subtype (int, optional): [background subtraction]. Defaults to 2.
            subbuff (int, optional): [background subtraction amount (subtract curved) 0 = 0ff,
            100 = good amount when on]. Defaults to 100.
            datanorm (int, optional): [data normalisation]. Defaults to 0.
            numit (int, optional): [number of iterations]. Defaults to 100.
            massbins (int, optional): [description]. Defaults to 1.
            mzsig (int, optional): [description]. Defaults to 0.
            startz (int, optional): [description]. Defaults to 10.
            endz (int, optional): [description]. Defaults to 100.
            zzsig (int, optional): [charge smooth width (smooth charge state distributions)]. Defaults to 1.
            psig (int, optional): [smooth nearby points (point smooth width, some = 1)]. Defaults to 0.
            beta (int, optional): [suppress artifacts (beta, some = 50)]. Defaults to 50.
            psfun (int, optional): [Peak shape function (gaussian, lorentzian, split G/L)]. Defaults to 0.
            peaknorm (int, optional): [description]. Defaults to 0.
            peakwindow (int, optional): [description]. Defaults to 10.
            exnorm (int, optional): [description]. Defaults to 0.
        """

        self.config.minmz = minmz
        self.config.maxmz = maxmz
        self.config.subtype = subtype #  - subtract curved
        self.config.subbuff = subbuff # background subtraction amount (subtract curved) 0 = 0ff, 100 = good amount when on
        self.config.datanorm = datanorm # turn off

        self.config.intthresh = intthresh
        # -- Deconvolution
        self.config.numit = numit # number of iterations
        self.config.adductmass = adductmass
        # mass range (default = 5000.0 to 500000.0 Da)
        self.config.massub = massub # upper
        self.config.masslb = masslb # lower

        self.config.massbins = massbins # sample mass every 0.1/1 Da?


        ###############
        self.config.numz = numz
        self.config.mtabsig = mtabsig
        self.config.molig = molig
        self.config.mzbins = mzbins
        # self.config.msig = msig # fwhm
        self.config.reductionpercent = reductionpercent
        self.config.aggressive = aggressive
        self.config.rawflag = rawflag
        self.config.nativezub = nativezub
        self.config.nativezlb = nativezlb
        self.config.poolflag = poolflag # m/z to mass transformation
        self.config.noiseflag = noiseflag
        self.config.linflag = linflag
        self.config.isotopemode = isotopemode
        self.config.baselineflag=baselineflag,
        self.config.orbimode=orbimode
        self.peakplotthresh = peakplotthresh

        # FWHM
        try:
            self.get_auto_peak_width2()

        except:
            self.config.mzsig = 1
            self.config.psfun = -1
        # self.config.mzsig =mzsig

        # charge range
        self.config.startz = 10
        self.config.endz = 100

        # smoothing
        self.config.zzsig = zzsig # charge smooth width (smooth charge state distributions)
        self.config.psig = psig # smooth nearby points (point smooth width, some = 1)
        self.config.beta = beta # suppress artifacts (beta, some = 50)
        self.config.smooth = smooth
        # self.config.psfun = psfun # Peak shape function (gaussian, lorentzian, split G/L)

        # -- Peak Selection and plotting
        self.config.peaknorm = peaknorm # Normalise peaks (0 = off)
        self.config.datanorm = datanorm
        self.config.peakwindow = peakwindow # peak window / Da
        self.config.exnorm = exnorm # extract normalisation
        self.config.peakthresh = peakthresh
        # self.config.nativeub = nativeub
        # self.config.nativelb = nativelb
        # self.data.export_hdf5(delete=False)
        # self.config.write_hdf5()

    def process_maps(self, groupby = 'Reaction', variable = 'Time', matchfilenames = False):
        """[summary]

        Args:
            groupby (str, optional): ['Reaction' or 'Substrate Conc']. Defaults to 'Reaction'.
        """
        # Reaction OR Substrate Conc
        # TODO: Add check for same reaction name in species and reaction map.
        # TODO: circumvent need for reaction name if all wells are same reaction.
        self.pmap2 = self.pmap[self.pmap['Type'] != 'empty']
        # self.pmap2.loc[:, 'Species'] = np.nan

        for skey, sval in self.speciesmap.groupby([groupby]):
            splist = [Species(spval.to_dict('records')[0], name = spkey) for spkey, spval in sval.groupby('Species')]
            splist = colorcodeclass(splist)

            for s in splist:
                self.pmap2.loc[:, s.__name__] = np.nan
                self.pmap2.loc[self.pmap2['Reaction']==skey, s.__name__] = self.pmap2.apply(lambda _:deepcopy(s), axis = 1)

            spnames = [s.__name__ for s in splist]
            self.pmap2.loc[self.pmap2['Reaction']==skey, 'Species'] = self.pmap2.apply(lambda _:spnames, axis = 1)

        # update_vars

        if len(self.data.spectra) == len(self.pmap2):
            for i, s in enumerate(self.data.spectra):
                if matchfilenames:
                    directory, name = os.path.split(s.name)

                    filt = self.pmap.loc[:, 'Filename']==name
                    self.pmap.loc[filt, 'Spectra']=s
                    s.var1 = self.pmap.loc[filt].index.name

                well_id = self.pmap2.index[i]
                timevar = self.pmap2[variable].iloc[i]
                s.attrs['Variable 1'] = well_id
                s.var1 = well_id
                s.attrs['Variable 2'] = timevar
                s.var2 = timevar
                self.pmap2.loc[well_id, 'Spectra'] = s
        else:
            raise Exception("Reaction map not same length as spectra.")



        spectra = {s.var1:s for s in self.data.spectra}

        # update species with well info/metadata


        for index, row in self.pmap2.iterrows():
            for specs in row['Species']:
                row[specs] = deepcopy(row[specs])
                row[specs].coord = row.name
        #         print(row[specs])
                vals = row[~row.index.isin(row['Species'])].to_dict()
                row[specs].__dict__.update(vals)

    def peak_match(self, window = 10, silent = False):
        """[summary]

        Args:
            window (int, optional): [In Daltons]. Defaults to 10.
        """
        intmat = np.array([])

        for index, row in self.pmap2.iterrows():

            rowints = np.array([])

            specieslist = list(row[row.index.isin(row['Species'])])
            theory_masses = np.array([sp.Mass for sp in specieslist])
            data_masses = np.array([p.mass for p in row['Spectra'].pks.peaks])
            pks = np.array([p for p in row['Spectra'].pks.peaks])

            # match algorithm
            tm, dm = np.meshgrid(theory_masses, data_masses)
            diff = abs(tm - dm)
            diff[diff>window] = np.nan
            for i, d in enumerate(diff):
                if np.isnan(d).all()==False:
                    minimum = np.nanargmin(d)
                    data_peak = data_masses[i]

                    specieslist[minimum].peak = pks[i]
                    specieslist[minimum].integral = pks[i].integral[0]
                    if silent != True:
                        print("{}, {} = {}".format(row[row.index.isin(row['Species'])][minimum].__name__, data_peak, pks[i]))

                    row[row.index.isin(row['Species'])][minimum].integral = pks[i].integral[0]
                    row[row.index.isin(row['Species'])][minimum].peak = pks[i]
                    np.append(rowints, pks[i].integral[0])
                    if silent != True:
                        print(row[row.index.isin(row['Species'])][minimum].integral)

    def normalise_peaks(self, silent = False):
        for index, row in self.pmap2.iterrows():
            ints = []
            for s in row[row['Species']]:
                if type(s.integral) != list:
                    ints.append(s.integral)

        #     ints = np.array([s.integral for s in row[row['Species']]])
            sum_ints = np.sum(ints)
            for s in row[row['Species']]:
                if type(s.integral) != list:
                    s.percentage = s.integral/sum_ints
                else:
                    s.percentage = 0
                if silent != True:
                    print("{}:{}".format(s.__name__, s.percentage))

    def update_vars(self, skip_empty = True, groupby = 'Time'):
        """[Updates vars for each spectra with well ID (starting from top) and stpulated 'groupby' parameter (usually 'Time')]

        Args:
            skip_empty (bool, optional): [removes empty wells from well plate]. Defaults to True.
            groupby (str, optional): [description]. Defaults to 'Time'.

        Raises:
            Exception: [description]
        """
        # filter pmap ]
        pmap = self.pmap
        pmap = pmap[pmap['Valid']==True]

        if skip_empty:
            pmap = self.pmap[self.pmap['Type'] != 'empty']


        if len(self.data.spectra) == len(pmap):
            for i, s in enumerate(self.data.spectra):
                well_id = pmap.index[i]
                groupbyvar = pmap[groupby].iloc[i]
                s.attrs['Variable 1'] = well_id
                s.var1 = well_id
                s.attrs['Variable 2'] = groupbyvar
                s.var2 = groupbyvar

        else:
            raise Exception("Compare map and TIC, try invalidate")

    def plot_tic(self, peak_windows = False, *args, **kwargs):

        fig, ax = plt.subplots(*args, **kwargs)
        ax.plot(self.tic[:, 0], self.tic[:, 1])

        if peak_windows == True:
            for w in self.chrompeaks_tranges:
                ax.axvspan(w[0], w[1], alpha = 0.3, color = 'orange')

        ax.set_ylabel("Intensity / a.u.")
        ax.set_xlabel("retention time / mins")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()

        # TODO: add export

    def _export_proc(self):
        """[Makes folder for data exports]

        Returns:
            [type]: [description]
        """
        dire, spec = os.path.split(self.path)
        spec = spec[:-5]
        path = os.path.join(dire, spec+"_export")
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        self.exp_path = path
        return path

    def _update_spectra(self):
        # TODO: find reactions dict and store at self.reactions
        spectra = {s.var1:s for s in self.data.spectra}
        try:
            for rkey, rval in self.rxn_group.items():
                    for t in rval:
                        t.spectra = spectra[t.coord]
                        t.extract_masses()

        except Exception:
            print("Error!")

        return self.rxn_group

    def process(self, groupby = 'Time', skip_empty = True):
        r_species = {}
        r_groupby = {}

        self.update_vars(skip_empty, groupby=groupby)

        for rkey, rval in self.speciesmap.groupby(['Reaction']): #TODO: add option to sort by conc.. if no rxn/no conc then don't groupby
            r_species[rkey] = [Species(spval.to_dict('records')[0], name = spkey) for spkey, spval in rval.groupby('Species')]

        #------------------------------------------------------------------------
        for rkey, rval in self.pmap.groupby('Reaction'):
            # update each species with time points and coordinates
            for species in r_species[rkey]:
                setattr(species, groupby, rval[groupby].values)
                setattr(species, "wells", np.array(rval.index))
            # ----------------------------------------------------
            # color code species
            r_species[rkey] = colorcodeclass(r_species[rkey])

            time = rval[groupby].values # change??
            coords = np.array(rval[groupby].index)
            r_groupby[rkey] = []
            for t, c in zip(time, coords):
                # make new species instance for each time point - avoids some memory issues
                species = [deepcopy(s) for s in r_species[rkey]]

                r_groupby[rkey].append(Time(species = species, time = float(t), coord = c))
        # -----------------------------------------------------------------------
        self.rxn_group = r_groupby
        self._update_spectra()

        return self.rxn_group

    def integrate_all(self, int_range = None):
        """[Creates new spectra attribute and stores areas for each peak there.
        (eventually put this in UniChrom)]

        Args:
            int_range ([type], optional): [description]. Defaults to None.
        """
        if int_range == None:
            lb, ub = -self.config.peakwindow, self.config.peakwindow
        elif type(int_range) == float:
            lb, ub = int_range[0], int_range[1]
        else:
            lb, ub = -int_range, int_range

        for s in self.data.spectra:
            peak_ints = []

            for p in s.pks.peaks:
                    p.integralrange = [p.mass+lb, p.mass+ub]
                    ints = (ud.integrate(s.massdat, p.integralrange[0], p.integralrange[1]))
                    p.integral = ints
                    peak_ints.append((ints))

            s.integrals = peak_ints

    def _set_spectra_colors(self, cmap = 'rainbow'):

        cmap = plt.get_cmap(cmap)
        colors = cmap(np.linspace(0, 1, len(self.data.spectra)))
        for i, s in enumerate(self.data.spectra):
            s.color = colors[i]

    def plot_all(self, dtype = 'massdat', show_ints = True, xlim = [],
                combine = False, cmap = 'Set1', export = False):
        """[Plots each spectra stored in Unichrom class]

        Args:
            dtype (str, optional): [description]. Defaults to 'massdat'.
            show_ints (bool, optional): [description]. Defaults to True.
            xlim (list, optional): [description]. Defaults to [].
            combine (bool, optional): [description]. Defaults to False.
            cmap (str, optional): [description]. Defaults to 'Set1'.
            export (bool, optional): [description]. Defaults to False.
        """
        self._set_spectra_colors(cmap)

        spectra = self.data.spectra
        xcounter = 0
        ycounter = 0

        if export:
            path = self._export_proc()

        if combine == True:
            fig, ax = plt.subplots(dpi = 100)

        for s in spectra:
            data = getattr(s, dtype)
            if combine == False:
                fig, ax = plt.subplots()
                ax.plot(data[:, 0], data[:, 1], color = s.color, linewidth = 0.5)
                ax.set_title(s.name)

            if combine == True:
                ax.plot(data[:, 0]+xcounter, data[:, 1]+ycounter, color = s.color, linewidth = 0.5, label = s.name)

            ax.set_xlabel('Mass / Da')
            ax.set_ylabel('Intensity')
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if dtype == 'massdat':
                for i, p in enumerate(s.pks.peaks):
                    if combine == False:
                        ax.scatter(p.mass, p.height, marker = '^', color = p.color, s=10)
                    if xlim != []:
                        ax.set_xlim(xlim[0], xlim[1])
                        if combine == True:
                            ax.set_xlim(xlim[0], xlim[1]+xcounter)
                    if show_ints == True:
                        ints = s.integrals[i][1]
                        if combine == False:
                            ax.fill_between(ints[:, 0], ints[:, 1], color = p.color, alpha = 0.3)
                        else:
                            ax.fill_between(ints[:, 0]+xcounter, ints[:, 1]+ycounter, ycounter, color = p.color, alpha = 0.25)
                            ax.legend()
            xcounter+=data[:, 0].max()*0.05
            ycounter+=data[:, 1].max()*0.05

            if combine == False and export==True:
                now = datetime.now()
                now = now.strftime("%d-%m-%Y-%H%M")
                name = now+"_"+dtype+"_"+s.var1+".png"
                figpath = os.path.join(path, name)
                try:
                    plt.savefig(figpath)
                except Exception:
                    print("Error saving")

        if combine and export:
            now = datetime.now()
            now = now.strftime("%d-%m-%Y-%H%M")
            name = now+"_"+dtype+"_combined.png"
            figpath = os.path.join(path, name)
            try:
                plt.savefig(figpath)
            except Exception:
                print("Error saving")
        plt.show()

    def match_peaks(self, window = 10):
        """[Peak matching algorithm with species defined in reaction map. Window is in Daltons.]

        Args:
            window (int, optional): [description]. Defaults to 10.
        """

        for rkey, rval in self.rxn_group.items():
            for t in rval:
                tm, dm = np.meshgrid(t.theory_masses, t.data_masses)
                diff = abs(tm - dm)
                diff[diff>window] = np.nan
                for i, d in enumerate(diff):
                    if np.isnan(d).all()==False:
                        minimum = np.nanargmin(d)
                        data_peak = t.data_masses[i]
                        t.species[minimum].peak = t.pks[i]
                        t.species[minimum].integral = t.pks[i].integral[0]
                        print("{} = {}".format(data_peak, t.pks[i]))

    def extract_data(self, groupby='Time', species = None, datatype = 'percentage'):
        pmap = self._get_valid(pmap="pmap2")
        df_dict = {}
        for k, v in pmap.groupby('Reaction'):
            group = v[groupby]
            speciesdct = {}
            speciestimedct = {}

            for index, row in v.iterrows():

                if species ==None:
                    species = row.Species
                if len(species) == 1:
                    species = [species]

                for s in species:
                    if s in speciesdct:
                        speciesdct[s].append(getattr(row[s], datatype))
                        speciestimedct[s].append(row[groupby])
                    else:
                        speciesdct[s] = [getattr(row[s], datatype)]
                        speciestimedct[s] = [row[groupby]]

            df = pd.DataFrame(speciesdct, index = group)
            df_dict[k]=df
        # if len(df_dict)==1:
        #     df_dict = df_dict[0]
        # for name, y in speciesdct.items():
        #     arr = np.array(y)==0
        #     if~arr.all():
        #         plt.plot(time, y, label = name)
        #         plt.legend(loc = 'center right')
        #         plt.title(self.path)
        #         plt.show()
        self.data_df = df_dict

    def plot_data(self, plot_type = None, species = None, groupby = 'Time', datatype = 'percentage'):

        self.extract_data(species = species, groupby=groupby, datatype=datatype)

        for key, df in self.data_df.items():

            if plot_type =='bar':
                df.plot.bar(rot=0)

                plt.ylabel("Percentage Intensity")
                plt.title(key)
                plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.show()
            else:
                for species in df.columns:
                    group = np.array(df[species])
                    if ~np.all((group==0)):

                        if plot_type==None:
                            df[species].plot(label = species)

                        plt.xlabel("{}".format(groupby))
                        plt.ylabel("Percentage Intensity")
                        plt.title(key)
                        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
                plt.show()

    def initial_rate(self, thresh = 0.05, plot = True):
        """[Generates plots of initial rate over a certain threshold,
        e.g. the first 5% of data points]

        Args:
            thresh (float, optional): [Initial rate fraction over data points]. Defaults to 0.05.
            plot (bool, optional): Defaults to True.
        """
        ratedct={}
        for key, data in self.data_df.items():
            x = np.array(data.index, dtype = float)
            x1i = int(len(x)*thresh -1)
            if x1i == 0:
                x1i = int(len(x)*thresh)
            if x1i == 0:
                x1i = 1
            x1 = float(x[x1i])

            y1 = data.iloc[x1i, :]
            y0 = data.iloc[0, :]
            x0 = float(x[0])

            m = (y0-y1)/(x0-x1)
            ratedct[key] = m
            c = y1-m*x1
            for species in data.columns:
                vy = m[species]*x +c[species]
                fig, ax = plt.subplots()
                ax.plot(x, data[species], label = species)
                ax.plot(x, vy, linestyle = 'dotted')
                ax.plot((x0, x1), (y1[species], y1[species]), color = 'purple', linestyle = 'dashed', linewidth = 0.5)
                ax.plot((x1, x1), (y0[species], y1[species]), color = 'purple', linestyle = 'dashed', linewidth = 0.5)
                ax.legend(bbox_to_anchor=(1,1), loc="upper left")
            plt.show()
            self.ratedict = ratedct

    def michaelis_analysis(self, x, y, *args, **kwargs):

        params, covar = scipy.optimize.curve_fit(michaelis, x, y, absolute_sigma = True, *args, **kwargs)
        return params

        # self.species_kinetics[rkey][species] = params

    def invalidate(self, name, group = 'Time'):
        if type(name) != list:
            name = [name]
        for n in name:
            filt = self.pmap2.loc[:, group].astype(type(n))==n
            try:
                self.pmap2.loc[filt, 'Valid'] = False
                print("{}:{} successfully invalidated".format(group, n))
            except Exception:
                print("Invalidation failed for pmap2")
            try:
                self.pmap.loc[filt, 'Valid'] = False
                print("{}:{} successfully invalidated".format(group, n))
            except Exception:
                print("Invalidation failed for pmap")

    def _get_valid(self, pmap = "pmap2"):
        self.pmap3 = getattr(self, pmap)
        self.pmap3 = self.pmap3[self.pmap3['Valid']==True]
        return self.pmap3

class SeqAnalysis(SeqChrom):
    def __init__(self):
        SeqChrom.__init__(self)
        pass

    # def _load_spectra(self, spectra_paths = [], spectra_dirpath = None):
    #     eng = SeqChrom()
    #     # option 1: list of spectra paths
    #     if len(spectra_paths) != 0:
    #         engines = [eng.load_mzml(s) for s in spectra_paths]
    #         return engines
    #     # create engines arr
    #     #########################################################################
    #     # option 2: provide folder containing spectra mzML files
    #     elif spectra_dirpath is not None:

    #     else:
    #         raise Exception("No spectra found")

    def _load_spectra2(self, dir):
        """Load folders containing spectra - where each folder is a separate reaction"""
        self.rxn_dict = {}
        self.plate_dict = {}
        for i, folder in enumerate(dir):
            filenames = os.listdir(folder)
            engines = []

            for dname, dirs, files in os.walk(folder):
                counter = 0
                for fname in files:
                    eng = SeqChrom()
                    if fname[-4:] == "mzML": # check if correct format

                        spectra_path = os.path.join(dname, fname)
                        eng.load_mzml(spectra_path)
                        print("loaded {}".format(spectra_path))
                        engines.append(eng)

                    if fname[-3:] == "csv":
                        plate_path = os.path.join(dname, fname)
                        counter += 1



                # check plate map correctly in folder
                if counter != 0:
                    raise Exception("no plate map in folder") # TODO: custom exceptions

            for eng in engines:
                eng.upload_map(plate_path)

            self.rxn_dict[folder] = engines




    def _load_plate_maps(self, plate_map, plate_map_dirpath):
        pass

    def load(self, spectra_paths = [], plate_map = None, spectra_dirpath = None, plate_map_dirpath = None):
        self._load_spectra(spectra_paths, spectra_dirpath)


    def set_config(self, masslb = 10000, massub = 100000):
        pass
    def unidec_all(self, chrom_peak_width = 0.05, peakthresh = 0.01, ticmin = 1,
    ticmax = 5.5, masslb = 10000, massub = 100000, minmz = "", maxmz = ""):

        for engines, folder in self.rxn_dict.values():
            for eng in engines:
                eng.get_chrom_peaks(lb = ticmin, ub = ticmax)
                eng.add_chrom_peaks2()
        #         eng.plot_tic(peak_windows = True)
                eng.update_config(masslb = masslb, massub = massub, minmz = minmz, maxmz = maxmz, peakthresh = peakthresh)
                eng.process_data()
                eng.run_unidec()
                eng.pick_peaks()
                eng.integrate_all()
        #         eng.plot_all(dtype = 'massdat', combine = True, cmap = 'viridis', xlim = [41000, 43000])
                eng.process_maps()
                eng.peak_match()
                eng.normalise_peaks()

    def unidec_to_df(self, species = None, datatype = 'percentage', rxndct = {}, groupby = 'Reaction'):

        dflist = []
        for engines, folder in self.rxn_dict.values():
            for eng in engines:
                rxns_tc = []
                for k, v in eng.pmap2.groupby(groupby):

                    time = v['Time']
                    speciesdct = {}
                    speciestimedct = {}

                    for index, row in v.iterrows():
                        if species == None:
                            species = row.Species

                        if len(species) == 1:
                            species = [species]

                        for s in species:
                            if s in speciesdct:
                                speciesdct[s].append(getattr(row[s], datatype))
                                speciestimedct[s].append(row['Time'])

                            else:
                                speciesdct[s] = [getattr(row[s], datatype)]
                                speciestimedct[s] = [row['Time']]

                    df = pd.DataFrame(speciesdct, index = time)
                    rxns_tc.append(df)

                if len(rxns_tc) == 1:
                    rxns_tc = rxns_tc[0]

            self.data_df = pd.concat(dflist)
            return self.data_df


        def analyze(self, key_species):
            pass