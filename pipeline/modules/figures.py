import os
import glob
import math
import copy
import itertools
import numpy as np
import pandas as pd
import sys, os.path
from astropy.io import fits
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy

sys.path.append(os.path.abspath('../'))
from pipeline.settings import config
from pipeline.helpers import boolean, remove_n_path_levels, get_star_spot_ratio

# a Spectrum object will only contain information regarding 1
# spectrum along with the parameters associated with that spectrum
class Spectrum:
    def __init__(self, params={}, wvs=[], uncalib_spect=[], uncalib_error=[], calib_spect=[], calib_error=[]):
        self.params = params
        self.wvs = wvs
        self.uncalib_spect = uncalib_spect
        self.uncalib_error = uncalib_error
        self.calib_spect = calib_spect
        self.calib_error = calib_error

# the Figures class will be tasked with reading in the spectra data and 
# creating figures from the data. There will be a dictionary of cases that
# maps to a list of Spectrum objects for that case. The list of Spectrum
# objects can then be resize to various dimensions depending on the parameters
# of each Spectrum object
class Figures:
    def __init__(self):
        # obtain list of case dir paths
        self.root_dir = config['Paths']['root_dir']
        self.root_figures_dir = os.path.join(self.root_dir, 'results/figures')
        self.case_dirs = config['Paths']['case_dir'].split(',')
        self.figures_dirs = []

        self.klipfm_files = {}
        for case in self.case_dirs:
            basename = os.path.join(case, 'klip/klipped/fm_spectra')
            path_to_name = os.path.join(self.root_dir, basename)
            temp = []
            for dirpath, subdirs, files in os.walk(path_to_name):
                for x in files:
                    if x.endswith(".fits"):
                        temp.append(os.path.join(dirpath, x))
            
            if len(temp) == 0:
                raise Exception('There are no klipped images for case {}!'.format(case))

            self.klipfm_files.update({os.path.basename(case):temp[0]})
            


        self.case_to_figure_dir = {}
        for i,case in enumerate(self.case_dirs):
            self.figures_dirs.append(os.path.join(case, 'results/figures'))
            if not os.path.isabs(case):
                self.case_dirs[i] = os.path.join(self.root_dir,case)
                self.figures_dirs[i] = os.path.join(self.root_dir,self.figures_dirs[i])
                self.case_to_figure_dir.update({os.path.basename(case):self.figures_dirs[i]})


            
        # check if figure directory exists. If not, create it.
        for figures_dir in self.figures_dirs:
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)

        # check if the root figures directory exists. If not, create it.
        if not os.path.exists(self.root_figures_dir):
            os.makedirs(self.root_figures_dir)

        # for each case, get all spectra associated with that case and put it
        # into a dictionary of case - list of Spectrum objects pair
        self.cases = {}
        for case in self.case_dirs: # iterating through each case
            case_spectra_list = []
            spectra_dir = os.path.join(case, 'results/spectra')
            dirs = self._get_all_subdirs(spectra_dir)
            for spectra in dirs: # iterating through each directory in a case
                if len(os.listdir(spectra)) != 0:
                    case_spectra_list = case_spectra_list + self._read_in_data(spectra)
            # update dictionary
            self.cases.update({os.path.basename(case):case_spectra_list})
    
        self.colors = config['Figure Parameters']['colors'].split(',')

    # create figues for spectra that is not calibrated. The output figures are 
    # then stored in their respective case directory
    def uncalib_fig(self, overwrite=False):
        # argument is section name of uncalibrated figures in config
        self._plot_small_large_spect('Uncalibrated Figures', overwrite)


    # create figures for spectra that is calibrated. The output figures are 
    # then stored in their respective case directory
    def calib_fig(self, overwrite=False):
        # argument is section name of uncalibrated figures in config
        self._plot_small_large_spect('Calibrated Figures', overwrite)


    # create figures for the difference in magnitude. The output figures are 
    # then stored in their respective case directory
    def mag_fig(self, overwrite=False):
        bands = {'J': 1.252, 'H': 1.6365, 'K': 2.197}
        # iterate through all the cases
        for casename, spectra in self.cases.items():
            figure_dir = self.case_to_figure_dir[casename]

            # case where there are existing figures
            if not overwrite and self._file_with_name(figure_dir, config['Magnitude Figures']['basename']):
                continue

            s = self._reshape(spectra, config['Figure Parameters']['regroup'])
            # then iterate thorugh each parameter group specified
            for spect_group in s:
                # create plot
                fig, ax = plt.subplots()
                # add lines to plot
                for i,spect in enumerate(spect_group):
                    # find the satellite to spot ratio
                    fits_image_filename = self.klipfm_files[casename]
                    hdul = fits.open(fits_image_filename)
                    hdr = hdul[0].header
                    try:
                        mod = float(hdr['X_GRDAMP'])
                    except:
                        return
                    mjd = float(hdr['MJD'])
                    
                    if boolean(config['Magnitude Figures']['manual_attenutation_factor']) != None:
                        mod = float(config['Magnitude Figures']['manual_attenutation_factor'])
                    mod *= 1000

                    spot_to_star_ratio = get_star_spot_ratio(mod, spect.wvs, mjd, manual=boolean(config['Magnitude Figures']['manual_attenutation_factor']))

                    # find a list of slice values that correspond to the bands J, H, and K
                    corresponding_slices = []
                    mag_wvs = []
                    for key,val in bands.items():
                        corres_val = min(spect.wvs, key=lambda x:abs(x-val))
                        corresponding_slices.append(spect.wvs.tolist().index(corres_val))
                        mag_wvs.append(corres_val)
                    
                    # convert uncalibrated spectra values and errors into ufloat type to correctly propogate 
                    # error bars
                    mag_ufloat = []
                    for s in corresponding_slices:
                        mag_ufloat.append(ufloat(spect.uncalib_spect[s], spect.uncalib_error[s]))
                    
                    # calculate dmag
                    dmag_ufloat = -2.5 * unumpy.log10(mag_ufloat * spot_to_star_ratio[corresponding_slices])

                    # separate values so it's easier to graph
                    dmag_value = []
                    dmag_err = []
                    for dmag in dmag_ufloat:
                        dmag_value.append(dmag.nominal_value)
                        dmag_err.append(dmag.std_dev)

                    # dynamically generate the title extension to make title more descriptive
                    title_extension = ' ( '
                    dynamic_title_extension = ' ( '
                    for key,val in spect.params.items():
                        title_extension = title_extension + '{}={} '.format(key, val)
                        if key == config['Figure Parameters']['regroup']:
                            dynamic_title_extension = dynamic_title_extension + '{}={}'.format(key,val)
                    title_extension = title_extension + ')'
                    dynamic_title_extension = dynamic_title_extension + ')'

                    # set plot attributes
                    ax.set_title(config['Magnitude Figures']['title_base'] + dynamic_title_extension)
                    ax.set(xlabel=config['Magnitude Figures']['x_axis'], ylabel=config['Magnitude Figures']['y_axis'])

                    # plot graph
                    ax.errorbar(mag_wvs, dmag_value, yerr=dmag_err, 
                                label=config['Figure Parameters']['legend_label'], 
                                capsize=3, color=self.colors[0])

                # plotting reference graph
                ref_path = config['Magnitude Figures']['reference_path']
                if ref_path != '':
                    if not os.path.isfile(ref_path):
                        raise Exception('Please enter a valid path to reference')
                    ref_spect = self._get_ref_spectra(ref_path)
                    if ref_spect.err_q:
                        ax.errorbar(ref_spect.wvs, ref_spect.uncalib_spect, yerr=ref_spect.uncalib_error, 
                                                        label=config['Magnitude Figures']['reference_label'], capsize=3, 
                                                        color='C0')
                    else:
                        ax.plot(ref_spect.wvs, ref_spect.uncalib_spect,
                                                    label=config['Magnitude Figures']['reference_label'],
                                                    color='C0')

                # adding the legend
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                fig.savefig(os.path.join(figure_dir,
                            '{} {}.png'.format(config['Magnitude Figures']['basename'], dynamic_title_extension)))
                


    # create figure that combines the calibrated spectra across different
    # cases
    def all_obs_fig(self, overwrite=False):
        # case where there are existing figures
        fs = os.listdir(self.root_figures_dir)
        if not overwrite and len(fs) != 0:
            return

        order_list = config['All Observation Figures']['regroup'].split(',')
        self.ol = order_list
        def obs_fig_recur(lst, order):
            if len(order) == 1:
                attr = order[0]
                spect_group = self._reshape(lst, attr)

                ncols = int(config['Figure Parameters']['n_columns'])
                fig0 = plt.figure(figsize=(20,20)) # figure of individal spectra with ref
                ax0 = fig0.subplots(nrows=math.ceil(len(spect_group)/ncols) + 1, ncols=ncols)
                subplot_tracker = [] # keeps track of the subplots that has been used

                for i,s in enumerate(spect_group):
                    # generate title extension for each plot
                    title_extension = ' ( '
                    for key in self.ol:
                        title_extension += '{}={} '.format(key, s[0].params[key])
                    title_extension += ')'
                
                    # subplot graph labels
                    title = config['All Observation Figures']['title_base'] + title_extension
                    x_axis = config['All Observation Figures']['x_axis']
                    y_axis = config['All Observation Figures']['y_axis']

                    # constructing actual subplot graphs
                    x_coord = math.floor(i / ncols)
                    y_coord = i % ncols
                    subplot_tracker.append(ax0[x_coord, y_coord])
                    ax0[x_coord, y_coord].set_title(title)
                    ax0[x_coord, y_coord].set(xlabel=x_axis, ylabel=y_axis)
                    ax0[x_coord, y_coord].set_ylim([0,3])

                    # plotting a spectrum for each case in the subplots
                    for j,spect in enumerate(s):
                        masked_wvs = spect.wvs
                        if boolean(config['All Observation Figures']['masking']):
                            masked_wvs = self._mask_wvs(masked_wvs)
                        ax0[x_coord, y_coord].errorbar(masked_wvs, spect.calib_spect, yerr=spect.calib_error, 
                                                        label=os.path.basename(spect.params['case']), capsize=3, 
                                                        color=('C'+str(j+1)))
                    
                    
                    # adding reference to subplot
                    ref_path = config['All Observation Figures']['reference_path']
                    if ref_path != '':
                        if not os.path.isfile(ref_path):
                            raise Exception('Please enter a valid path to reference')
                        ref_spect = self._get_ref_spectra(ref_path)
                        if ref_spect.err_q:
                            ax0[x_coord, y_coord].errorbar(ref_spect.wvs, ref_spect.uncalib_spect, yerr=ref_spect.uncalib_error, 
                                                            label=config['All Observation Figures']['reference_label'], capsize=3, 
                                                            color='C0')
                        else:
                            ax0[x_coord, y_coord].plot(ref_spect.wvs, ref_spect.uncalib_spect,
                                                        label=config['All Observation Figures']['reference_label'],
                                                        color='C0')
                    
                    #ax0[x_coord,y_coord].legend()

                # loop through all the axes in ax0 and determine if it has been used.
                # If not, delete the subplot
                for a in ax0.flatten():
                    if not(a in subplot_tracker):
                        fig0.delaxes(a)
                fig0.tight_layout()

                # export plots
                fig0.savefig(os.path.join(self.root_figures_dir,
                            '{} {}.png'.format(config['All Observation Figures']['basename'],title_extension)))

                    

            else:
                curr_order = order[0]
                o = order[1:]
                reshpaed_list = self._reshape(lst, curr_order)
                for l in reshpaed_list:
                    obs_fig_recur(l, o)
        
        # concatenate all spectra list in self.cases into one giant list
        all_spectra = []
        for casename, spect_list in self.cases.items():
            all_spectra += spect_list
        
        obs_fig_recur(all_spectra, order_list)
    



    # finds the first order child of a directory given by path
    def _get_all_subdirs(self, path):
        subdirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
        return subdirs

    # given a directory where the spectra and errors are located,
    # create a list of Spectrum objects by parsing the directory name and 
    # reading the files in the directory. 
    def _read_in_data(self, directory):
        # list of spectrum objects
        spectrums = []

        # the set of number of modes used for KLIP in string form
        numbasis_str = config['Klip-static']['numbasis'].split(',')

        # parse directory name to get params
        param = {'case':remove_n_path_levels(directory, 3)}
        dir_in_list = os.path.basename(directory).split('_')
        for i in range(1,len(dir_in_list) - 1):
            param_in_list = dir_in_list[i].split('=')
            param.update({param_in_list[0]:param_in_list[1]})

        # read in data files
        files = glob.glob(os.path.join(directory, '*.csv'))
        for f in files:
            f_basename = os.path.basename(f)
            if 'calibrated' in f_basename and 'error' not in f_basename:
                calib_spect = pd.read_csv(f)
            elif 'calibrated' in f_basename and 'error' in f_basename:
                calib_error = pd.read_csv(f)
            elif 'uncalib' in f_basename and 'error' not in f_basename:
                uncalib_spect = pd.read_csv(f)
            elif 'uncalib' in f_basename and 'error' in f_basename:
                uncalib_error = pd.read_csv(f)

        # get list of wavelengths
        wvs = calib_spect['wvs'].to_numpy()
        for basis in numbasis_str:
            # get corresponding column in datasets
            cs = calib_spect[basis].to_numpy()
            ce = calib_error[basis].to_numpy()
            ucs = uncalib_spect[basis].to_numpy()
            uce = uncalib_error[basis].to_numpy()
            p = copy.copy(param)
            p.update({'KL':int(basis)})

            # create and append the Spectrum objects
            s = Spectrum(p, wvs, ucs, uce, cs, ce)
            spectrums.append(s)

        return spectrums

    # reshapes the array of Spectrum objects based on different values that 
    # correspond to certain attributes. Spectrum objects with the same 
    # attr values will be grouped together
    def _reshape(self, arr, attr):
        result = []
        for spect in arr:
            for item in result:
                if item[0].params[attr] == spect.params[attr]:
                    item.append(spect)
                    break
            else:
                result.append([spect])
        return result

        
    # from the config file, read in the location of the reference spectra, 
    # find the reference spectra, and retrun a Spectrum object
    def _get_ref_spectra(self, path):
        if not os.path.isfile(path):
            raise Exception('This is not a reference spectrum.')
        df = pd.read_csv(path)
        arr_df = df.to_numpy().transpose()
        wvs = arr_df[0]
        val = arr_df[1]
        try:
            err = arr_df[2]
            spectrum = Spectrum(wvs=wvs, uncalib_spect=val, uncalib_error=err)
            spectrum.err_q = True
        except:
            spectrum = Spectrum(wvs=wvs, uncalib_spect=val)
            spectrum.err_q = False
        
        return spectrum

    # takes in a set of wavelengths and masks them such that the ones that they 
    # only lie in bandpass ranges
    def _mask_wvs(self, wvs):
        j_mag_range = (1.176, 1.328)
        h_mag_range = (1.490, 1.783)
        k_mag_range = (2.019, 2.375)

        mask = []
        for wv in wvs:
            b = self._in_band_range(wv, j_mag_range) or self._in_band_range(wv, h_mag_range) or self._in_band_range(wv, k_mag_range)
            mask.append(not b)

        return np.ma.masked_array(wvs, mask=mask)

    # determine if a value is within the given range. The range argument is a 
    # tuple of 2 values, (lower, upper)
    def _in_band_range(self, val, rang):
        if val >= rang[0] and val <= rang[1]:
            return True
        return False
    
    def _file_with_name(self, path, name):
        files = glob.glob(os.path.join(path, '*.png'))
        for f in files:
            if name in os.path.basename(f):
                return True
        return False

    # since self.uncalib_fig and self.calib_fig are extremely similar in 
    # functionality, both their functions can be generalized with the 
    # function self._plot_small_large_spect which takes in a key string 
    # for a section of the config file.
    def _plot_small_large_spect(self, key_string, overwrite):
        # check if entered key string exists in the config file
        if not config.has_section(key_string):
            raise Exception('The entered key string does not exist!')

        # iterate through all the cases
        for casename, spectra in self.cases.items():
            figure_dir = self.case_to_figure_dir[casename]

            # case where there are existing figures
            if not overwrite and self._file_with_name(figure_dir, config[key_string]['basename']):
                continue

            s = self._reshape(spectra, config['Figure Parameters']['regroup'])

            # then iterate thorugh each parameter group specified
            for spect_group in s:
                ncols = int(config['Figure Parameters']['n_columns'])
                fig0 = plt.figure(figsize=(20,20)) # figure of individal spectra with ref
                ax0 = fig0.subplots(nrows=math.ceil(len(spect_group)/ncols) + 1, ncols=ncols)
                subplot_tracker = [] # keeps track of the subplots that has been used
                fig1,ax1 = plt.subplots()
                for i,spect in enumerate(spect_group):
                    title_extension = ' ( '
                    dynamic_title_extension = ' ( '
                    for key,val in spect.params.items():
                        if key != 'case':
                            title_extension = title_extension + '{}={} '.format(key, val)
                            if key == config['Figure Parameters']['regroup']:
                                dynamic_title_extension = dynamic_title_extension + '{}={} '.format(key,val)
                    title_extension = title_extension + ')'
                    dynamic_title_extension = dynamic_title_extension + ')'

                    # small graph labels
                    title = config[key_string]['small_title_base'] + title_extension
                    x_axis = config[key_string]['small_x_axis']
                    y_axis = config[key_string]['small_y_axis']

                    # constructing actual subplot of small graphs
                    x_coord = math.floor(i / ncols)
                    y_coord = i % ncols
                    subplot_tracker.append(ax0[x_coord, y_coord])
                    masked_wvs = spect.wvs
                    if boolean(config[key_string]['masking']):
                        masked_wvs = self._mask_wvs(masked_wvs)
                    ax0[x_coord, y_coord].set_title(title)
                    ax0[x_coord, y_coord].set(xlabel=x_axis, ylabel=y_axis)

                    # compares with section name of uncalibrated figures in config
                    if key_string == 'Uncalibrated Figures':  
                        ax0[x_coord, y_coord].errorbar(masked_wvs, spect.uncalib_spect, yerr=spect.uncalib_error, 
                                                        label=config['Figure Parameters']['legend_label'], capsize=3, 
                                                        color=self.colors[0])
                    elif key_string == 'Calibrated Figures':
                        ax0[x_coord, y_coord].errorbar(masked_wvs, spect.calib_spect, yerr=spect.calib_error, 
                                                        label=config['Figure Parameters']['legend_label'], capsize=3, 
                                                        color=self.colors[0])
                    
                    # large graph labels
                    title = config[key_string]['large_title_base'] + dynamic_title_extension
                    x_axis = config[key_string]['large_x_axis']
                    y_axis = config[key_string]['large_x_axis']

                    # a composition of all the graphs
                    ax1.set_title(title)
                    ax1.set(xlabel=x_axis, ylabel=y_axis)
                    if key_string == 'Uncalibrated Figures':  
                        ax1.plot(masked_wvs, spect.uncalib_spect, label=title_extension)
                    elif key_string == 'Calibrated Figures':
                        ax1.plot(masked_wvs, spect.calib_spect, label=title_extension)

                    # adding reference to subplot
                    ref_path = config[key_string]['reference_path']
                    if ref_path != '':
                        if not os.path.isfile(ref_path):
                            raise Exception('Please enter a valid path to reference')
                        ref_spect = self._get_ref_spectra(ref_path)
                        if ref_spect.err_q:
                            ax0[x_coord, y_coord].errorbar(ref_spect.wvs, ref_spect.uncalib_spect, yerr=ref_spect.uncalib_error, 
                                                            label=config[key_string]['reference_label'], capsize=3, 
                                                            color='C0')
                        else:
                            ax0[x_coord, y_coord].plot(ref_spect.wvs, ref_spect.uncalib_spect,
                                                        label=config[key_string]['reference_label'],
                                                        color='C0')
                    ax0[x_coord,y_coord].legend()
                    ax1.legend(prop={'size': 6})

                # loop through all the axes in ax0 and determine if it has been used.
                # If not, delete the subplot
                for a in ax0.flatten():
                    if not(a in subplot_tracker):
                        fig0.delaxes(a)
                fig0.tight_layout()

                # export plots
                fig0.savefig(os.path.join(figure_dir,
                            '{}_small_{}.png'.format(config[key_string]['basename'],dynamic_title_extension)))
                fig1.savefig(os.path.join(figure_dir, 
                            '{}_large_{}.png'.format(config[key_string]['basename'],dynamic_title_extension)))

#f = Figures()
#f.uncalib_fig(overwrite=True)
#f.calib_fig(overwrite=True)
#f.mag_fig(overwrite=True)
#f.all_obs_fig(overwrite=True)
                