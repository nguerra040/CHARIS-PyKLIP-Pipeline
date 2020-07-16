import os
import glob
import math
import copy
import pandas as pd
import sys, os.path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from settings import config

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
        self.case_dirs = config['Paths']['case_dir'].split(',')
        self.figures_dirs = []
        self.case_to_figure_dir = {}
        for i,case in enumerate(self.case_dirs):
            self.figures_dirs.append(os.path.join(case, 'results/figures'))
            if not os.path.isabs(case):
                self.case_dirs[i] = os.path.join(self.root_dir,case)
                self.figures_dirs[i] = os.path.join(self.root_dir,self.figures_dirs[i])
                self.case_to_figure_dir.update({os.path.basename(case):self.figures_dirs[i]})

            
        # check if figure directory exists, if not, create it
        for figures_dir in self.figures_dirs:
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)

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
    def uncalib_fig(self):
        # iterate through all the cases
        for casename, spectra in self.cases.items():
            figure_dir = self.case_to_figure_dir[casename]
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
                        title_extension = title_extension + '{}={} '.format(key, val)
                        if key == config['Figure Parameters']['regroup']:
                            dynamic_title_extension = dynamic_title_extension + '{}={}'.format(key,val)
                    title_extension = title_extension + ')'
                    dynamic_title_extension = dynamic_title_extension + ')'

                    # small graph labels
                    title = config['Uncalibrated Figures']['small_title_base'] + title_extension
                    x_axis = config['Uncalibrated Figures']['small_x_axis']
                    y_axis = config['Uncalibrated Figures']['small_x_axis']

                    # constructing actual subplot of small graphs
                    x_coord = math.floor(i / ncols)
                    y_coord = i % ncols
                    subplot_tracker.append(ax0[x_coord, y_coord])
                    ax0[x_coord, y_coord].set_title(title)
                    ax0[x_coord, y_coord].set(xlabel=x_axis, ylabel=y_axis)
                    ax0[x_coord, y_coord].errorbar(spect.wvs, spect.uncalib_spect, yerr=spect.uncalib_error, 
                                                    label=config['Figure Parameters']['legend_label'], capsize=3, 
                                                    color=self.colors[0])
                    
                    # large graph labels
                    title = config['Uncalibrated Figures']['large_title_base'] + dynamic_title_extension
                    x_axis = config['Uncalibrated Figures']['large_x_axis']
                    y_axis = config['Uncalibrated Figures']['large_x_axis']

                    # a composition of all the graphs
                    ax1.set_title(title)
                    ax1.set(xlabel=x_axis, ylabel=y_axis)
                    ax1.plot(spect.wvs, spect.uncalib_spect, label=title_extension)

                    # adding reference to subplot
                    ref_path = config['Uncalibrated Figures']['reference_path']
                    if ref_path != '':
                        if not os.path.isfile(ref_path):
                            raise Exception('Please enter a valid path to reference')
                        ref_spect = self._get_ref_spectra(ref_path)
                        if ref_spect.err_q:
                            ax0[x_coord, y_coord].errorbar(ref_spect.wvs, ref_spect.uncalib_spect, yerr=ref_spect.uncalib_error, 
                                                            label=config['Uncalibrated Figures']['reference_label'], capsize=3, 
                                                            color='C0')
                        else:
                            ax0[x_coord, y_coord].plot(ref_spect.wvs, ref_spect.uncalib_spect,
                                                        label=config['Uncalibrated Figures']['reference_label'],
                                                        color='C0')
                    ax0[x_coord,y_coord].legend()
                    ax1.legend(prop={'size': 6})
                # loop through all the axes in ax0 and determine if it has been used.
                # If not, delete the subplot
                for a in ax0.flatten():
                    if not(a in subplot_tracker):
                        fig0.delaxes(a)
                plt.tight_layout()

                # export plots
                fig0.savefig(os.path.join(figure_dir,
                            '{}_small_{}.png'.format(config['Uncalibrated Figures']['basename'],dynamic_title_extension)))
                fig1.savefig(os.path.join(figure_dir, 
                            '{}_large_{}.png'.format(config['Uncalibrated Figures']['basename'],dynamic_title_extension)))
            

    # create figures for spectra that is calibrated. The output figures are 
    # then stored in their respective case directory
    def calib_fig(self):
        # iterate through all the cases
        for casename, spectra in self.cases.items():
            figure_dir = self.case_to_figure_dir[casename]
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
                        title_extension = title_extension + '{}={} '.format(key, val)
                        if key == config['Figure Parameters']['regroup']:
                            dynamic_title_extension = dynamic_title_extension + '{}={}'.format(key,val)
                    title_extension = title_extension + ')'
                    dynamic_title_extension = dynamic_title_extension + ')'

                    # small graph labels
                    title = config['Calibrated Figures']['small_title_base'] + title_extension
                    x_axis = config['Calibrated Figures']['small_x_axis']
                    y_axis = config['Calibrated Figures']['small_x_axis']

                    # constructing actual subplot of small graphs
                    x_coord = math.floor(i / ncols)
                    y_coord = i % ncols
                    subplot_tracker.append(ax0[x_coord, y_coord])
                    ax0[x_coord, y_coord].set_title(title)
                    ax0[x_coord, y_coord].set(xlabel=x_axis, ylabel=y_axis)
                    ax0[x_coord, y_coord].errorbar(spect.wvs, spect.calib_spect, yerr=spect.calib_error, 
                                                    label=config['Figure Parameters']['legend_label'], capsize=3, 
                                                    color=self.colors[0])
                    
                    # large graph labels
                    title = config['Calibrated Figures']['large_title_base'] + dynamic_title_extension
                    x_axis = config['Calibrated Figures']['large_x_axis']
                    y_axis = config['Calibrated Figures']['large_x_axis']

                    # a composition of all the graphs
                    ax1.set_title(title)
                    ax1.set(xlabel=x_axis, ylabel=y_axis)
                    ax1.plot(spect.wvs, spect.calib_spect, label=title_extension)

                    # adding reference to subplot
                    ref_path = config['Calibrated Figures']['reference_path']
                    if ref_path != '':
                        if not os.path.isfile(ref_path):
                            raise Exception('Please enter a valid path to reference')
                        ref_spect = self._get_ref_spectra(ref_path)
                        if ref_spect.err_q:
                            ax0[x_coord, y_coord].errorbar(ref_spect.wvs, ref_spect.uncalib_spect, yerr=ref_spect.uncalib_error, 
                                                            label=config['Calibrated Figures']['reference_label'], capsize=3, 
                                                            color='C0')
                        else:
                            ax0[x_coord, y_coord].plot(ref_spect.wvs, ref_spect.uncalib_spect,
                                                        label=config['Calibrated Figures']['reference_label'],
                                                        color='C0')
                    ax0[x_coord,y_coord].legend()
                    ax1.legend(prop={'size': 6})

                # loop through all the axes in ax0 and determine if it has been used.
                # If not, delete the subplot
                for a in ax0.flatten():
                    if not(a in subplot_tracker):
                        fig0.delaxes(a)
                plt.tight_layout()

                # export plots
                fig0.savefig(os.path.join(figure_dir,
                            '{}_small_{}.png'.format(config['Calibrated Figures']['basename'],dynamic_title_extension)))
                fig1.savefig(os.path.join(figure_dir, 
                            '{}_large_{}.png'.format(config['Calibrated Figures']['basename'],dynamic_title_extension)))

    # create figures for the difference in magnitude. The output figures are 
    # then stored in their respective case directory
    def mag_fig(self):
        pass

    # create figure that combines the calibrated spectra across different
    # cases
    def all_obs_fig(self):
        pass
    









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
        param = {}
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
            err = arr_df[3]
            spectrum = Spectrum(wvs=wvs, uncalib_spect=val, uncalib_error=err)
            spectrum.err_q = True
        except:
            spectrum = Spectrum(wvs=wvs, uncalib_spect=val)
            spectrum.err_q = False
        
        return spectrum

f = Figures()
f.uncalib_fig()
#f.calib_fig()