import os
import sys
import glob
import copy
import pandas as pd

sys.path.append(os.path.abspath('../'))
from pipeline.settings import config
from pipeline.helpers import boolean, remove_n_path_levels

class Data:

    def __init__(self):
        # obtain list of case dir paths
        self.root_dir = config['Paths']['root_dir']
        self.case_dirs = config['Paths']['case_dir'].split(',')

        for i,case in enumerate(self.case_dirs):
            self.case_dirs[i] = os.path.join(self.root_dir, case)

        # for each case, get all spectra associated with that case and put it
        # into a self.df
        self.df = pd.DataFrame()
        for case in self.case_dirs: # iterating through each case
            spectra_dir = os.path.join(case, 'results/spectra')
            dirs = self._get_all_subdirs(spectra_dir)
            for spectra in dirs: # iterating through each directory in a case
                if len(os.listdir(spectra)) != 0:
                    self._read_in_data(spectra)

    # return the data frame without sorting
    def get_data(self):
        return self.df
        
    # takes in an argument of what column to not sort by, and use the rest
    # to sort the dataframe
    def get_data_sorted(self, omit=''):
        forbidden = ['calibrated error', 'calibrated spectrum', 
                    'uncalibrated error', 'uncalibrated spectrum', 
                    'wvs']

        if not omit in self.df.columns and not omit == '':
            raise ValueError('The key you inputted is not a valid dataframe column.')

        forbidden.append(omit)
        sort_list = []    
        for col in self.df.columns:
            if not col in forbidden:
                sort_list.append(col)

        return self._data_sort(sort_list)

    # return rows based on dictionary where keys represent columns and values represent
    # dataframe equal values
    def get_rows(self, d):
        conditional = True
        for key,val in d.items():
            conditional &= (self.df[key] == val)

        return self.df.loc[conditional]
    
    # helper function for returning a sorted dataframe
    def _data_sort(self, columns, ascending=None):
        if ascending == None:
            ascending = [True for i in range(len(columns))]
        return self.df.sort_values(columns, ascending=ascending)

    # finds the first order child of a directory given by path
    def _get_all_subdirs(self, path):
        subdirs = [os.path.join(path,d) for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
        return subdirs

    # given a directory where the spectra and errors are located,
    # create a dictionary of the spectra attributes and relevant names
    # by parsing the directory name and reading the files in the directory. 
    def _read_in_data(self, directory):

        # the set of number of modes used for KLIP in string form
        numbasis_str = config['Klip-static']['numbasis'].split(',')

        # parse directory name to get params
        param = {'case':os.path.basename(remove_n_path_levels(directory, 3))}
        dir_in_list = os.path.basename(directory).split('_')
        for i in range(1,len(dir_in_list) - 1):
            param_in_list = dir_in_list[i].split('=')
            if param_in_list[1].isnumeric():
                param_in_list[1] = float(param_in_list[1])
            param.update({param_in_list[0]:param_in_list[1]})

        # read in data files
        files = glob.glob(os.path.join(directory, '*.csv'))
        for f in files:
            f_basename = os.path.basename(f)
            if 'calibrated' in f_basename and 'error' not in f_basename and 'nobars' in f_basename:
                calib_spect = pd.read_csv(f)
            elif 'calibrated' in f_basename and 'error' in f_basename and 'nobars' in f_basename:
                calib_error = pd.read_csv(f)
            elif 'uncalib' in f_basename and 'error' not in f_basename and 'nobars' in f_basename:
                uncalib_spect = pd.read_csv(f)
            elif 'uncalib' in f_basename and 'error' in f_basename and 'nobars' in f_basename:
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
            p.update({'KL':int(basis), 'wvs':wvs, 'uncalibrated spectrum':ucs, 
                    'uncalibrated error':uce, 'calibrated spectrum':cs, 
                    'calibrated error':ce})

            self.df = self.df.append(p, ignore_index=True) 


#d = Data()
#d.get_rows({'KL':2, 'movement':7})