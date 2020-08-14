import os
import sys
import copy
import itertools
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy.stats import anderson_ksamp, ks_2samp

sys.path.append(os.path.abspath('../'))
from pipeline.settings import config
from load_data import Data

class Anderson_Darling_Test:

    def __init__(self):
        # define root dir
        self.root_dir = config['Paths']['root_dir']
        # setup test ouput dir
        self.test_results_dir = os.path.join(self.root_dir, 'results/stat_tests/' + config['Anderson Darling Test']['dir_name'])
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)

        # istantiate a Data object
        self.data = Data()

    def perform_k_samp_test(self):
        # uncalibrated dataframe
        uncalib_df = pd.DataFrame()
        calib_df = pd.DataFrame()

        # generate file name
        uncalib_file_name = config['Anderson Darling Test']['uncalib_name_base'] + 'anderson_darling.csv'
        calib_file_name = config['Anderson Darling Test']['calib_name_base'] + 'anderson_darling.csv'

        # stores all the dynamic params and their list
        params = {}

        # set KL as a param
        KLs = list(map(int, config['Klip-static']['numbasis'].split(',')))
        params.update({'KL':KLs})
        
        # set all the dynamic variables in Klip-dynamic params
        for item in config['Klip-dynamic'].items():
            l = item[1].split(',')
            if l[0].isnumeric():
                l = list(map(float,l))
            params.update({item[0]:l})

        # separate key and values into separate lists for easier permutations
        key_list = []
        val_list = []
        for key,val in params.items():
            key_list.append(key)
            val_list.append(val)

        # get all combinations of data in the val list
        comb = list(itertools.product(*val_list))
        
        # iterate through all the combinations and perform the adt
        for tupl in comb:
            param_dict = {}
            for i,key in enumerate(key_list):
                param_dict.update({key:tupl[i]})
            
            # get the corresponding df rows for the adt
            rows = self.data.get_rows(param_dict)

            # iterate through all the rows in rows to get the list of 
            # uncalibrated spectra, calibrated spectra, and cases
            cases = []
            uncalib_list = []
            calib_list = []
            for i, row in rows.iterrows():
                cases.append(row['case'])
                uncalib_list.append(row['uncalibrated spectrum'])
                calib_list.append(row['calibrated spectrum'])

            # perform the k sample anderson darling test on both calibrated 
            # and uncalibrated spectra
            uncalb_result = anderson_ksamp(uncalib_list)
            calib_result = anderson_ksamp(calib_list)

            # generate results dataframe
            critical = ['critical value 25%', 
                        'critical value 10%',
                        'critical value 5%', 
                        'critical value 2.5%',
                        'critical value 1%',
                        'critical value 0.5%',
                        'critical value 0.1%']
            uncalib_csv_row = {}
            calib_csv_row = {}
            
            for i,key in enumerate(key_list):
                uncalib_csv_row.update({key:tupl[i]})
                calib_csv_row.update({key:tupl[i]})
            
            uncalib_csv_row.update({'statistic':uncalb_result[0]})
            calib_csv_row.update({'statistic':calib_result[0]})
            
            for i,key in enumerate(critical):
                uncalib_csv_row.update({key:uncalb_result[1][i]})
                calib_csv_row.update({key:calib_result[1][i]})
            
            uncalib_csv_row.update({'significance level':uncalb_result[2]})
            calib_csv_row.update({'significance level':calib_result[2]})

            uncalib_df = uncalib_df.append(uncalib_csv_row, ignore_index=True)
            calib_df = calib_df.append(calib_csv_row, ignore_index=True)

        # export df as csv
        uncalib_df.to_csv(os.path.join(self.test_results_dir, uncalib_file_name))
        calib_df.to_csv(os.path.join(self.test_results_dir, calib_file_name))

        return (uncalib_df, calib_df)
            



class Data_Variance:

    def __init__(self):
        # define root dir
        self.root_dir = config['Paths']['root_dir']

        # setup test ouput dir
        self.test_results_dir = os.path.join(self.root_dir, 'results/stat_tests/' + config['Data Variance']['dir_name'])
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)

        # istantiate a Data object
        self.data = Data()

    def get_variance(self, spacing, n_samples):
        # stores all the dynamic params and their list
        params = {}

        # set KL as a param
        KLs = list(map(int, config['Klip-static']['numbasis'].split(',')))
        params.update({'KL':KLs})
        
        # set all the dynamic variables in Klip-dynamic params
        for item in config['Klip-dynamic'].items():
            l = item[1].split(',')
            if l[0].isnumeric():
                l = list(map(float,l))
            params.update({item[0]:l})

        # separate key and values into separate lists for easier permutations
        key_list = []
        val_list = []
        for key,val in params.items():
            key_list.append(key)
            val_list.append(val)

        # get all combinations of data in the val list
        comb = list(itertools.product(*val_list))

        # the dataframe to be exported at the very end
        ret_df = pd.DataFrame()
        
        # iterate through all the combinations and perform the adt
        for tupl in comb:
            param_dict = {}
            for i,key in enumerate(key_list):
                param_dict.update({key:tupl[i]})
            
            # get the corresponding df rows for the adt
            rows = self.data.get_rows(param_dict)

            # iterate through all the rows in rows to get the list of 
            # uncalibrated spectra, calibrated spectra, and cases
            cases = []
            uncalib_list = []
            calib_list = []
            for i, row in rows.iterrows():
                cases.append(row['case'])
                uncalib_list.append(row['uncalibrated spectrum'])
                calib_list.append(row['calibrated spectrum'])

            # iterate through each wavelength 
            wvs = rows['wvs'].iloc[0]
            for i, wv in enumerate(wvs):
                # tranpose the array so that each 1st order list will represent all the values at 1 wavelength
                t_uncalib = np.array(uncalib_list).transpose()
                t_calib = np.array(calib_list).transpose()

                # get the proper wavelength
                arr_uncalib = t_uncalib[i]
                arr_calib = t_calib[i]

                # calculate the variance
                var_uncalib = self._get_variance(arr_uncalib, spacing, n_samples)
                var_calib = self._get_variance(arr_calib, spacing, n_samples)

                # calculate the mean
                mean_uncalib = self._get_mean(arr_uncalib, spacing, n_samples)
                mean_calib = self._get_mean(arr_calib, spacing, n_samples)

                # get the histogram
                self._get_histogram(arr_calib, param_dict, wv)
            
                # make dictinary to represent new df row
                ret_row = copy.deepcopy(param_dict)
                ret_row.update({'wv':wv, 'var uncalib':var_uncalib, 'var calib': var_calib, 'mean uncalib':mean_uncalib, 'mean calib':mean_calib})

                # apped new row to dataframe
                ret_df = ret_df.append(ret_row, ignore_index=True)

        file_name = 'spacing=' + str(spacing) + '_variance.csv'
        ret_df.to_csv(os.path.join(self.test_results_dir, file_name))
        
        return ret_df

        

    # helper function to get the variance of elements in a list
    def _get_variance(self, arr, spacing, n_samples):
        if spacing < 1:
            raise Exception('Spacing must be a positive integer greater than 0.')

        if spacing * n_samples > len(arr):
            raise Exception('The array you gave is not large enough to accomodate the spacing.')

        # calculate elements to find variance for given an index spacing 
        iarr = list(range(0, spacing * n_samples, spacing))
        arr_want = [arr[i] for i in iarr]

        # find the variance of the elements
        var = statistics.variance(arr_want)
        return var

    # helper function to get the mean of elements in a list
    def _get_mean(self, arr, spacing, n_samples):
        if spacing < 1:
            raise Exception('Spacing must be a positive integer greater than 0.')

        if spacing * n_samples > len(arr):
            raise Exception('The array you gave is not large enough to accomodate the spacing.')

        # calculate elements to find variance for given an index spacing 
        iarr = list(range(0, spacing * n_samples, spacing))
        arr_want = [arr[i] for i in iarr]

        # find the variance of the elements
        mean = statistics.mean(arr_want)
        return mean

    def _get_histogram(self, arr, params, wv):
        # calculate elements to find variance for given an index spacing 
        arr_want = arr

        # clear plot
        plt.clf()
        
        # generate histogram
        plt.hist(arr_want, bins=int(config['Data Variance']['n_bins']))

        # generate the file name
        basename = 'histogram'
        descriptor = ''
        for key,val in params.items():
            descriptor += '_{}={}'.format(key, val)
        descriptor += '_wv={}'.format(wv)
        extension = '.png'
        file_name = basename + descriptor + extension

        print(os.getcwd())
        # save the figure
        plt.savefig(os.path.join(self.test_results_dir, file_name))






        

        
#adt = Anderson_Darling_Test()
#adt.perform_k_samp_test()

var = Data_Variance()
for i in range(1,7):
    var.get_variance(i, 5)
        


