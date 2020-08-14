import os
import glob
import copy
import subprocess
import numpy as np
import pandas as pd
import configparser
from uncertainties import ufloat
from scipy import stats as spstat
from scipy import interpolate

import pyklip.fm as fm
import pyklip.fakes as fakes
import pyklip.fmlib.extractSpec as es
import pyklip.parallelized as parallelized
from pyklip.instruments import CHARIS as charis

from .settings import config
from .helpers import boolean, get_bash_path, get_star_spot_ratio

# Set the value of the system environment variable $PYSTN_CDBS
# to be the location of the pysynphot cdbs directory.
cdbs_path = config['Paths']['cdbs_dir']
os.environ['PYSYN_CDBS'] = cdbs_path
import pysynphot as S

# A KLIP object is tasked with completing stage 2 of the 
# pipeline, where the extracted cubes from stage 1 are fed 
# into pyKLIP to extract the spectrum of the planet using 
# KLIP-FM. Each KLIP object only operates on a single case 
# and therefore could easily be parallelized in the future 
# for multiple cases.
class KLIP:
    # Initialize directory variables and create new dirrectories if needed
    def __init__(self, case_dir, params):
        postfix = ''
        for param in params:
            postfix = postfix + '_' + param + '=' + str(params[param])
        postfix = postfix + '_' + config['Output']['postfix']

        self.parameters = params
        for item in config.items('Klip-static'):
            self.parameters.update({item[0]:item[1]})

        self.case_dir = case_dir
        self.cubes_dir = os.path.join(case_dir, 'extracted/cubes')
        self.spectra_dir = os.path.join(case_dir, 'results/spectra/spectra' + postfix)
        self.fakes_dir = os.path.join(case_dir, 'klip/fakes/fakes' + postfix)
        self.klipped_fm_spectra_dir = os.path.join(case_dir, 'klip/klipped/fm_spectra/fm_spectra' + postfix)
        self.klipped_general_dir = os.path.join(case_dir, 'klip/klipped/general/general' + postfix)

        if not os.path.exists(self.case_dir):
            raise Exception('Case directory does not exist!')
        if not os.path.exists(self.cubes_dir):
            raise Exception('Reduced cubes directory does not exist!')
        if not os.path.exists(self.spectra_dir):
            os.makedirs(self.spectra_dir)
        if not os.path.exists(self.fakes_dir):
            os.makedirs(self.fakes_dir)
        if not os.path.exists(self.klipped_fm_spectra_dir):
            os.makedirs(self.klipped_fm_spectra_dir)
        if not os.path.exists(self.klipped_general_dir):
            os.makedirs(self.klipped_general_dir)



    # A helper function for self.klip_general and self.klip_fm_spect
    # that reads in a dataset and creates a PSF cube for that dataset.
    def _intgest_data(self):
        # obtain all image files
        image_files = glob.glob(os.path.join(self.cubes_dir,"*.fits"))
        image_files.sort()
        self.data = image_files
        if config['Readin']['skipslices'] == '':
            self.skipslices = []
        else:
            self.skipslices = list(map(int, config['Readin']['skipslices'].split(",")))

        # set the guessing spot location and index if specified
        guess_spot_loc = config['Readin']['guess_spot_loc']
        if guess_spot_loc == '':
            guess_spot_loc = None
            guess_spot_index = 0
        else:
            guess_spot_loc = guess_spot_loc.split('|')
            for i,loc in enumerate(guess_spot_loc):
                guess_spot_loc[i] = list(map(float,loc.split(',')))
            guess_spot_index = int(config['Readin']['guess_spot_index'])

        # read in image files as a charis object
        self.dataset = charis.CHARISData(self.data,
                                        skipslices=self.skipslices, 
                                        update_hdrs=boolean(config['Readin']['updatehdrs']),
                                        guess_spot_locs=guess_spot_loc,
                                        guess_spot_index=guess_spot_index)
        # generate a PSF cube for that dataset from satellite spots
        self.dataset.generate_psfs(int(self.parameters['boxrad']))
        self.PSF_cube = self.dataset.psfs

        # commonly used variables
        self.N_frames = len(self.dataset.input)
        self.N_cubes = np.size(np.unique(self.dataset.filenums))
        self.nl = self.N_frames // self.N_cubes
        self.numbasis = list(map(int, self.parameters['numbasis'].split(',')))
        self.planet_sep, self.planet_pa = list(map(float, self.parameters['pars'].split(',')))
        self.stamp_size = int(self.parameters['stampsize'])



    # Perform KLIP via pyKLIP on the dataset, used to produce 
    # a processed image so user can keep determine the planet's
    # parallactic angle and separation.
    def klip_general(self, overwrite=False):
        dir_contents = os.listdir(self.klipped_general_dir)
        if overwrite or len(dir_contents) == 0:
            self._intgest_data()
            parallelized.klip_dataset(self.dataset, 
                                        outputdir=self.klipped_general_dir, 
                                        fileprefix=self.parameters['fileprefix'],
                                        movement=int(self.parameters['movement']), 
                                        annuli=int(self.parameters['annuli']), 
                                        subsections=int(self.parameters['subsections']),
                                        annuli_spacing=self.parameters['annuli_spacing'], 
                                        numbasis=self.numbasis, 
                                        maxnumbasis=int(self.parameters['maxnumbasis']), 
                                        mode=self.parameters['mode'], 
                                        highpass=boolean(self.parameters['highpass']))

    # Perform KLIP-FM via pyKLIP on the dataset, used to produce
    # the spectrum. The method first extracts an intial spectra. 
    # then using the initial spectra, inject fake planets to obtain
    # the error bars. Following that, perform algorithmic calibration
    # and spectral calibration. Finally, export the spectra as a csv file.
    def klip_fm_spect(self, overwrite=False):
        klipped_fm_spectra_dir_contents = os.listdir(self.klipped_fm_spectra_dir)
        spect_dir_contents = os.listdir(self.spectra_dir)

        if overwrite or (len(klipped_fm_spectra_dir_contents) == 0 and len(spect_dir_contents) == 0):
            self._intgest_data()

            # forward modeling class
            self.fm_class = es.ExtractSpec(self.dataset.input.shape,
                                        numbasis=self.numbasis,
                                        sep=self.planet_sep,
                                        pa=self.planet_pa,
                                        input_psfs=self.PSF_cube,
                                        input_psfs_wvs=np.unique(self.dataset.wvs),
                                        stamp_size = self.stamp_size)

            if self.parameters['spectrum'] == '':
                spectrum = None
            else:
                spectrum = self.parameters['spectrum']
            
            # perform KLIP using pyKLIP
            fm.klip_dataset(self.dataset, self.fm_class,
                            fileprefix=self.parameters['fileprefix'],
                            annuli=[[self.planet_sep - self.stamp_size,self.planet_sep + self.stamp_size]],
                            subsections=[[(self.planet_pa - self.stamp_size)/180.*np.pi,\
                                        (self.planet_pa + self.stamp_size)/180.*np.pi]],
                            movement=int(self.parameters['movement']), 
                            numbasis=self.numbasis,
                            spectrum=spectrum,
                            save_klipped=boolean(self.parameters['saveklipped']), 
                            highpass=boolean(self.parameters['highpass']),
                            outputdir=self.klipped_fm_spectra_dir)

            # interpolate the nan values in the forward model
            print('################## NAN',np.argwhere(np.isnan(self.dataset.fmout[:,:,-1,:])))
            self._interpolate_fm(self.dataset)

            # invert the FM to get the spectra
            self.exspect, self.fm_matrix = es.invert_spect_fmodel(self.dataset.fmout, 
                                                                self.dataset, 
                                                                units=self.parameters['units'],
                                                                scaling_factor=self.parameters['scalefactor'],
                                                                method=self.parameters['reverse_method'])
            
            # calculate the fake planet spectra
            fake_spectra_all_bases = self._calc_error_bars()
            self.exspect_error = []
            self.fake_mean = []
            for i in range(len(fake_spectra_all_bases)):
                for j in range(self.nl):
                    x = fake_spectra_all_bases[i][:,j]
                    err = spstat.iqr(x)
                    m = ufloat(np.mean(x), np.std(x))
                    self.exspect_error.append(err)
                    self.fake_mean.append(m)

            # error bars
            self.exspect_error = np.array(self.exspect_error).reshape(int(len(self.exspect_error)/self.nl),self.nl)

            # algorithmic bias correction factor
            self.fake_mean = np.array(self.fake_mean).reshape(int(len(self.fake_mean)/self.nl),self.nl)

            #convert to ufloat
            self.exspect_ufloat = np.ndarray(shape=np.shape(self.exspect), dtype=object)
            for i in range(len(self.exspect)):
                for j in range(self.nl):
                    self.exspect_ufloat[i][j] = ufloat(self.exspect[i,j],self.exspect_error[i,j])

            # perform algorithmic calibration
            if boolean(config['Calibration']['algo_calibrate']):
                self.exspect_ufloat = np.add(self.exspect_ufloat,(np.subtract(self.exspect_ufloat,self.fake_mean)))

            # export spectra as csv
            self._export_csv_dataset(self.exspect_ufloat, 'uncalib')

            # perform spectral calibration 
            if boolean(config['Calibration']['spect_calibrate']):
                if boolean(config['Calibration']['icat']):

                    r_star_val, r_star_err = list(map(float, config['Icat']['R_star'].split(',')))
                    R_star = ufloat(r_star_val, r_star_err)
                    d_star_val, d_star_err = list(map(float, config['Icat']['D_star'].split(',')))
                    D_star = ufloat(d_star_val, d_star_err)

                    # satellite spot to star ratio
                    spot_to_star_ratio = self._get_satellite_spots()

                    # star spectrum
                    sp = S.Icat(config['Icat']['model_name'], float(config['Icat']['eff_temp']), 
                                float(config['Icat']['metal']), float(config['Icat']['logg']))
                    sp.convert(config['Icat']['x-axis'])
                    sp.convert(config['Icat']['y-axis'])
                    star_spectrum = np.array([sp.sample(self.dataset.wvs[i]) for i in range(self.nl)]) * (R_star / D_star)**2

                    # perform actual calibration
                    self.exspect_ufloat_calibrated = self.exspect_ufloat * star_spectrum * spot_to_star_ratio

                    # export data
                    self._export_csv_dataset(self.exspect_ufloat_calibrated, 'calibrated')
                else:
                    # from https://scholar.princeton.edu/charis/capabilities
                    j_mag_upper = 1.328
                    h_mag_upper = 1.783
                    k_mag_upper = 2.375

                    # create
                    star_mag = np.array([])
                    for wv in self.dataset.wvs[0:self.nl]:
                        if wv < j_mag_upper:
                            val, err = list(map(float, config['Non-Icat']['J_band'].split(',')))
                            star_mag = np.append(star_mag, ufloat(val, err))
                        elif wv < h_mag_upper:
                            val, err = list(map(float, config['Non-Icat']['H_band'].split(',')))
                            star_mag = np.append(star_mag, ufloat(val, err))
                        elif wv < k_mag_upper:
                            val, err = list(map(float, config['Non-Icat']['K_band'].split(',')))
                            star_mag = np.append(star_mag, ufloat(val, err))

                    # satellite spot to star ratio
                    spot_to_star_ratio = self._get_satellite_spots()

                    #star spectrum
                    sp = S.FileSpectrum(config['Non-Icat']['filename'])
                    sp.convert(config['Non-Icat']['x-axis'])
                    sp.convert(config['Non-Icat']['y-axis'])
                    star_spectrum = np.array([sp.sample(self.dataset.wvs[i]) for i in range(self.nl)])
                    star_spectrum=star_spectrum * 10**((star_mag - 0)/-2.5)

                    #perform actual calibration
                    self.exspect_ufloat_calibrated = self.exspect_ufloat * star_spectrum * spot_to_star_ratio

                    #export spectra data as csv file
                    self._export_csv_dataset(self.exspect_ufloat_calibrated, 'calibrated')            


    # Helper function for self.klip_fm_spect. Function injects fake planets 
    # into the dataset and then forward models their spectra. These spectra 
    # is returned and is used to determine the error bars as well as to correct
    # for algorithmic bias.
    def _calc_error_bars(self):
        if self.parameters['spectrum'] == '':
            spectrum = None
        else:
            spectrum = self.parameters['spectrum']


        def fake_spect(pa, fake_flux, basis):
            fakepsf = []
            psfs = np.tile(self.PSF_cube, (self.N_cubes, 1, 1))
            for i in range(len(psfs)):
                spec = fake_flux[i % 22] * psfs[i]
                fakepsf.append(spec)
            
            tempdataset = charis.CHARISData(self.data)
            fakes.inject_planet(tempdataset.input, tempdataset.centers, fakepsf, tempdataset.wcs, self.planet_sep, pa)
            
            fm_class = es.ExtractSpec(tempdataset.input.shape,
                                        numbasis=basis,
                                        sep=self.planet_sep,
                                        pa=pa,
                                        input_psfs=self.PSF_cube,
                                        input_psfs_wvs=np.unique(self.dataset.wvs),
                                        stamp_size=self.stamp_size)
            
            fm.klip_dataset(tempdataset, fm_class,
                        fileprefix="fmspect_"+"pa="+str(pa),
                        annuli=[[self.planet_sep - self.stamp_size, self.planet_sep + self.stamp_size]],
                        subsections=[[(pa - self.stamp_size)/180.*np.pi,\
                                    (pa + self.stamp_size)/180.*np.pi]],
                        movement=int(self.parameters['movement']),
                        numbasis=basis,
                        spectrum=spectrum,
                        save_klipped=boolean(self.parameters['saveklipped']), 
                        highpass=boolean(self.parameters['highpass']),
                        outputdir=os.path.join(self.fakes_dir, str(basis)))
            
            # interpolate the nan values in the forward model
            print('################## NAN',np.argwhere(np.isnan(tempdataset.fmout[:,:,-1,:])))
            self._interpolate_fm(tempdataset)

            exspect_fake, fm_matrix_fake = es.invert_spect_fmodel(tempdataset.fmout, 
                                                                tempdataset, 
                                                                units=self.parameters['units'],
                                                                scaling_factor=self.parameters['scalefactor'],
                                                                method=self.parameters['reverse_method'])
            
            del tempdataset
            return exspect_fake


        nplanets = int(config['Errorbars']['nplanets'])

        # same separation as the real planet, equal radial spacing between 
        # the n fake planets
        range_angle = float(config['Errorbars']['range_angle'])
        positive = (np.linspace(self.planet_pa, (self.planet_pa + range_angle), num=nplanets+1) % 360)[1:]
        negative = (np.linspace(self.planet_pa, (self.planet_pa - range_angle), num=nplanets+1) % 360)[1:]
        pas = np.sort(np.unique(np.concatenate([positive,negative])))
    
        #pas = (np.linspace(self.planet_pa, self.planet_pa+360, num=nplanets+2)%360)[1:-1] 
        fake_spectra_all_bases = []

        # iterate through all the requested modes
        for i in range(len(self.numbasis)): 
            input_spect = self.exspect[i,:]
            fake_spectra = np.zeros((len(pas), self.nl))

            # iterate through all the requested planets
            for p, pa in enumerate(pas): 
                basis_dir = os.path.join(self.fakes_dir, str(self.numbasis[i]))
                if not os.path.exists(basis_dir):
                    os.makedirs(basis_dir)
                fake_spectra[p,:] = fake_spect(pa, input_spect, self.numbasis[i])
                percent = ((p + len(pas)*i)/(len(self.numbasis) * len(pas))) * 100
                print(str(percent) + "%")
            fake_spectra_all_bases.append(fake_spectra)

        return fake_spectra_all_bases

    # Helper function for self.klip_fm_spectm, used to 
    # calculate the satellite spot to star ratio with 
    # the function get_star_spot_ratio in helpers.py.
    def _get_satellite_spots(self):
        # determine satellite spot to star ratio
        mod_list = [header['X_GRDAMP'] for header in self.dataset.prihdrs]
        mjd_list = [header['MJD'] for header in self.dataset.prihdrs]

        # check if all the modulations are the same 
        if len(mod_list) == 0 or mod_list.count(mod_list[0]) != len(mod_list): 
            raise Exception('There is no modulation amplitude or there are multiple')
        if max(mjd_list) - min(mjd_list) > 1:
            raise Exception('Cubes are not from the same observation night')
        mod = mod_list[0] * 1000 # to convert from um to nm
        wvs = self.dataset.wvs[0:self.nl]
        mjd = mjd_list[0]

        # determine if user wants to input a manual attenuation factor
        if boolean(config['Calibration']['manual_attenutation_factor']) == None: 
            manual = None
        else:
            try:
                manual = float(config['Calibration']['manual_attenutation_factor'])
            except:
                raise Exception('manual_attenuation_factor must be a float type object')
        spot_to_star_ratio = get_star_spot_ratio(mod, wvs, mjd, manual=manual)
        
        return spot_to_star_ratio

    # Helper function for self.klip_fm_spect, 
    # takes in a spectra and a file prefix and 
    # exports the spectra as a csv file.
    def _export_csv_dataset(self, spect, prefix):        
        d = {"wvs": self.dataset.wvs[:self.nl]}
        for i in range(len(self.numbasis)):
            d[self.numbasis[i]] = [val.nominal_value for val in spect[i]]

        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.spectra_dir, prefix + "_spectra.csv"))

        d = {"wvs": self.dataset.wvs[:self.nl]}
        for i in range(len(self.numbasis)):
            d[self.numbasis[i]] = [val.std_dev for val in spect[i]]

        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.spectra_dir, prefix + "_spectra_error.csv"))


    # interplates the nan values so that the fm won't have nans
    def _interpolate_fm(self, dataset):
        klipped = dataset.fmout[:,:,-1,:]
        hoz = klipped.shape[2]
        nan_locs = np.argwhere(np.isnan(klipped))
        if nan_locs.size == 0:
            return None
        for loc in nan_locs:
            if not np.isnan(klipped[tuple(loc)]):
                continue

            loc_left = copy.deepcopy(loc)
            loc_right = copy.deepcopy(loc)
            
            while np.isnan(klipped[tuple(loc_left)]) and loc_left[2] > 0:
                loc_left[2] -= 1

            while np.isnan(klipped[tuple(loc_right)]) and loc_right[2] < hoz - 1:
                loc_right[2] += 1
            
            print(loc_left, loc_right)

            if np.isnan(klipped[tuple(loc_right)]) and np.isnan(klipped[tuple(loc_left)]):
                raise Exception("Please take a look at a particular slice {}".format(loc_left[1]))
            elif np.isnan(klipped[tuple(loc_right)]) and not np.isnan(klipped[tuple(loc_left)]):
                for i in range(loc_left[2] + 1, hoz):
                    klipped[(loc[0],loc[1],i)] = klipped[tuple(loc_left)]
            elif not np.isnan(klipped[tuple(loc_right)]) and np.isnan(klipped[tuple(loc_left)]):
                for i in range(loc_right[2]):
                    klipped[(loc[0],loc[1],i)] = klipped[tuple(loc_right)]
            else:
                x = [loc_left[2], loc_right[2]]
                y = [klipped[tuple(loc_left)], klipped[tuple(loc_right)]]
                f = interpolate.interp1d(x, y)
                for i in range(loc_left[2]+1, loc_right[2]):
                    klipped[(loc[0],loc[1],i)] = f(i)

        return klipped