import os
import glob
import subprocess
import numpy as np
import pandas as pd
import configparser
from uncertainties import ufloat
from scipy import stats as spstat

import pyklip.fm as fm
import pyklip.fakes as fakes
import pyklip.fmlib.extractSpec as es
import pyklip.parallelized as parallelized
from pyklip.instruments import CHARIS as charis

from .settings import config, get_star_spot_ratio
from .helpers import boolean, get_bash_path

cdbs_path = config['Paths']['cdbs_dir']
bash_command = 'export PYSYN_CDBS={}'.format(get_bash_path(cdbs_path))
output = os.system(bash_command)
print("output: ", output)
import pysynphot as S

class KLIP:
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

    def _intgest_data(self):
        image_files = glob.glob(os.path.join(self.cubes_dir,"*.fits"))
        image_files.sort()
        self.data = image_files
        if config['Readin']['skipslices'] == '':
            self.skipslices = []
        else:
            self.skipslices = list(map(int, config['Readin']['skipslices'].split(",")))
        self.dataset = charis.CHARISData(self.data,
                                        skipslices=self.skipslices, 
                                        update_hdrs=boolean(config['Readin']['updatehdrs']))
        self.dataset.generate_psfs(int(self.parameters['boxrad']))
        self.PSF_cube = self.dataset.psfs




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
                                        numbasis=list(map(int, self.parameters['numbasis'].split(','))), 
                                        maxnumbasis=int(self.parameters['maxnumbasis']), 
                                        mode=self.parameters['mode'], 
                                        highpass=boolean(self.parameters['highpass']))




    def klip_fm_spect(self, overwrite=False):
        klipped_fm_spectra_dir_contents = os.listdir(self.klipped_fm_spectra_dir)
        spect_dir_contents = os.listdir(self.spectra_dir)

        if overwrite or (len(klipped_fm_spectra_dir_contents) == 0 and len(spect_dir_contents) == 0):
            self._intgest_data()

            N_frames = len(self.dataset.input)
            N_cubes = np.size(np.unique(self.dataset.filenums))
            nl = N_frames // N_cubes
            planet_sep, planet_pa = list(map(float, self.parameters['pars'].split(',')))
            numbasis = list(map(int, self.parameters['numbasis'].split(',')))
            stamp_size = int(self.parameters['stampsize'])

            self.fm_class = es.ExtractSpec(self.dataset.input.shape,
                                        numbasis=numbasis,
                                        sep=planet_sep,
                                        pa=planet_pa,
                                        input_psfs=self.PSF_cube,
                                        input_psfs_wvs=np.unique(self.dataset.wvs),
                                        stamp_size = stamp_size)

            if self.parameters['spectrum'] == '':
                spectrum = None
            else:
                spectrum = self.parameters['spectrum']
            fm.klip_dataset(self.dataset, self.fm_class,
                            fileprefix=self.parameters['fileprefix'],
                            annuli=[[planet_sep-stamp_size,planet_sep+stamp_size]],
                            subsections=[[(planet_pa-stamp_size)/180.*np.pi,\
                                        (planet_pa+stamp_size)/180.*np.pi]],
                            movement=int(self.parameters['movement']), 
                            numbasis=numbasis,
                            spectrum=spectrum,
                            save_klipped=boolean(self.parameters['saveklipped']), 
                            highpass=boolean(self.parameters['highpass']),
                            outputdir=self.klipped_fm_spectra_dir)

            self.exspect, self.fm_matrix = es.invert_spect_fmodel(self.dataset.fmout, 
                                                                self.dataset, 
                                                                units=self.parameters['units'],
                                                                scaling_factor=self.parameters['scalefactor'],
                                                                method=self.parameters['reverse_method'])
            fake_spectra_all_bases = self._calc_error_bars()
            self.exspect_error = []
            self.fake_mean = []
            for i in range(len(fake_spectra_all_bases)):
                for j in range(nl):
                    x = fake_spectra_all_bases[i][:,j]
                    err = spstat.iqr(x)
                    m = ufloat(np.mean(x), np.std(x))
                    self.exspect_error.append(err)
                    self.fake_mean.append(m)
            self.exspect_error = np.array(self.exspect_error).reshape(int(len(self.exspect_error)/nl),nl)
            self.fake_mean = np.array(self.fake_mean).reshape(int(len(self.fake_mean)/nl),nl)

            #convert to ufloat
            self.exspect_ufloat = np.ndarray(shape=np.shape(self.exspect), dtype=object)
            for i in range(len(self.exspect)):
                for j in range(nl):
                    self.exspect_ufloat[i][j] = ufloat(self.exspect[i,j],self.exspect_error[i,j])

            # perform algorithmic calibration
            if boolean(config['Calibration']['algo_calibrate']):
                self.exspect_ufloat = np.add(self.exspect_ufloat,(np.subtract(self.exspect_ufloat,self.fake_mean)))

            self._export_csv_dataset(self.exspect_ufloat, 'uncalib')

            if boolean(config['Calibration']['spect_calibrate']):
                if boolean(config['Calibration']['icat']):

                    r_star_val, r_star_err = list(map(float, config['Icat']['R_star'].split(',')))
                    R_star = ufloat(r_star_val, r_star_err)
                    d_star_val, d_star_err = list(map(float, config['Icat']['D_star'].split(',')))
                    D_star = ufloat(d_star_val, d_star_err)

                    # determine satellite spot to star ratio
                    mod_list = [header['X_GRDAMP'] for header in self.dataset.prihdrs]
                    mjd_list = [header['MJD'] for header in self.dataset.prihdrs]
                    if len(mod_list) == 0 or mod_list.count(mod_list[0]) == len(mod_list): # check if all the modulations are the same 
                        raise Exception('There is no modulation amplitude or there are multiple')
                    if max(mjd_list) - min(mjd_list) > 1:
                        raise Exception('Cubes are not from the same observation night')
                    mod = mod_list[0]
                    wvs = self.dataset.wvs[0:nl]
                    mjd = mjd_list[0]
                    spot_to_star_ratio = get_star_spot_ratio(mod, wvs, mjd)

                    # star spectrum
                    sp = S.Icat(config['Icat']['model_name'], float(config['Icat']['eff_temp']), 
                                float(config['Icat']['metal']), float(config['Icat']['logg']))
                    sp.convert(config['Icat']['x-axis'])
                    sp.convert(config['Icat']['y-axis'])
                    star_spectrum = np.array([sp.sample(self.dataset.wvs[i]) for i in range(nl)]) * (R_star / D_star)**2

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
                    for wv in self.dataset.wvs[0:nl]:
                        if wv < j_mag_upper:
                            val, err = list(map(float, config['Non-Icat']['J_band'].split(',')))
                            star_mag = np.append(star_mag, ufloat(val, err))
                        elif wv < h_mag_upper:
                            val, err = list(map(float, config['Non-Icat']['H_band'].split(',')))
                            star_mag = np.append(star_mag, ufloat(val, err))
                        elif wv < k_mag_upper:
                            val, err = list(map(float, config['Non-Icat']['K_band'].split(',')))
                            star_mag = np.append(star_mag, ufloat(val, err))

                    #satellite spot to star ratio
                    mod_list = [header['X_GRDAMP'] for header in self.dataset.prihdrs]
                    mjd_list = [header['MJD'] for header in self.dataset.prihdrs]
                    if len(mod_list) == 0 or mod_list.count(mod_list[0]) == len(mod_list): # check if all the modulations are the same 
                        raise Exception('There is no modulation amplitude or there are multiple')
                    if max(mjd_list) - min(mjd_list) > 1:
                        raise Exception('Cubes are not from the same observation night')
                    mod = mod_list[0]
                    wvs = self.dataset.wvs[0:nl]
                    mjd = mjd_list[0]
                    spot_to_star_ratio = get_star_spot_ratio(mod, wvs, mjd)

                    #star spectrum
                    sp = S.FileSpectrum(config['Non-Icat']['filename'])
                    sp.convert(config['Non-Icat']['x-axis'])
                    sp.convert(config['Non-Icat']['y-axis'])
                    star_spectrum = np.array([sp.sample(self.dataset.wvs[i]) for i in range(nl)])
                    star_spectrum=star_spectrum * 10**((star_mag - 0)/-2.5)

                    #perform actual calibration
                    self.exspect_ufloat_calibrated = self.exspect_ufloat * star_spectrum * spot_to_star_ratio

                    #export data
                    self._export_csv_dataset(self.exspect_ufloat_calibrated, 'calibrated')





    def _calc_error_bars(self):
        N_frames = len(self.dataset.input)
        N_cubes = np.size(np.unique(self.dataset.filenums))
        nl = N_frames // N_cubes
        planet_sep, planet_pa = list(map(float, self.parameters['pars'].split(',')))
        numbasis = list(map(int, self.parameters['numbasis'].split(',')))
        stamp_size = int(self.parameters['stampsize'])

        if self.parameters['spectrum'] == '':
            spectrum = None
        else:
            spectrum = self.parameters['spectrum']


        def fake_spect(pa, fake_flux, basis):
            fakepsf = []
            psfs = np.tile(self.PSF_cube, (N_cubes, 1, 1))
            for i in range(len(psfs)):
                spec = fake_flux[i % 22] * psfs[i]
                fakepsf.append(spec)
            
            tempdataset = charis.CHARISData(self.data)
            fakes.inject_planet(tempdataset.input, tempdataset.centers, fakepsf, tempdataset.wcs, planet_sep, pa)
            
            fm_class = es.ExtractSpec(tempdataset.input.shape,
                                        numbasis=basis,
                                        sep=planet_sep,
                                        pa=pa,
                                        input_psfs=self.PSF_cube,
                                        input_psfs_wvs=np.unique(self.dataset.wvs),
                                        stamp_size=stamp_size)
            
            fm.klip_dataset(tempdataset, fm_class,
                        fileprefix="fmspect_"+"pa="+str(pa),
                        annuli=[[planet_sep-stamp_size,planet_sep+stamp_size]],
                        subsections=[[(pa-stamp_size)/180.*np.pi,\
                                    (pa+stamp_size)/180.*np.pi]],
                        movement=int(self.parameters['movement']),
                        numbasis=basis,
                        spectrum=spectrum,
                        save_klipped=boolean(self.parameters['saveklipped']), 
                        highpass=boolean(self.parameters['highpass']),
                        outputdir=os.path.join(self.fakes_dir, str(basis)))
            
            exspect_fake, fm_matrix_fake = es.invert_spect_fmodel(tempdataset.fmout, 
                                                                tempdataset, 
                                                                units=self.parameters['units'],
                                                                scaling_factor=self.parameters['scalefactor'],
                                                                method=self.parameters['reverse_method'])
            
            del tempdataset
            return exspect_fake


        nplanets = int(config['Errorbars']['nplanets'])
        pas = (np.linspace(planet_pa, planet_pa+360, num=nplanets+2)%360)[1:-1]
        fake_spectra_all_bases = []
        for i in range(len(numbasis)):
            input_spect = self.exspect[i,:]
            fake_spectra = np.zeros((nplanets, nl))
            for p, pa in enumerate(pas):
                basis_dir = os.path.join(self.fakes_dir, str(numbasis[i]))
                if not os.path.exists(basis_dir):
                    os.makedirs(basis_dir)
                fake_spectra[p,:] = fake_spect(pa, input_spect, numbasis[i])
                percent = ((p + len(pas)*i)/(len(numbasis) * len(pas))) * 100
                print(str(percent) + "%")
            fake_spectra_all_bases.append(fake_spectra)

        return fake_spectra_all_bases

    def _export_csv_dataset(self, spect, prefix):
        N_frames = len(self.dataset.input)
        N_cubes = np.size(np.unique(self.dataset.filenums))
        nl = N_frames // N_cubes
        numbasis = list(map(int, self.parameters['numbasis'].split(',')))
        
        d = {"wvs": self.dataset.wvs[:nl]}
        for i in range(len(numbasis)):
            d[numbasis[i]] = [val.nominal_value for val in spect[i]]

        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.spectra_dir, prefix + "_spectra.csv"))

        d = {"wvs": self.dataset.wvs[:nl]}
        for i in range(len(numbasis)):
            d[numbasis[i]] = [val.std_dev for val in spect[i]]

        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.spectra_dir, prefix + "_spectra_error.csv"))


        