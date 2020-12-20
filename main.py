import os
import itertools
import configparser
from pipeline.klip import KLIP
from pipeline.settings import config
from pipeline.charisdrp import ExtractCubes
from pipeline.helpers import boolean
from pipeline.modules.figures import Figures

def main():
    root_dir = config['Paths']['root_dir']
    case_dir = config['Paths']['case_dir'].split(',')

    for case in case_dir:
        case_path = case
        if not os.path.isabs(case):
            case_path = os.path.join(root_dir,case)
    
        # stage 1: convert raw images into datacubes
        if boolean(config['Stage']['stage1']):
            ex = ExtractCubes(case_path)
            ex.generate_wavecal()
            ex.generate_extract_config()
            ex.generate_reduced_cubes()
            del ex
        
        # stage 2: run KLIP-FM on datacubes from stage 1 to 
        # get spectrum
        if boolean(config['Stage']['stage2']):
            dynamic_vals = []
            dynamic_val_keys = []
            for param, vals in config.items('Klip-dynamic'):
                dynamic_vals.append(vals.split(','))
                dynamic_val_keys.append(param)
            param_comb = list(itertools.product(*dynamic_vals))
            for comb in param_comb:
                params = dict(zip(dynamic_val_keys, comb))
                k = KLIP(case_path, params)
                k.klip_general()
                k.klip_fm_spect(overwrite=True)
                del k


    # stage 3: perform various forms of analysis on the 
    # spectra from stage 2

    # use the spectrum in stage 2 to plot graphs of the 
    # spectra
    #f = Figures()
    #f.uncalib_fig()
    #f.calib_fig()
    #f.mag_fig()
    #f.all_obs_fig()
                    





















if __name__ == '__main__':
    main()