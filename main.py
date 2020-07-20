import os
import itertools
import configparser
from pipeline.klip import KLIP
from pipeline.settings import config
from pipeline.charisdrp import ExtractCubes
from pipeline.helpers import boolean

def main():
    root_dir = config['Paths']['root_dir']
    case_dir = config['Paths']['case_dir'].split(',')

    for case in case_dir:
        case_path = case
        if not os.path.isabs(case):
            case_path = os.path.join(root_dir,case)
    
        if boolean(config['Stage']['stage1']):
            ex = ExtractCubes(case_path)
            ex.generate_wavecal()
            ex.generate_extract_config()
            ex.generate_reduced_cubes()
            del ex
        
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
                k.klip_fm_spect()
            del k
                    





















if __name__ == '__main__':
    main()