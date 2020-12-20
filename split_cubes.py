import os
import glob
import math
from shutil import copy2

# parameters
old_cubes_location = '/media/peizhiliu/Backup Plus/2020_Summer/CHARIS/Kappa_and/2017-09-10_exp=20_pa=189-94_combined/extracted/cubes'
new_cubes_dir = '/media/peizhiliu/Backup Plus/2020_Summer/CHARIS/Kappa_and'
dir_basename = '2017-09-10_exp=20_pa=189-94'
interval = 5





files = glob.glob(os.path.join(old_cubes_location, '*.fits'))
outarr = []
for i, f in enumerate(files):
    first = interval * math.floor(i/interval) + 1
    last = interval * math.floor(i/interval) + interval
    dir_name = dir_basename + '_' +  str(first) + '-' + str(last)
    file_base = os.path.basename(f)
    file_path = os.path.join(new_cubes_dir, dir_name, 'extracted/cubes')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    outarr.append(os.path.basename(dir_name))

outarr = list(dict.fromkeys(outarr))
outstring = ''
for out in outarr:
    outstring += ','+out
    
print(outstring)
    #copy2(f, file_path)

