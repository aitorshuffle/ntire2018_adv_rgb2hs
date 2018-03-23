# -*- coding: utf-8 -*-
import pandas as pd
import os
import sacred
import glob
from sacred import Experiment
ex = Experiment('rename_to_samename')

@ex.config
def config():
    results_home_dir = os.path.abspath('/home/aitor/dev/adv_rgb2hs_pytorch/results')

@ex.automain
def select_model(results_home_dir):
    res_dir_list = glob.glob(results_home_dir + '/*')
    dfall_list = []
    for res_dir in res_dir_list:
        exp = os.path.basename(res_dir)
        fpath = os.path.join(res_dir, 'scores.txt')

        try:
            f = open(fpath)
        except IOError:
            print(fpath + ' does not exist')
        else:
            with f:
                content = f.readlines()
                content = [x.strip() for x in content]
                results = dict([elem.split(':') for elem in content])
                results = {k: [v] for k, v in results.items()}  # from_dict() needs iterable as value per key/column name
                results['exp'] = [exp]
                dfbuff = pd.DataFrame.from_dict(results)
                dfbuff = dfbuff.set_index('exp')
                dfall_list.append(dfbuff)
    dfall = pd.concat(dfall_list)
    dfall = dfall.astype(float)
    print(dfall.sort_values(by='RMSE', ascending=True))
    print(dfall.sort_values(by='MRAE', ascending=True))
    pass
