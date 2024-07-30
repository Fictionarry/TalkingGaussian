'''
The AUE evaluation is based on OpenFace (https://github.com/TadasBaltrusaitis/OpenFace). 
First, use OpenFace's FeatureExtraction to process the reconstructed video ("A_generated.mp4" for example) 
and the corresponding GT ("A_GT.mp4" for example) respectively.
Then, run "python auerror.py A_generated A_GT"

Default directory structure is:

|--- auerror.py
|--- OpenFace_2.2.0_win_x64
     |--- proceessed
     |--- ...

'''


import pandas as pd
import os
import sys
import numpy as np

AUitems = [' AU01_r',' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']

df_1 = pd.read_csv(os.path.join('./OpenFace_2.2.0_win_x64/processed', sys.argv[1]+'.csv'))[AUitems]
df_2 = pd.read_csv(os.path.join('./OpenFace_2.2.0_win_x64/processed', sys.argv[2]+'.csv'))[AUitems]

error = (df_1-df_2)**2
print(error.mean().sum())


AUitems_lower = [' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r']
AUitems_upper = [' AU01_r',' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU45_r']

df_1 = pd.read_csv(os.path.join('./OpenFace_2.2.0_win_x64/processed', sys.argv[1]+'.csv'))[AUitems]
df_2 = pd.read_csv(os.path.join('./OpenFace_2.2.0_win_x64/processed', sys.argv[2]+'.csv'))[AUitems]

error_l = (df_1[AUitems_lower]-df_2[AUitems_lower])**2
error_u = (df_1[AUitems_upper]-df_2[AUitems_upper])**2

print('l:', error_l.mean().sum(), 'u', error_u.mean().sum())