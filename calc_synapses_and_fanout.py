# This file will build SPAUN either with all modules or the 
# WorkingMemory module only and run preprocessing on the model. 
# After building and preprocessing, compute_stats function is passed the 
# preprocessed model to calculate the compression ratio as well as the 
# total memory resource count and total encoding/decoding weights required.

import build_spaun_all_modules as build_sp
#import build_spaun_wm_only as wmonly

import compute_stats as cp


cp.compute_stats(build_sp.model)
