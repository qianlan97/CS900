# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import random
import time

import torch
import time
import socket
import pickle

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#########################################################

start_time = time.time()

abort_sigal = 0
# getting inputs from user
loaded_arrays = np.load('user_input.npy', allow_pickle=True)
c, alpha, xw, w = loaded_arrays
print("loaded from user: --- %s seconds ---" % (time.time() - start_time))

##### Implement permutation here

##### getting computed result from server
CT = np.dot(c,w)
print("computed: --- %s seconds ---" % (time.time() - start_time))
##### Implement de-permutation here

y = CT - alpha

if(np.array_equal(y, xw)):
    print("Verification pass: --- %s seconds ---" % (time.time() - start_time))
else:
    print("Verification fail: --- %s seconds ---" % (time.time() - start_time))
