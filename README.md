# Gramine-SGX

## This program runs on `Gramine` which is a layer OS on top of Intel's SGX.

There are os permission errors, so we need to run `download-pretrained-model.py` to download pre-trained models and save them to pt prior to use.

# Important

In `pytorch.manifest.template`, we have to put every file we're using into `trusted_file` and `allowed_files` prior to compiling, otherwise it will output permission denied.

- `trusted_files` are those we need read access.
- `allowed_files` are those we need both read and write access.

# BUG

~~Currently it seems like the whole gramine manifest template are too large so that a simple helloworld would crash the system. Need to clean it up and delete things out of the template.~~ Fixed

# TODO

1. ~~Python script  with pass-in/pass-out with 1 layer done. However, this is outside the pytorch framwork.~~
2. ~~Do the linear/non-linear 1 layer coputation all in TEE, record time.~~
3. ~~Current implementation uses the default model. If we want to achieve layer by layer loop, we might need to implemnt the model by hand and add current scripts into the `foward()` function. After that, loop the layer runs without restarting the whole enclave(try sleep for 5 sec and read the passed out data from a file.)~~
4. Add encryption/decription between user and tee communication, record tee decryption time for AES256 in tee during the first data uploading.
5. Plot the runtime as bar chart.
6. ~~If the performance of python is too bad, estimate how much time it would take to rebuild the whole system in C/C++ with existed intel sgx code.~~ No need, performance of Gramine is enough.

- Feasibility: one plot to show that inference/gradient computation can replica same result in CPU and GPU.
- Computational cost measurements.
- Memory/ thread management for better performance/ lower system requirements.
- `Warning: Emulating a raw syscall instruction. This degrades performance, consider patching your application to use Gramine syscall API.` Will need to work on this to see if can speed up.

# Runtime

## Diff between TEE and Protocol for DL models for 1 epoch 1 image
### Resnet 18
- tee: --- 22.586578845977783 seconds ---
- protocol: --- 10.98341703414917 seconds ---

### VGG 19
- tee: --- 41.56727194786072 seconds ---
- cpu: --- 9.403263807296753 seconds ---
- gpu: --- 0.2404041290283203 seconds ---
- protocol: --- 5.796084880828857 seconds ---

### Alexnet
- tee: --- 8.39589786529541 seconds ---
- protocol: --- 4.18131010055542 seconds ---(3.38s between first and second conv layer. Don't know why yet.)

### timestamp for protocol of VGG19
```
running on cpu
Files already downloaded and verified
Files already downloaded and verified
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 0.2095952033996582 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 1.580354928970337 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 1.8986999988555908 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 2.153087854385376 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 2.3200559616088867 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 2.577785015106201 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 2.726041078567505 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 2.854192018508911 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.0494608879089355 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.2291581630706787 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.4095630645751953 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.5729730129241943 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.704503059387207 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.8312480449676514 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 3.95023512840271 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Conv2d layer: --- 4.065775156021118 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Linear layer: --- 4.863068103790283 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Linear layer: --- 4.995892286300659 ---
Tensor written to shared memory. Waiting for the server to process it.
New tensor received from the server:
Linear layer: --- 5.021748065948486 ---
backward: --- 5.777837038040161 ---
batch # 1 finished
BatchNorm time: --- 2.730213165283203 --- seconds
Pooling time: --- 0.16330432891845703 --- seconds
ReLU time: --- 0.5273535251617432 --- seconds
TEE->GPU client side time: --- 0.18332338333129883 --- seconds
GPU->TEE client side time: --- 0.30060482025146484 --- seconds
Total time: --- 5.796084880828857 --- seconds
```

## 1 epoch, 64 images in total, batch size 4
- GPU:
```
running on cuda
Files already downloaded and verified
Files already downloaded and verified
--- 7.01060938835144 seconds ---
```

TEE:
```
running on cpu
Files already downloaded and verified
Files already downloaded and verified
--- 713.7495138645172 seconds ---
```

Protocol:
```
All finished: --- 324.5917081832886 seconds ---
```
####  For detailed log, see `log.md`

### AES256 Encryption/ Decryption time
- byte file length: 184442413 byte
- in CPU
```
data encrypted: --- 0.47696709632873535 seconds ---
data decrypted: --- 1.0293693542480469 seconds ---
```
- in TEE
```
data encrypted: --- 7.482712745666504 seconds ---
data decrypted: --- 16.785684823989868 seconds ---
```

#### Summary of VGG19 Result
TEE side: BN = 3.13992s, ReLU = 0.71691s, Pooling = 0.21443s, BP = 0.60691s;    
Server + Comm = 2.41706s; Comm 1(TEE-> GPU) = 0.53886s, Comm 2 (GPU-> TEE) =?, GPU comp = ?


## Diff between TEE and Protocol with different matrix size on a single layer
### 2304 * 1024: 
#### tee:
- loaded from user: --- 0.3929779529571533 seconds ---
- computed: --- 0.4339468479156494 seconds ---
- Verification pass: --- 0.43481993675231934 seconds ---

#### protocol:
- loaded from user: --- 0.39055705070495605 seconds ---
- saved to pickle: --- 0.526043176651001 seconds ---
- sent to server: --- 0.5617671012878418 seconds ---
- received from server: --- 0.6261270046234131 seconds ---
- Verification pass: --- 0.6280198097229004 seconds ---

- compare times: 0.692366

### 23040 * 1024:
#### tee:
- loaded from user: --- 5.102967739105225 seconds ---
- computed: --- 253.9476580619812 seconds ---
- Verification pass: --- 253.95157313346863 seconds ---

#### protocol:
- loaded from user: --- 5.141739130020142 seconds ---
- saved to pickle: --- 8.405448913574219 seconds ---
- sent to server: --- 10.560967922210693 seconds ---
- received from server: --- 11.047080039978027 seconds ---
- Verification pass: --- 11.049587965011597 seconds ---

- compare times: 22.98

The total memory size of this PC is 32GB. 
- Too small size(4GB) can't run the model inside enclave.
- Too large size(16GB) might cause the PC to crash.
- Also, the `sgx.max_threads` must be at least 12 to get the program running.

It looks like `sgx.enclave_size` / `sgx.max_threads` must be larger than the memory needs for the model to compute(302MB for `resnet18.py`).

Thus here we're setting the enclave size to 16GB(with closing all other apps on the PC to let it not crash), and enclave maximum threads amount to 12.

Sometimes it half-crashes with `[P1:T1:python3.7] error: process creation failed`, but still works after that error message. Confusing.

- pure python3 (1 epoch, running on cuda): --- 25.08575963973999 seconds ---
- gramine-direct (20 epoch, running on cpu but without enclave, 4 GB size): --- 7221.1591012477875 seconds ---
- gramine-direct (1 epoch, running on cpu but without enclave, 16 GB size): --- 459.85171031951904 seconds ---
- gramine-sgx (1 epoch, running on cpu and inside enclave, 16GB size): --- 8500.321489810944 seconds ---

- pure python3 (1 layer test case with server pass in/out, nothing with pytorch): --- 2.1681034564971924 seconds ---
- gramine-sgx (1 layer test case with server pass in/out, nothing with pytorch): --- 4.269568920135498 seconds ---
- gramine-sgx (1 layer test case without server): --- 0.4296298027038574 seconds ---

### on server gpu:
- load: --- 0.019591331481933594 seconds ---
- saved: --- 0.06630301475524902 seconds ---
- uploaded npy: --- 1.858811378479004 seconds ---
- uploaded script: --- 1.8832411766052246 seconds ---
- server computed: --- 1.8898158073425293 seconds ---
- tee received: --- 2.1368095874786377 seconds ---
- Verification Pass
- verification: --- 2.1370346546173096 seconds ---
### on local client side:
- loaded from user: --- 0.013624906539916992 seconds ---
- saved to pickle: --- 0.02732682228088379 seconds ---
- sent to server: --- 0.049510955810546875 seconds ---
- received from server: --- 0.10812211036682129 seconds ---
- Verification pass: --- 0.10817861557006836 seconds ---

### on local server side:
- Server is listening...
- received: --- 0.07168030738830566 seconds ---
- converted to pytorch: --- 0.07173705101013184 seconds ---
- computed: --- 0.10769248008728027 seconds ---
- converted to numpy: --- 0.10773301124572754 seconds ---
- sent: --- 0.10791635513305664 seconds ---

# Compare gradient of training

`resnet18_train.py` will save the result vector to `gradient.pt`

For convenience, I will manually save the two results to `gradients_outside.pt` and `gradients_inside.pt`

After setting the global random seed, the gradients are exact the same from training on a single image with 1 iteration.

# Compare output of inference

`resnet18_inference.py` will save the result vector to `result.txt`

For convenience, I will manually save the two results to `result_outside.txt` and `result_inside.txt`

The results are exactly the same.

# Script to server

Since others ports are all closed, `ssh` is the only way to commute with the server. 

For easier implemtation, the matrix will be stored in two local txt files and pass to the server using `ssh`. Then the server will extract matrix from file, do the computation, save result to a file and pass back to tee side using `ssh`.

# Memory

The `resnet18.py` model currently takes about 302 MB of memory and it's reaching the limit, causing memory allocation error.

Not sure if this can be fixed by adjusting the manifest template. 

Not sure it this requires cache or memory. 

Currently using 1 epoch instead of 20 epochs. Hope this works.

# Files

- `helloworld.py` is the test file.

- `resnet18.py` is what currently being working on.

To add more files, add as `"file:helloworld.py",` in `pytorch.manifest.template`'s `trust file` section.

# Pre-requisites

The following steps should suffice to run the workload on a stock Ubuntu 18.04
installation.

- `sudo apt install libnss-mdns libnss-myhostname` to install additional
  DNS-resolver libraries.
- `sudo apt install python3-pip lsb-release` to install `pip` and `lsb_release`.
  The former is required to install additional Python packages while the latter
  is used by the Makefile.
- `pip3 install --user torchvision pillow` to install the torchvision and pillow
  Python packages and their dependencies (usually in $HOME/.local). WARNING:
  This downloads several hundred megabytes of data!
- `python3 download-pretrained-model.py` to download and save the pre-trained
  model. WARNING: this downloads about 200MB of data!

# Build

Run `make` to build the non-SGX version and `make SGX=1` to build the SGX
version.

# Run

Execute any one of the following commands to run the workload:

- natively: `python3 pytorchexample.py`
- Gramine w/o SGX: `gramine-direct ./pytorch ./pytorchexample.py`
- Gramine with SGX: `gramine-sgx ./pytorch ./pytorchexample.py`

Here the `./pytorch` doesn't mean folder directory, but the project name is `pytorch`. To modify it, change the `manifest.template` file name along with those file names in `Makefile`.
