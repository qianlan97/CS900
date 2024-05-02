# Gramine-SGX

## This program runs on `Gramine` which is a layer OS on top of Intel's SGX.

Tutorials about installing SGX dependency is listed in [Most up to date installation guide (2/28/2023 version)](https://download.01.org/intel-sgx/latest/linux-latest/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf)
- In depth installation guide, useful for information about required drivers, the github below is easier to follow for PSW and SDK installation
- In case of bios with "software-enabled" option in bios and driver not found in dev folder use https://github.com/intel/sgx-software-enable

[Linux sgx github with ReadMe for installing Intel SGX PSW and Intel SGX PSW](https://github.com/intel/linux-sgx)

[Installation of Intel sgx sdk on linux 20.04 walkthrough](https://www.youtube.com/watch?v=X0YzzT4uAY4)
- This video does not cover psw installation, refer to github for that, useful for visual represenation

For hardware requirements, you have to use an Intel CPU which probabaly before 10th generation(for example, 9700k), but you can look it up in each CPU's specification to check whether they support Intel SGX or not. Type your cpu name into search bar [Check for valid intel cpu](https://ark.intel.com/content/www/us/en/ark/products/186604/intel-core-i79700k-processor-12m-cache-up-to-4-90-ghz.html)

Also, you must use a Linux OS that is not installed on a Mac machine. Since Mac doesn't allow you to enter bios, you can't enable SGX features.

Everything about installing the Gramine/ tuning software environment is listed in [See Gramine document](https://gramine.readthedocs.io/en/stable/index.html).

# Important

In `pytorch.manifest.template`, we have to put every file we're using into `trusted_file` and `allowed_files` prior to compiling, otherwise it will output permission denied.

- `trusted_files` are those we need read access.
- `allowed_files` are those we need both read and write access.

# Files

- `pytorch.manifest.template` is the template file we need to build gramine application on top of SGX.

- `sdc_vgg19.py` is the file to record sdc probabilities of each layer.

- `client_vgg19.py` and `server.py` are files to record total extra time.

- `DBSCAN.py` is the file for DBSCAN classification. 

- `linear formula.ipynb` is the file for linear classification.

- `plot.py` is the file to plot the layer properties.

- `table data.txt` stores the data we collected from SGX experiment.

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

# Build

First make sure that you have SGX environment installed on your device with the correct PATH variable settings so you can call SGX inside the github repo directory.

Then run `make` to build the non-SGX version and `make SGX=1` to build the SGX
version.

Use `make clean` to clean files after the run.

# Run

Execute any one of the following commands to run the workload:

- natively: `python3 pytorchexample.py`
- Gramine w/o SGX: `gramine-direct ./pytorch ./pytorchexample.py`
- Gramine with SGX: `gramine-sgx ./pytorch ./pytorchexample.py`

Here the `./pytorch` doesn't mean folder directory, but the project name is `pytorch`. To modify it, change the `manifest.template` file name along with those file names in `Makefile`.

Example: `gramine-sgx ./pytorch ./client_vgg19.py`

- For recording total extra time experiment, we need to first run `server.py` using plain python, then we can run `client_vgg19.py` inside SGX/gramine.

- For all other python programs, just run it with plain python.
