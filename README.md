# Gramine-SGX

## This program runs on `Gramine` which is a layer OS on top of Intel's SGX.
[See Gramine document](https://gramine.readthedocs.io/en/stable/index.html)

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

# Run

Execute any one of the following commands to run the workload:

- natively: `python3 pytorchexample.py`
- Gramine w/o SGX: `gramine-direct ./pytorch ./pytorchexample.py`
- Gramine with SGX: `gramine-sgx ./pytorch ./pytorchexample.py`

Here the `./pytorch` doesn't mean folder directory, but the project name is `pytorch`. To modify it, change the `manifest.template` file name along with those file names in `Makefile`.

- For `sdc_vgg19.py`, just run it inside SGX/gramine.

- For recording total extra time experiment, we need to first run `server.py` using plain python, then we can run `client_vgg19.py` inside SGX/gramine.

- For other python programs, just run it with plain python.
