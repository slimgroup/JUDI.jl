# Reverse Time Migration

RTM is done using JUDI.

Before running this one should process the data (see `../proc` directory) and complete the FWI to prepare accurate velocity model (see `../fwi` directory).

Be sure to set the correct path to the model computed at FWI step to `model_file` variable in `rtm.jl` script.