# Viking Graben Line 12

[Viking Graben Line 12](https://wiki.seg.org/wiki/Mobil_AVO_viking_graben_line_12) is an open source marine 2D seismic datasets.

It contains segy data, source signature, well logs and the description.

Basically this example is dived in two parts: the processing (using Madagascar open source software) and inversion/migration (FWI/RTM/LSRTM) using `JUDI` interface.

Processing scripts are located in `proc` subfolder.

As a rule you have to process the data before inversion/migration.

To download the data use `download_data.ipynb` notebook.

To run FWI/RTM examples you will probably need to have 20-30 Gb RAM available.

The results that the user may get are shown in `FWI_RTM_LSRTM_VikingGraben_line12.pdf` presentation.

The example is provided by [Kerim Khemraev](https://github.com/kerim371)