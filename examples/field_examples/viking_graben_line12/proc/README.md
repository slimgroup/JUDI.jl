# Viking Graben Line 12: processing

The processing is done using [Madagascar](https://www.reproducibility.org/wiki/Main_Page) open source software. Thus we assume that Madagascar is present on the system.

To install python dependencies use `requirements.txt` file: `python -m pip install -r requirements.txt`

The processing is partly based on [Rodrigo Morelatto work](https://github.com/rmorel/Viking).

The graph is pretty straightforward:

0. Geometry correction
1. Deghosting
2. Gain
3. Turning waves muting
4. Turning waves supression using dip filter (not necessary but good for Madagascar use experience)
5. Multiples supression using Radon transform
6. Automatic velocity picking
7. Export to segy

The result of processing (exported to segy) is used by JUDI while RTM/LSRTM.