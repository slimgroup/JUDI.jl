# Viking Graben Line 12: processing

The processing is done using [Madagascar](https://www.reproducibility.org/wiki/Main_Page) open source software. Thus we assume that Madagascar is present on the system.

The processing is partly based on [Rodrigo Morelatto work](https://github.com/rmorel/Viking).

The processing is pretty straightforward:

0. geometry correction
1. deghosting
2. gain
3. turning waves muting
4. turning waves supression using dip filter (not necessary but good for Madagascar use experience)
5. multiples supression using Radon transform
6. automatic velocity picking
7. export to segy

The result of processing (exported to segy) is used by JUDI while RTM/LSRTM.