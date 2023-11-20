# Full Waveform Inversion

FWI is done using JUDI.

The idea is the following:

1. Prepare initial model
2. Trim segy to 4 sec to reduce computation time
3. Run 10 iteration of FWI at a given frequency

Steps 2 and 3 run recursively while increasing low-pass frequency.

Possible frequencies are: 0.005, 0.008, 0.012, 0.018, 0.025, 0.035 kHz (don't forget to increase space order JUDI option to 32 points for frequencies > 0.012 kHz)

To reduce the amount of RAM one may want to try to add `subsampling_factor=10` to JUDI options. 
Usually at low frequencies 10 times subsampling doesn't affect the result of FWI.
