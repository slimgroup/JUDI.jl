# Full Waveform Inversion

FWI is done using JUDI.

The idea is the following:

1. prepare initial model
2. filter segy using low pass filter
3. run 10 iteration of FWI using filtered segy

Steps 2 and 3 run recursively while increasing low-pass frequency.

For example first iteration we run FWI with raw segy file (`seismic.segy`) filtered with 5 Hz lowpass frequency filter and initial model.

The second iteration FWI is ran using raw segy file filtered with 8 Hz lowpass filter and with the model calculated from previous iteration.

Continue doing so until required resolution is achieved.

