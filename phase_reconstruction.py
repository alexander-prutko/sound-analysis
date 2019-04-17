import numpy as np
import rtpghi as rtpghi
import pghi as pghi
import scipy.signal as signal
import time

n_fft = 2048
sr = 44100
hop_length = 512

# n_fft
M = n_fft

# win_length
win_length = n_fft
gl = win_length

# window
g = signal.windows.hann(gl)    
gamma = gl**2*.25645
Fs = sr

redundancy = win_length/hop_length

p = pghi.PGHI(redundancy=redundancy, M=M, gl=gl, g=g, gamma=gamma, tol=1e-6, show_frames=10, verbose=False, Fs=Fs)

etime = time.clock()        
stereo = []

for magnitude_frames in magnitude_frames_array:
    phase_estimated_frames = p.magnitude_to_phase_estimate(magnitude_frames)
    signal_out = p.magphase_frames_to_signal(magnitude_frames, phase_estimated_frames)
    stereo.append(signal_out)
p.plt.signal_to_file(np.stack(stereo), "sample.wav") 
p.logprint('elapsed time = {:8.2f} seconds\n'.format(time.clock()- etime))

p = rtpghi.PGHI(redundancy=redundancy, M=M, gl=gl, g=g, gamma=gamma, tol=1e-6, show_frames=10, verbose=False, Fs=Fs)

etime = time.clock()        
stereo = []

for magnitude_frames in magnitude_frames_array:
    phase_estimated_frames = p.magnitude_to_phase_estimate(magnitude_frames)
    signal_out = p.magphase_frames_to_signal(magnitude_frames, phase_estimated_frames)
    stereo.append(signal_out)
p.plt.signal_to_file(np.stack(stereo), "sample_rt.wav") 
p.logprint('elapsed time = {:8.2f} seconds\n'.format(time.clock()- etime))