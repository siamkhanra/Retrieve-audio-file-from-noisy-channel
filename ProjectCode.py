from scipy.io.wavfile import read, write
from matplotlib import pyplot as plt
import multiprocessing as mltp
import numpy as np
import os
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal import spectrogram, normalize
#%% 1)
# Playing the Audio (Duration is about 1 min 56 sec)
# It will open your default media player and
# play the music − November Rain (Guiter Lead)

os.system('november_rain.wav')
sampling_rate, audio_sig = read('november_rain.wav') # Reading the given audio file
n_channel = audio_sig.shape[1] # Number of channels in audio data

#sampling_ratesampling_rate = audio_sig[0] # Number of Samples per second

audio_array = np.array(audio_sig) # Converting audio data into numpy array

# Duration and the Time Grid
Duration = audio_sig.shape[0]/sampling_rate
Time = np.linspace(0, Duration, len(audio_array[:,0]))

print("Read = ",audio_sig)
print("Array Signal = ",audio_array)
print("Duration = ",Duration)

# Plot
plt.figure(1, figsize = (8,3.5), dpi = 300)
plt.plot(Time, audio_array[:, 0],'--b')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Channe-1 Signal')
plt.grid()

plt.figure(2, figsize = (8,3.5), dpi = 300)
plt.plot(Time, audio_array[:, 1],'--b')
plt.xlabel('Time,(t)')
plt.ylabel('Amplitude')
plt.title('Channe-2 Signal')
plt.grid()
#%% 2)
Enrg1 = np.sum(audio_array[:,0]**2)
Enrg2 = np.sum(audio_array[:,1]**2)
Int1 = np.max(audio_array[:,0])
Int2 = np.max(audio_array[:,1])

print('Energy of Channel-1 :',Enrg1)
print('Max Intensity of Channel-1 :',Int1)
print('Energy of Channel-2 :',Enrg2)
print('Max Intensity of Channel-2 :',Int2)

# Normalizing the Audio Array
audio_array_norm = np.zeros((len(audio_array[:,0]), n_channel)) # Initialize Norm Array
max_sigs = np.zeros(n_channel) # Max value of each channel
for i in range(n_channel):
    max_sigs[i] = max(audio_array[:,i]) # will be required to generate the audio
    audio_array_norm[:,i] = audio_array[:,i]/max_sigs[i]
    
chnl1 = audio_array_norm[:,0]
chnl2 = audio_array_norm[:,1]

# Plotting the normalized audio signal of each channel in time domain
plt.figure(3, dpi = 300)
plt.plot(Time, audio_array_norm[:,0])
plt.xlabel("Time")
plt.ylabel("Normalized Audio of Channel−1")

plt.figure(4, dpi = 300)
plt.plot(Time, audio_array_norm[:,1])
plt.xlabel("Time")
plt.ylabel("Normalized Audio of Channel−2")
#%% 3)
# Defining function to generate frequency domain signal
def FT(x, t):
     N = int(2**np.ceil(np.log2(len(x)))); # 2's power larger than signal length
     Ts = np.mean(np.diff(t)); # Sampling time
     x_paded = np.append(x, [0]*(N-len(x))) # Zero Padding
     X_freq = fftshift(fft(x_paded))/N
     freq = fftshift(fftfreq(N, Ts))
     return X_freq, freq

Y1, f1 = FT(chnl1,Time)
Y2, f2 = FT(chnl2,Time)

combine = Y1 + Y2
c = (combine - np.min(combine)) / (np.max(combine) - np.min(combine))
freq = f1 + f2

plt.figure(5,figsize = (7,3.5), dpi = 300)
plt.plot(f1, abs(Y1),'g')
plt.title('Spectrums of Channel-1')
plt.xlabel('Frequency')
plt.grid()

plt.figure(6,figsize = (7,3.5), dpi = 300)
plt.plot(f2, abs(Y2),'r')
plt.title('Spectrums of Channel-2')
plt.xlabel('Frequency')
plt.grid()

plt.figure(7,figsize = (7,3.5), dpi = 300)
plt.plot(freq, abs(c),'b')
plt.title('Combined Spectrums of Channel-1 and Channel-2 ')
plt.xlabel('Frequency')
plt.grid()

plt.figure(8,figsize = (7,3.5), dpi = 300)
plt.plot(freq, abs(combine),'y')
plt.title('Combined Spectrums Without Normalization for both Channel')
plt.xlabel('Frequency')
plt.grid()
#%% 4)
Fs = sampling_rate
Ts = 1/Fs
A = len(audio_sig)

# Plot for Channel-1 Signal
f, t, s = spectrogram(chnl1, Fs, nperseg = int (A/10))
plt.figure(9, figsize = (7,3.5), dpi = 300)
plt.pcolormesh(t, f, s, vmin = 0, vmax = 0.0003)
plt.title('Spectrogram for Channel-1')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.ylim((0,3000))
plt.show()

# Plot for Channel-2 Signal
f, t, s = spectrogram(chnl2, Fs, nperseg = int (A/10))
plt.figure(10, figsize = (7,3.5), dpi = 300)
plt.pcolormesh(t, f, s, vmin = 0, vmax = 0.0003)
plt.title('Spectrogram for Channel-2')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.ylim((0,3000))
plt.show()
#%% 5)
c2 = chnl1 + chnl2
c3 = (c2 - np.min(c2)) / (np.max(c2) - np.min(c2)) # Normalization

f, t, s = spectrogram(c3, Fs, nperseg = int (A/10))
plt.figure(11, figsize = (7,3.5), dpi = 300)
plt.pcolormesh(t, f, s, vmin = 0, vmax = 0.0001)
plt.title('Combined Spectrogram for both Channel')
plt.xlabel('Time')
plt.ylabel('Freq')
plt.ylim((0,3000))
plt.show()
#%% 6)
print("For Channel-1 Parson's Code is = *DRUDURDUD")
print("For Channel-2 Parson's Code is = *DRUDURDUD")
print("For Channel-1 Parson's Code is = *DURDURRRD")
#%% 7)
Fg = 20000 # Global frequency
Fs = 1000
Bin_Size = int(Fg/Fs)
x_sig = audio_array_norm[:,0] # For channel 1
x_sam = x_sig[0:-1:Bin_Size]
t_sam = Time[0:-1:Bin_Size]
#%% Quantization for Channel-1
L = 256 # Level
b = int(np.ceil(np.log2(L)))
Delta = (max(x_sam)-min(x_sam))/(L-1)
Q_level = np.linspace(min(x_sam),max(x_sam),L)

plt.figure(12, figsize = (8,3.5), dpi = 300)
plt.stem(t_sam,x_sam,'--b',use_line_collection=True)
plt.title("Channel-1 Quantized Signal")
for q in Q_level:
    plt.plot(t_sam,q*np.ones(len(t_sam)),'--b',linewidth = 0.5)
    
# Rounding
x_qnt = []
for x in x_sam:
    idx = np.argmin(abs(Q_level-x))
    x_qnt.append(Q_level[idx])
    
plt.figure(13, figsize = (8,3.5), dpi = 300)
plt.stem(t_sam,x_qnt,'--b',use_line_collection=True)
plt.title("Channel-1 Quantized Signal with Q-levels")
for q in Q_level:
    plt.plot(t_sam,q*np.ones(len(t_sam)),'--b',linewidth = 0.5)
#%%
x_sig1 = audio_array_norm[:,1] # For channel 2
x_sam1 = x_sig1[0:-1:Bin_Size]
t_sam1 = Time[0:-1:Bin_Size]
#%% Quantization for Channel-2
Delta1 = (max(x_sam1)-min(x_sam1))/(L-1)
Q_level1 = np.linspace(min(x_sam1),max(x_sam1),L)

plt.figure(14, figsize = (8,3.5), dpi = 300)
plt.stem(t_sam1,x_sam1,'--g',use_line_collection=True)
plt.title("Channel-2 Quantized Signal")
for q in Q_level1:
    plt.plot(t_sam1,q*np.ones(len(t_sam1)),'--g',linewidth = 0.5)
    
# Rounding
x_qnt1 = []
for x in x_sam1:
    idx1 = np.argmin(abs(Q_level1-x))
    x_qnt1.append(Q_level1[idx1])
    
plt.figure(15, figsize = (8,3.5), dpi = 300)
plt.stem(t_sam1,x_qnt1,'--b',use_line_collection=True)
plt.title("Channel-2 Quantized Signal with Q-levels")
for q in Q_level1:
    plt.plot(t_sam1,q*np.ones(len(t_sam1)),'--g',linewidth = 0.5)
#%% 8) Channel-1 PCM Encoding
bit_dict = {}
b = int(np.ceil(np.log2(L)))
for i in range(len(Q_level)):
    bt = bin(i)[2:].zfill(b)
    bit_dict[Q_level[i]] = bt
print("Channel-1 Q-Level:Bit Dictionary =\n")
[print(keys, ":", vals) for keys, vals in bit_dict.items()]
    
x_pcm = ''
for x in x_qnt:
    idx = np.argmin(abs(x-Q_level))
    x_pcm += bit_dict[Q_level[idx]]
x_hex = hex(int(x_pcm,2)).upper()[2:] 

print("\nChannel-1 PCM Bit String (Hex) =", x_hex)
  
err = x_sam - x_qnt
SQNR = 10*np.log10(np.mean(x_sam**2)/np.mean(err**2)) # Simulation
print('SQNR for Channel-1 = ',SQNR,'dB')
#%% Channel-2 PCM Encoding
bit_dict1 = {}
b = int(np.ceil(np.log2(L)))
for i in range(len(Q_level1)):
    bt = bin(i)[2:].zfill(b)
    bit_dict1[Q_level1[i]] = bt
print("Channel-2 Q-Level:Bit Dictionary =\n")
[print(keys, ":", vals) for keys, vals in bit_dict.items()]
    
x_pcm1 = ''
for x in x_qnt1:
    idx1 = np.argmin(abs(x-Q_level1))
    x_pcm1 += bit_dict1[Q_level1[idx1]]
x_hex1 = hex(int(x_pcm1,2)).upper()[2:] 

print("\nChannel-2 PCM Bit String (Hex) =", x_hex1)
  
err1 = x_sam1 - x_qnt1
SQNR1 = 10*np.log10(np.mean(x_sam1**2)/np.mean(err1**2)) # Simulation
print('SQNR for Channel-2 = ',SQNR1,'dB')
#%% Channel-1 Decoding ( Knows bit_dict, bit_rate = Rb) --- Receive x_pcm
inv_bit_dict = dict([(vals,keys) for keys, vals in bit_dict.items()])

for vals in bit_dict.values():
    b = len(vals)
    
Ts = 1/Fs
t_rec = []
x_rec = []

for i in range(0,len(x_pcm),8):
    bt = x_pcm[i: i+8]
    x_rec.append(inv_bit_dict[bt])
    t_rec.append(i*Ts)

plt.figure(16, figsize = (8,3.5), dpi=300)
plt.title("Retrieved Channel-1 Signal")
plt.stem(t_rec,x_rec,markerfmt='--r',use_line_collection=True)
#%% Channel-2 Decoding ( Knows bit_dict, bit_rate = Rb) --- Receive x_pcm1
inv_bit_dict1 = dict([(vals,keys) for keys, vals in bit_dict.items()])

for vals in bit_dict.values():
    b = len(vals)
    
Ts = 1/Fs
t_rec1 = []
x_rec1 = []

for i in range(0,len(x_pcm1),8):
    bt = x_pcm1[i: i+8]
    x_rec1.append(inv_bit_dict1[bt])
    t_rec1.append(i*Ts)

plt.figure(16, figsize = (8,3.5), dpi=300)
plt.title("Retrieved Channel-2 Signal")
plt.stem(t_rec1,x_rec1,markerfmt='--r',use_line_collection=True)
#%% 9)
# Initialize the audio array to be generated
gen_audio_array = np.zeros((len(audio_array_norm[:,0]), n_channel), dtype='int16')
for i in range(n_channel):
    gen_audio_array[:,i] = audio_array_norm[:,i]*max_sigs[i]

# Writing the array in the audio .wav file
write("Generated_Audio.wav", sampling_rate, gen_audio_array)
os.system("Generated_Audio.wav") # Playing the sound in default media app