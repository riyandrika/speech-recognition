# speech-recognition

### Agenda
Building a custom hardware accelerator for NLP applications, specifically speech recognition.

### Process & Execution

* Sampling .wav files of voice recordings into time-domain representation of speech --> frequency-domain feature extraction (spectrogram) through Fast Fourier Transform (FFT).
* Constructing a Recurrent-Convolutional Neural Network (RCNN) network for training and inference. Convolutional layers made up of Conv2D and recurrent layers made up of LSTM.
* Network will be reconstructed on lower level abstraction such as C/C++
* C/C++ implementation to be synthesised to register-transfer level (RTL) using Vivado High Level Synthesis (HLS).
* Hardware accelerator to be implemented using a Field Programmable Gate Array (FPGA).
