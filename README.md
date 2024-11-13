# Hidden Markov Model for Audio Processing

This repository contains an implementation of a Hidden Markov Model (HMM) with discrete transition probabilities and multivariate Gaussian emission probabilities for audio signal processing.

## Requirements

To run the code, you need the following libraries:
- `torch`
- `numpy`
- `soundfile`
- `pandas`
- `scipy`
- `librosa`
- `tqdm`

You can install the required libraries using pip:
```bash
pip install -r requirement.txt
```

## PyTorch Installation

To install PyTorch, you can use the following command:
```bash
pip install torch
```

For more detailed instructions and options, please visit the official PyTorch website: [PyTorch Installation](https://pytorch.org/get-started/locally/)

## Usage
<!-- 
### Initialization

To initialize the HMM model, you need to specify the number of states and the number of features:
```python
from hmm import HMM

n_states = 5
n_features = 13
model = HMM(n_states, n_features)
```

### Training

To train the HMM model, you need to provide the audio data as a numpy array and specify the number of EM iterations:
```python
audio_data = ...  # Load your audio data as a numpy array
n_iter = 100
model.train(audio_data, n_iter)
```

### Forward Pass

To perform the forward pass of the HMM, you need to provide the encoded data:
```python
encoded_data, tau = model.encode(audio_data)
alpha = model.hmm_forward(encoded_data)
```

## File Structure

- `hmm.py`: Contains the implementation of the HMM and EmissionModel classes.
- `README.md`: This file.

## License

This project is licensed under the MIT License.

## Acknowledgements

This implementation is based on the concepts of Hidden Markov Models and their application in audio signal processing. -->
