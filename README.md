# VSmood
Visual scanning data and analysis code for patients with bipolar and unipolar study.

The technical description of the algorithm was presented in the paper "Learning Differences between Visual Scanning Patterns can Disambiguate Bipolar and Unipolar Patients", AAAI 2018.

## Dependencies
- python2.7
- [Tensorflow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)

## Usage
There are two main files that can be used main_JAD.py, and main_AAAI.py. 
- _main_JAD.py_ provides a direct interface to explore the patient's visual scanning sequences. The number of hidden states and epochs are available parameters.
    **Note**: The JAD implementation uses: a grid based region definition, the log-prob multiple instance, image type and position features.

- _main_AAAI.py_ provides an exploratory interface to test multiple parameters: 
	- Data type: RNN: "fix", "glance", LRCN: "fix-sequence"
	- Multiple instance: how to solve the multiple instance problem options: mean, 2d-mean, max-likelihood, similar, log-prob
	- Model investigations:
		- Number of CNN layers
		- Region models (semantic5, grid9, semantic8, grid16)
	- Additional features:
		- scan_path, glance duration (for glance the data type)
		- image type and image position
		
    To run the 3 conditions presented in the AAAI paper:
        1) python main_AAAI.py fix 128 mean --region_model_type semantic8 (or semantic5)
        2) python main_AAAI.py fix 128 mean --region_model_type grid16 (or grid9)
        3) python main_AAAI.py fix-sequence 256 mean --print_sub

### Note:
In keras.json, the "image_data_format" should be set to "channels_last" and "backend" should be set to "tensorflow".
