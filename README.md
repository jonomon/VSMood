# VSmood
Visual scanning data and analysis code for patients with bipolar and unipolar study.

The technical description of the algorithm was presented in the paper "Learning Differences between Visual Scanning Patterns can Disambiguate Bipolar and Unipolar Patients", AAAI 2018.

## Getting started
The analysis code requires the following dependencies (the installation instructions are linked below).
- [python2.7](https://www.python.org/downloads/)
- [Tensorflow](https://www.tensorflow.org/install/) 
- [NumPy](https://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Pandas](https://pandas.pydata.org/getpandas.html)
- [scikit-learn](http://scikit-learn.org/stable/install.html)
- [Keras](https://keras.io/#installation)
- [Matplotlib](https://matplotlib.org/users/installing.html/)

Note: using python from [Anaconda](https://anaconda.org/anaconda) can obtain many of the above dependencies.

## Usage
Run to following code to obtain the analysis results.
```python
python main_JAD.py
```
The proceduces are as follows:
1) Reading, parsing, and processing the data. The following line will be printed on the screen
    ```
    Reading, parsing, and processing visual scanning data
    ```
2) Training the neural network for the leave one out set of subjects (note that this takes about 30 minutes on a laptop computer). Training is complete when the second message is printed.
    ```
    1) Training leave one out classification model...
    Training leave one out classification model complete
    ```
3) The AUC and classification of the leave one out set are then provided and the results should be (tp= true positive, fp=false positive, fn=false negative, tp=true negative):
    ```
    Results 1
    AUC score of the leave one out set=0.874762808349
    Classification of leave one out set
    tp=15, fp=2, fn=7, tn=24
    ```
4) This is followed by training and evaluting the held-out set. The messages are similar to steps 2 and 3. (Note that training for the held-out set should take a couple of minutes).
5) The results of both the leave one out and held out sets, remitted bipolar, remitted unipolar and healthy controls are provided.

    5.1) The AUC of classifying bipolar and unipolar (both in a depressed state) from the both the leave one out and held out set are provided as follows:
    ```
    AUC score of group 1 + 2 = 0.867430441899
    ```
    5.2) The AUC of classifying bipolar (depressed + remitted) vs unipolar (depressed + remitted) and controls from the held out set are provided as follows:
    ```
    AUC for Bipolar (depressed + remitted) vs unipolar (depressed + remitted) and controls = 0.839037927845
    Classification of Bipolar (depressed + remitted) vs unipolar (depressed + remitted) and controls
    tp=33, fp=13, fn=14, tn=80
    ```
    5.3) The similarity index (values in Figure 2) and the corresponding statistic tests are then provided:
    ```
    Similarity index
        Depressed bipolar disorder -0.468464264873+1.27849608688
        Depressed unipolar disorder 1.04657521002+0.881033207938
        Remitted bipolar disorder -0.119614326249+1.14093650413
        Remitted unipolar 1.34760658937+0.899898773382
        Healthy control 1.16913592799+1.07028219018
    Statistic tests (t-tests)
		Bipolar depressed vs unipolar depressed t=-3.34788360318, p=0.00278888654443
		Unipolar depressed vs remitted t=-0.989352985765, p=0.329281182329
		Unipolar depressed vs healthy controls t=-0.375520481766, p=0.709257514171
		Unipolar depressed vs Bipolar remitted t=3.26934124438, p=0.0024714469487
		Bipolar depressed vs unipolar remitted t=-4.28287147359, p=0.000195991579003
		Bipolar depressed vs healthy controls t=-3.64659619235, p=0.00090619356333
		Bipolar depressed vs bipolar remitted t=-0.70748451032, p=0.485327890579
		Healthy controls vs bipolar remitted t=3.84708163655, p=0.000382196980871
		Healthy controls vs unipolar remitted t=-0.596550638954, p=0.553796871336
    ```
    5.4) Finally the plots are generated (please view the img folder for the desired image):
    ```
    ROCs saved in img/rocs.png..
    Similarity index box plots saved in img/mle_bar.png...
    ```
    ![mle_bar.png](https://github.com/jonomon/VSMood/blob/master/img/mle_bar.png)
### Note:
- In keras.json, the "image_data_format" should be set to "channels_last" and "backend" should be set to "tensorflow".
- For the technical description use _main_AAAI.py_. This implementation provides an exploratory interface to test multiple parameters:
	- Data type: RNN: "fix", "glance", LRCN: "fix-sequence"
	- Multiple instance: how to solve the multiple instance problem options: mean, 2d-mean, max-likelihood, similar, log-prob
	- Model investigations:
		- Number of CNN layers
		- Region models (semantic5, grid9, semantic8, grid16)
	- Additional features:
		- scan_path, glance duration (for glance the data type)
		- image type and image position
		
	To run the 3 conditions presented in the AAAI paper:	
		```python
		python main_AAAI.py fix 128 mean --region_model_type semantic8 (or semantic5)
        python main_AAAI.py fix 128 mean --region_model_type grid16 (or grid9)
        python main_AAAI.py fix-sequence 256 mean --print_sub
		```
