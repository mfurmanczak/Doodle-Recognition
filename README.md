# Doodle Classification using Convolutional Neural Networks

Software for training a CNN Model on the `Quickdraw` dataset using `Tensorflow`. 

## Installation

To install the required packages, run the following command:

```
pip install -r requirements.txt
```

You might also need to install Ghostscript and add it to path in system variables. This is available under:
`https://ghostscript.com/releases/gsdnld.html`

You will need to download the dataset used, from:
`https://livenorthumbriaac-my.sharepoint.com/:f:/g/personal/w20027449_northumbria_ac_uk/EqjNr_ZZ7VBCpQtg27OXK8oBYt2qXgi4vQCbPhFHZ03Y4g?e=cU543N`
and place it in the `dataset` folder.

## Scripts

The project includes the following scripts:

- `main.py`: This script includes the application for drawing your own images and predicting the output.
- `train.py`: This script train the model on provided dataset.
- `test.py`: This script tests the model performance.

Additionally, the repository includes the `quickdraw_model.h5` which reaches 89% accuracy on the testing set.


## To train the model
```
python train.py
```

## To test the model
```
python test.py
```

## To draw your own sketches and predict the output
```
python main.py
```

## License

This project is licensed under the [MIT LICENSE](LICENSE).
