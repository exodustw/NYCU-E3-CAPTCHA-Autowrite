NYCU E3 CAPTCHA Autowrite Script v1.0

Tested in Chrome 91 with Tampermonkey v4.13
----------------------------------------
Requirement

Chrome extension:
Tampermonkey

Python:
pip install tensorflow
pip install Pillow
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pybase64
pip install Flask
----------------------------------------
Setup

add js file "JS/NYCU E3 CAPTCHA Autowrite.user.js" into Tampermonkey
run python api "Flask API/api.py"
use Chrome and go to URL "https://e3.nycu.edu.tw/login/index.php" (NYCU E3 Login Page)
----------------------------------------
Known Bug (not fix yet)

Sometimes E3 login verification went wrong, although the
prediction of the CAPTCHA is correct. It might be an issue
between HTML with JS.
----------------------------------------
For user who don't have gpu / gpu is disabled:

Disable these lines in "Flask API/cnn_loader.py":
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

Warning: This method can't be used with tensorflow-gpu version.
----------------------------------------
License

Creative Commons License BY-NC-SA	
*DO NOT USE THIS AS YOUR PROGRAMMING CLASS PROJECT/HOMEWORK*
