Before using my code do this stuff:

create a new folder and in the cmd change your directory, so that your working directory is this folder

/////////
pip install -r requirements.txt
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights
/////////


Using this website I learnt how to convert the .weights file to a .h5 file (WHOOHOO): https://stackoverflow.com/questions/57229199/how-to-load-darknet-yolov3-model-from-cfg-file-and-weights-from-weights-file

/////////
python.exe convert.py yolov3-tiny.cfg model_data\yolov3-tiny.weights model_data/yolov3-tiny.h5
/////////