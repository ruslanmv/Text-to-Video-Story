# Step 1 - Libraries Installation
nvidia-smi
pip install min-dalle
pip install gradio -q
pip install transformers torch requests moviepy huggingface_hub opencv-python
pip install moviepy
pip install imageio-ffmpeg
pip install imageio==2.4.1
apt install imagemagick
cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
pip install gTTS
pip install mutagen
#We reset the runtime in Google Colab
#exit()