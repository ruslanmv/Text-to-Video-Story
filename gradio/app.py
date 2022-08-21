# Step 2 - Importing Libraries
from moviepy.editor import *
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
import gradio as gr
import torch, torch.backends.cudnn, torch.backends.cuda
from min_dalle import MinDalle
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
import textwrap
from mutagen.mp3 import MP3
from gtts import gTTS
from pydub import AudioSegment
from os import getcwd
import glob
import nltk
import subprocess
nltk.download('punkt')
description = " Video Story Generator with Audio \n PS:  Generation of video by using Artifical Intellingence by dalle-mini and distilbart and gtss "
title = "Video Story Generator with Audio by using dalle-mini and distilbart and gtss  "
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#def log_gpu_memory():
#    print(subprocess.check_output('nvidia-smi').decode('utf-8'))
#log_gpu_memory()


def get_output_video(text):
  inputs = tokenizer(text, 
                    max_length=1024, 
                    truncation=True,
                    return_tensors="pt")
    
  summary_ids = model.generate(inputs["input_ids"])
  summary = tokenizer.batch_decode(summary_ids, 
                                  skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False)
  plot = list(summary[0].split('.'))

  '''
  The required models will be downloaded to models_root if they are not already there.
  Set the dtype to torch.float16 to save GPU memory.
  If you have an Ampere architecture GPU you can use torch.bfloat16. 
  Set the device to either "cuda" or "cpu". Once everything has finished initializing,
  float32 is faster than float16 but uses more GPU memory.
          
  '''

  def generate_image(
      is_mega: bool,
      text: str,
      seed: int,
      grid_size: int,
      top_k: int,
      image_path: str,
      models_root: str,
      fp16: bool,):
      model = MinDalle(
          is_mega=is_mega, 
          models_root=models_root,
          is_reusable=True,
          is_verbose=True,
          dtype=torch.float16 if fp16 else torch.float32, #param ["float32", "float16", "bfloat16"] 
          #device='cuda' #'cpu' 
      ) 
      #log_gpu_memory()

      image = model.generate_image(
          text, 
          seed, 
          grid_size, 
          top_k=top_k, 
          is_verbose=True
      )

      return image 
      
  generated_images = []
  for senten in plot[:-1]:
    image=generate_image(
      is_mega= True,
      text=senten,
      seed=1,
      grid_size=1, #param {type:"integer"}
      top_k=256, #param {type:"integer"}

      image_path='generated',
      models_root='pretrained',
      fp16=256,)   
    generated_images.append(image)

  # Step 4- Creation of the subtitles
  sentences =plot[:-1]
  num_sentences=len(sentences)
  assert len(generated_images) == len(sentences) , print('Something is wrong')
  #We can generate our list of subtitles
  from nltk import tokenize
  c = 0
  sub_names = []
  for k in range(len(generated_images)): 
    subtitles=tokenize.sent_tokenize(sentences[k])
    sub_names.append(subtitles)

  # Step 5- Adding Subtitles to the Images
  def draw_multiple_line_text(image, text, font, text_color, text_start_height):
      draw = ImageDraw.Draw(image)
      image_width, image_height = image.size
      y_text = text_start_height
      lines = textwrap.wrap(text, width=40)
      for line in lines:
          line_width, line_height = font.getsize(line)
          draw.text(((image_width - line_width) / 2, y_text), 
                    line, font=font, fill=text_color)
          y_text += line_height

  def add_text_to_img(text1,image_input):
      '''
      Testing draw_multiple_line_text
      '''
      image =image_input
      fontsize = 13  # starting font size
      path_font="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
      font = ImageFont.truetype(path_font, fontsize)
      text_color = (255,255,0)
      text_start_height = 200
      draw_multiple_line_text(image, text1, font, text_color, text_start_height)
      return image
      
  generated_images_sub = []
  for k in range(len(generated_images)): 
    imagenes = generated_images[k].copy()
    text_to_add=sub_names[k][0]
    result=add_text_to_img(text_to_add,imagenes)
    generated_images_sub.append(result)
  # Step  7 - Creation of audio 
  c = 0
  mp3_names = []
  mp3_lengths = []
  for k in range(len(generated_images)):
      text_to_add=sub_names[k][0]
      print(text_to_add)
      f_name = 'audio_'+str(c)+'.mp3'
      mp3_names.append(f_name)
      # The text that you want to convert to audio
      mytext = text_to_add
      # Language in which you want to convert
      language = 'en'
      # Passing the text and language to the engine,
      # here we have marked slow=False. Which tells
      # the module that the converted audio should
      # have a high speed
      myobj = gTTS(text=mytext, lang=language, slow=False)
      # Saving the converted audio in a mp3 file named
      sound_file=f_name
      myobj.save(sound_file)
      audio = MP3(sound_file)
      duration=audio.info.length
      mp3_lengths.append(duration)
      print(audio.info.length)
      c+=1
 
  # Step 8 - Merge audio files
  cwd = (getcwd()).replace(chr(92), '/')
  #export_path = f'{cwd}/result.mp3'
  export_path ='result.mp3'
  MP3_FILES = glob.glob(pathname=f'{cwd}/*.mp3', recursive=True)
  silence = AudioSegment.silent(duration=500)
  full_audio = AudioSegment.empty()    # this will accumulate the entire mp3 audios
  for n, mp3_file in enumerate(mp3_names):
      mp3_file = mp3_file.replace(chr(92), '/')
      print(n, mp3_file)

      # Load the current mp3 into `audio_segment`
      audio_segment = AudioSegment.from_mp3(mp3_file)

      # Just accumulate the new `audio_segment` + `silence`
      full_audio += audio_segment + silence
      print('Merging ', n)

  # The loop will exit once all files in the list have been used
  # Then export    
  full_audio.export(export_path, format='mp3')
  print('\ndone!')

  # Step 9 - Creation of the video with adjusted times of the sound
  c = 0
  file_names = []
  for img in generated_images_sub:
    f_name = 'img_'+str(c)+'.jpg'
    file_names.append(f_name)
    img = img.save(f_name)
    c+=1
  print(file_names)
  clips=[]
  d=0
  for m in file_names:
    duration=mp3_lengths[d]
    print(d,duration)
    clips.append(ImageClip(m).set_duration(duration+0.5))
    d+=1
  concat_clip = concatenate_videoclips(clips, method="compose")
  concat_clip.write_videofile("result_new.mp4", fps=24)

  # Step 10 - Merge Video + Audio
  movie_name = 'result_new.mp4'
  export_path='result.mp3'
  movie_final= 'result_final.mp4'

  def combine_audio(vidname, audname, outname, fps=60): 
      import moviepy.editor as mpe
      my_clip = mpe.VideoFileClip(vidname)
      audio_background = mpe.AudioFileClip(audname)
      final_clip = my_clip.set_audio(audio_background)
      final_clip.write_videofile(outname,fps=fps)
  combine_audio(movie_name, export_path, movie_final) # create a new file
  return 'result_final.mp4'
text ='Once, there was a girl called Laura who went to the supermarket to buy the ingredients to make a cake. Because today is her birthday and her friends come to her house and help her to prepare the cake.'
demo = gr.Blocks()
with demo:
    gr.Markdown("# Video Generator from stories with Artificial Intelligence")
    gr.Markdown("A story can be input by user. The story is summarized using DistillBART model. Then, then it is generated the images by using Dalle-mini and created the subtitles and audio gtts. These are generated as a video.")
    with gr.Row():
        # Left column (inputs)
        with gr.Column():
            
            input_start_text = gr.Textbox(value=text, label="Type your story here, for now a sample story is added already!")
            with gr.Row():
                button_gen_video = gr.Button("Generate Video")
        # Right column (outputs)
        with gr.Column():
            output_interpolation = gr.Video(label="Generated Video")
    gr.Markdown("<h3>Future Works </h3>")
    gr.Markdown("This program text-to-video AI software generating videos from any prompt! AI software to build an art gallery. The future version will use Dalle-2 For more info visit [ruslanmv.com](https://ruslanmv.com/) ")
    button_gen_video.click(fn=get_output_video, inputs=input_start_text, outputs=output_interpolation)

demo.launch(debug=False)