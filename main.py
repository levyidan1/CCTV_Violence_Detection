"""
CCTV Violence Detection System for Children's Hospitals

This project was done as part of the CS Hackathon 2023: https://www.cshack-technion.com/ which took place on 30-31/03/2023.
The hackathon's aim was to improve the experience of hospital staff involved in the care of children and enhance the well-being of children and their families during their hospital stay.
This project won the 1st place.

This is the main file of the project. It runs the video and audio analysis in parallel and displays the results in a GUI.
Install the conda environment using the command: conda create --name CCTV_Violence_Detection --file requirements.txt
Insert the livestream URL, the OpenAI organization key and api key in the relevant constants.
Change the constants in the beginning of the file to change the behavior of the system. 
We used an android phone as a CCTV camera using the IP Webcam app.

Architecture:
    A live stream is taken from a CCTV camera.
    The live stream is split into audio and video, and each is analyzed separately, producing a score.
    The individual scores are combined to produce a final score.

    Audio Analysis:
        - The audio analysis is done in a separate thread. It takes the audio from the livestream and saves it to a temporary file.
        - The temporary file is then sent to the audio analysis function, which transcribes the audio, translates it to English, and analyzes the sentiment.
        - If the sentiment is negative, an emotion classifier is used to determine the emotion (anger or fear) and the sentiment score is set to the maximum of the two.
        - The sentiment score is then added to the average volume of the audio segment and a final score is calculated.

    Video Analysis:
        - The video analysis is done in a separate thread. It takes the video from the livestream and analyzes it every FRAME_FREQUENCY seconds.
        - The video is analyzed using the CLIP model, which is a vision transformer model that takes as input an image and a list of text prompts, and outputs a probability for each prompt.
        - The prompts used in this project are 'violent scene', 'non-violent scene'.
        - The probability of the 'violent scene' prompt is multiplied by the VIDEO_WEIGHT.
    
    GUI:
        - The GUI is done using tkinter.
        - The GUI displays the video from the livestream.
        - If the final score is above HIGH_PROB_THRESHOLD, the background of the video is set to red.
        - If the final score is above MEDIUM_PROB_THRESHOLD, the background of the video is set to yellow.
        - If the final score is below MEDIUM_PROB_THRESHOLD, the background of the video is set to green.

    Authors:
        - Idan Levy
        - Nadav Rubinstein
        - Gil Litvin
        - Edo Cohen

"""


import threading
import queue
import time
import cv2
import time
import torch
from transformers import pipeline
from googletrans import Translator
import openai
import librosa
import numpy as np
import pyaudio
import wave
import time
import requests
import os
from pydub import AudioSegment
import tkinter as tk
from PIL import ImageTk, Image

from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
audio_analysis_device = 0 if torch.cuda.is_available() else -1
print(f'Using device: {device}')

# Constants
HIGH_PROB_THRESHOLD = 0.8
MEDIUM_PROB_THRESHOLD = 0.6
VIDEO_WEIGHT = 0.8 # weight of video analysis in the final score

FRAMES_TO_ANALYZE = 5 # number of frames to analyze in each iteration
FRAME_FREQUENCY = 0.4 # seconds
VIDEO_LIVESTREAM_URL = "INSERT YOUR VIDEO LIVESTREAM URL HERE"

AUDIO_LIVESTREAM_URL = "INSERT YOUR AUDIO LIVESTREAM URL HERE"
AUDIO_LANGUAGE = "he"
SENTIMENT_WEIGHT = 0.5 # weight of sentiment analysis in the final score. The rest is the audio volume
KID_REDUCTION_FACTOR = 0.5 # the final score is multiplied by this factor if a kid is detected
AUDIO_SEGMENT_LENGTH = 1 # seconds
KID_FREQUENCY_THRESHOLD = 250 # Hz
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
openai.organization = "INSERT YOUR ORGANIZATION HERE"
openai.api_key = "INSERT YOUR API KEY HERE"

UI_COLORS = ['#48ff00', '#f6ff00', '#ff0000'] # green, yellow, red
ALERT_LENGTH = 5 # If an alert is raised, it will be displayed for this amount of seconds


output_queue = queue.Queue()

def get_frame_from_live_stream(url, frame_number):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = cap.read()
        cap.release()
        return frame

def video_analysis():
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    
    def get_probabilities_for_frame(image, labels=['violent scene', 'non-violent scene']):
        inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        probs_dict = {labels[i]: probs[0][i].item() for i in range(len(labels))}
        return probs_dict

    last_scores = []
    while True:
        try:
            frame = get_frame_from_live_stream(VIDEO_LIVESTREAM_URL, 0)
            image = Image.fromarray(frame)
            image = image.convert("RGB")
            probs = get_probabilities_for_frame(image)
            violent_probability = probs["violent scene"]
        except Exception as e:
            violent_probability = 0
        
        last_scores.append(violent_probability)
        if len(last_scores)>FRAMES_TO_ANALYZE:
            last_scores = last_scores[1:]
    
        final_score = max(last_scores)
        output_queue.put((0, final_score))
        time.sleep(FRAME_FREQUENCY)

def audio_analysis():
    # Initialize sentiment analysis and emotion classification pipelines
    sentiment_pipeline = pipeline("sentiment-analysis", device=audio_analysis_device)
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=audio_analysis_device)

    def recording_kid_or_adult(file_path):
        # Load audio file
        audio_file = file_path
        y, _ = librosa.load(audio_file)
        
        # Extract pitch (fundamental frequency)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        
        # Print fundamental frequency
        clean_f0 = [x for x in f0 if str(x) != 'nan']
        if(np.average(clean_f0) > KID_FREQUENCY_THRESHOLD):
            return 'kid'
        else:
            return 'adult'
        
    def classify_audio(file_path):
        # Transcribe audio and translate to English
        audio_file= open(file_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, language=AUDIO_LANGUAGE)
        text = transcript.text
        translator = Translator()
        try:
            translation = translator.translate(text, dest='en', src=AUDIO_LANGUAGE)
        except Exception as e:
            return -1
        
        # Perform sentiment analysis and emotion classification
        basic_sentiment = sentiment_pipeline(translation.text)[0]['label']
        if basic_sentiment == 'NEGATIVE':
            prediction = classifier(translation.text)[0]
            sentiment_value = max([prediction[3]['score'], prediction[4]['score']]) # max(anger_score, fear_score)
        else:
            sentiment_value = 0
        
        # Calculate volume score
        audio_file = AudioSegment.from_file(file_path, format="mp3")
        average_volume = audio_file.dBFS
        adj_average_volume = (average_volume + 60) / 60
        
        # Compute final score
        final_score = SENTIMENT_WEIGHT*sentiment_value + (1-SENTIMENT_WEIGHT)*adj_average_volume

        if recording_kid_or_adult(file_path) == 'kid':
            return final_score*KID_REDUCTION_FACTOR
        
        return final_score

    
    # Initialize PyAudio and set up temporary audio file
    p = pyaudio.PyAudio()
    temp_audio_file = wave.open('temp_audio.wav', 'wb')
    temp_audio_file.setnchannels(CHANNELS)
    temp_audio_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
    temp_audio_file.setframerate(RATE)
    interval_start_time = time.time()

    # Continuously fetch audio data and classify segments
    try:
        with requests.get(AUDIO_LIVESTREAM_URL, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=CHUNK):
                # Write audio data to temporary file
                temp_audio_file.writeframes(chunk)
                # Check if AUDIO_SEGMENT_LENGTH second interval has passed
                if time.time() - interval_start_time >= AUDIO_SEGMENT_LENGTH:
                    # Close temporary audio file
                    temp_audio_file.close()
                    # Classify audio segment
                    classification = classify_audio('temp_audio.wav')
                    output_queue.put((1, classification))
                    # Re-open temporary audio file for next segment
                    temp_audio_file = wave.open('temp_audio.wav', 'wb')
                    temp_audio_file.setnchannels(CHANNELS)
                    temp_audio_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
                    temp_audio_file.setframerate(RATE)
                    # Set new interval start time
                    interval_start_time = time.time()
    except KeyboardInterrupt:
        temp_audio_file.close()
        os.remove('temp_audio.wav')

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.final_prob = 0
        self.last_probs = [None, None]
        self.frame_number = 0
        self.last_high = time.time()-ALERT_LENGTH
        self.last_medium = time.time()-ALERT_LENGTH
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Load the images
        img1 = Image.open("cctv.jpg")
        img2 = Image.open("cctv.jpg")
        img3 = Image.open("hospital_floor.jpg")

        # Resize the images
        img1 = img1.resize((640, 360), Image.ANTIALIAS)
        img2 = img2.resize((640, 360), Image.ANTIALIAS)
        img3 = img3.resize((640, 360), Image.ANTIALIAS)

        # Create Tkinter image objects
        self.tkimg1 = ImageTk.PhotoImage(img1)
        self.tkimg2 = ImageTk.PhotoImage(img2)
        self.tkimg3 = ImageTk.PhotoImage(img3)

        # Create three labels to hold the images
        self.label0 = tk.Label(self, image=self.tkimg3)
        self.video_label = tk.Label(self, image=self.tkimg1)
        self.label2 = tk.Label(self, image=self.tkimg2)

        # Pack the labels onto the window
        self.label0.pack(fill=tk.X)
        self.label2.pack(side=tk.LEFT)
        self.video_label.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, width=265, height=93, bg='#48ff00', highlightthickness=0)
        self.canvas.place(x=652, y=90)

        # Start the analysis threads and update the UI
        self.start_analysis()
        self.update_ui()

    def start_analysis(self):
        # Start the analysis threads
        self.t1 = threading.Thread(target=video_analysis)
        self.t2 = threading.Thread(target=audio_analysis)

        self.t1.start()
        self.t2.start()

    def update_color(self):
        if self.final_prob>=HIGH_PROB_THRESHOLD:
            next_color = UI_COLORS[2]
            self.canvas.create_text(130, 45, text='ALERT', font=("Helvetica", 30), fill='black')
        elif self.final_prob>=MEDIUM_PROB_THRESHOLD:
            next_color = UI_COLORS[1]
            self.canvas.create_text(130, 45, text='ALERT', font=("Helvetica", 30), fill='black')
        else:
            next_color = UI_COLORS[0]
            self.canvas.delete("all")
        self.canvas.config(bg=next_color)
        self.video_label.config(bg=next_color,highlightbackground=next_color, highlightthickness=5)

    def update_ui(self):
        # Get output from the queue
        try:
            output = output_queue.get_nowait()
            self.last_probs[output[0]] = output[1]
        except queue.Empty:
            # If there's no output in the queue, just continue the loop
            pass

        self.final_prob = 0
        if self.last_probs[0] is not None and self.last_probs[0] >= 0:
            if self.last_probs[1] is not None and self.last_probs[1] >= 0:
                self.final_prob = VIDEO_WEIGHT*self.last_probs[0] + (1-VIDEO_WEIGHT)*self.last_probs[1]
            if self.last_probs[1] is None or self.last_probs[1] < 0:
                self.final_prob = self.last_probs[0]
        elif self.last_probs[1] is not None and self.last_probs[1] >= 0:
            self.final_prob = self.last_probs[1]
        else:
            self.final_prob = 0

        frame = get_frame_from_live_stream(VIDEO_LIVESTREAM_URL, self.frame_number)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_number += 1

        if self.final_prob > HIGH_PROB_THRESHOLD:
            self.last_high = time.time()
        elif self.final_prob > MEDIUM_PROB_THRESHOLD:
            self.last_medium = time.time()

        high_diff = time.time()-self.last_high
        medium_diff = time.time()-self.last_medium
        if high_diff<ALERT_LENGTH:
            self.final_prob = HIGH_PROB_THRESHOLD
            self.update_color()
        elif medium_diff<ALERT_LENGTH:
            self.final_prob = MEDIUM_PROB_THRESHOLD
            self.update_color()
        else:
            self.update_color()
        print(f'Video Score: {self.last_probs[0]}, Audio Score: {self.last_probs[1]}, Final Score: {self.final_prob}')

        # Update the video label
        img = Image.fromarray(frame)
        img = img.resize((640, 360), Image.ANTIALIAS)

        tkimg = ImageTk.PhotoImage(img)

        self.video_label.configure(image=tkimg)
        self.video_label.image = tkimg

        # Schedule the next UI update
        self.after(30, self.update_ui)


# Create the main window
root = tk.Tk()
root.title("CCTV Violence Detection Application")

# Instantiate and start the application
app = Application(master=root)
app.mainloop()

# Wait for both threads to finish
app.t1.join()
app.t2.join()