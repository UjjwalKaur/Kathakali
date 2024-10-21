from flask import Flask,render_template,Response,request,redirect,url_for, flash, jsonify, send_from_directory,send_file

from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import pandas as pd
import os
import csv
import time
from datetime import datetime
import tensorflow as tf
from playsound import playsound
from werkzeug.utils import secure_filename

from moviepy.editor import VideoFileClip
import librosa
import zipfile
from io import BytesIO

'''
Things to add/change
- File Names
- Page names
- Picture
- Aesthetics (optional)
- Emotion csv data for live feed should be collected in 10 second intervals
- Add more specific instructions for the user
- Change how scores are calculated
- Fix video box height and width
'''




app=Flask(__name__)
app.secret_key = 'supersecretkey'

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ts=time.time()
date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")


base_path = 'emotion_data'
if not os.path.exists(base_path):
    os.makedirs(base_path)


studentIdName = './students/student_details.csv'
activity_file="activity_" + date+".csv"
# number_detection = 10
# loading cv model for face detection
facedetect=cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_model = tf.keras.models.load_model('./models/emotion_model.h5')

def activityMarking(collected_student_activity_time,filepath):
    df_activity = pd.read_csv(filepath)
    df_output = pd.concat([df_activity, collected_student_activity_time])
    df_output.to_csv(filepath,index=False)




# Creating csv file 
def create_csv_file(file_path, headers):
    """
    Create a CSV file if it doesn't exist and write headers to it.

    Parameters:
    - file_path: The path to the CSV file.
    - headers: A list of column headers for the CSV file.
    """
    # file_path = os.path.join(outputdir,filename)  

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Create the file and write headers
        with open(file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
        print(f"CSV file '{file_path}' created successfully.")
    else:
        print(f"CSV file '{file_path}' already exists.")

    return file_path


# student_activity_time = {'STUDENT_ID':[],'ACTIVITY':[],'TIME':[]}
# filepath = create_csv_file(f"./Activity/{activity_file}",['STUDENT_ID','ACTIVITY','TIME'])


def live_generate_frames():

    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (1280, 720))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            num_faces = facedetect.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/live_video')
def live_video():
    return Response(live_generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/end_stream')
def end_stream():
    # Redirect to the index page or any other page you desire after the stream ends
    return redirect(url_for('index'))

@app.route('/aefv')
def aefv():
    return render_template('aefv.html')



@app.route('/analyze_video', methods=['GET', 'POST'])
def analyze_video():
    if request.method == 'POST':
        if 'mp4_file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        
        file = request.files['mp4_file']
        
        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)
        
        if file and file.filename.endswith('.mp4'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                emotions = process_downloaded_video(file_path)
                audio_path = extract_audio(file_path)
                audio_times = extract_audio_timestamps(audio_path)

                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emotion.csv')
                    zip_file.write(csv_file_path, os.path.basename(csv_file_path))

                    zip_file.write(audio_path, os.path.basename(audio_path))

                zip_buffer.seek(0)
                
                return send_file(zip_buffer, as_attachment=True, download_name='emotion_and_audio.zip')

            except Exception as e:
                flash(str(e))
                return redirect(request.url)
        else:
            flash("Invalid file format. Please upload an MP4 file.")
            return redirect(request.url)

    return render_template('analyze_video.html')





@app.route('/play_video/<filename>')
def play_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/change_video_source')
def change_video_source():
    source = request.args.get('source', default=0, type=int)
    # Logic to change the video feed source
    # You might need to restart the video stream with the new source
    return jsonify({'status': 'success', 'new_source': source})

def process_downloaded_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 10)  # 10-second interval
    data = []

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        timestamp = frame_pos / fps
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + interval)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES))
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        if len(num_faces) == 0:
            data.append((timestamp, "No face detected"))
        else:
            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                data.append((timestamp, emotion_dict[maxindex]))
                
    emotions = pd.DataFrame(data, columns=['timestamp', 'emotion'])
    # emotions.set_index('timestamp', inplace=True)
    cap.release()

        
    csv_file = os.path.join(base_path, 'emotion.csv')
    # excel_file = os.path.join(base_path, 'emotion.xlsx')

    # Save the emotion data to CSV
    emotions.to_csv(csv_file,index=False)
    print(f"CSV file '{csv_file}' created successfully.")
 
    # Save the emotion data to Excel
    # emotions.to_excel(excel_file)
    # print(f"Excel file '{excel_file}' created successfully.")
        
    return emotions

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = video_path.replace('.mp4', '.mp3')
    video.audio.write_audiofile(audio_path)
    return audio_path

def extract_audio_timestamps(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    return times

def match_emotions_and_audio(emotions_df, audio_times, threshold=0.5):
    matched_emotions = 0
    total_emotions = len(emotions_df)
    
    for idx, row in emotions_df.iterrows():
        timestamp = idx
        if any(abs(timestamp - time) < threshold for time in audio_times):
            matched_emotions += 1

    return matched_emotions, total_emotions

@app.route('/comp')
def comp():
    return render_template('comp.html')


start_time = None
emotions_data = []

@app.route('/comp_live_video')
def comp_live_video():
    global start_time
    start_time = time.time()  
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/play_audio/<filename>')
def play_audio(filename):
    # Play audio and wait for it to finish
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    playsound(audio_path)
    
    # After audio is finished, perform comparison
    comparison_result = perform_comparison()

    result_message = "Good job!" if comparison_result else "Try again!"
    return render_template('comparison_results.html', result_message=result_message)

@app.route('/compare_csv_results')
def compare_csv_results():
    return render_template('compare_results.html')

@app.route('/compare_csv', methods=['POST'])
def compare_csv():
    if 'csv_file_1' not in request.files or 'csv_file_2' not in request.files:
        flash('Please upload both CSV files')
        return redirect(url_for('index'))

    csv_file_1 = request.files['csv_file_1']
    csv_file_2 = request.files['csv_file_2']

    if csv_file_1.filename == '' or csv_file_2.filename == '':
        flash('One or both files were not selected.')
        return redirect(url_for('index'))

    try:
        df1 = pd.read_csv(csv_file_1)
        df2 = pd.read_csv(csv_file_2)

        # Get the number of rows in df1
        num_rows_df1 = df1.shape[0]

        # Trim df2 to have the same number of rows as df1
        df2_trimmed = df2.iloc[:num_rows_df1]

        # Compare df1 and df2_trimmed
        comparison = df1.equals(df2_trimmed)
        if comparison:
            return render_template('compare_results.html', comparison=comparison, differences="Good job! 100% match.")
        else:
            # Calculate the percentage of matching rows
            differences = df1.compare(df2_trimmed)
            num_matching_rows = num_rows_df1 - len(differences)
            match_percentage = (num_matching_rows / num_rows_df1) * 100
            
            if match_percentage > 10:

                 return render_template('compare_results.html', comparison=comparison, difference=f"Good job! {match_percentage:.2f}% match.")
            else:

                return render_template('compare_results.html', comparison=comparison, differences=f"Try again. Only {match_percentage:.2f}% match.")

    except Exception as e:
        flash(f'An error occurred while processing the files: {str(e)}')
        return redirect(url_for('compare_results'))

def gen_frames():
    # Define file paths
 
    global livestarttime, live_emotions_data

    # if livestarttime is None:
    #     return  # No video feed started, exit the generator

    camera = cv2.VideoCapture(0)
    
    livestarttime = time.time()  # Start the video feed with timestamp

    if not camera.isOpened():
        print("Error: Camera not opened.")
        return
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    live_emotions_data = []

    while True:
        success, frame = camera.read()
        
        if not success:
            print("Error: Failed to capture image.")
            break
        
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            emotion = emotion_dict[maxindex]
            cv2.putText(frame, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            end_timestamp = time.time() - livestarttime
            live_emotions_data.append((end_timestamp, emotion))

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    
    camera.release()

def perform_comparison():
    reference_emotions_df = pd.read_csv('reference_emotions.csv')
    emotions_df = pd.DataFrame(emotions_data, columns=['timestamp', 'emotion'])
    matched_emotions, total_emotions = match_emotions_and_audio(emotions_df, reference_emotions_df)

    match_percentage = (matched_emotions / total_emotions) * 100
    return match_percentage >= 10

def match_emotions_and_audio(emotions_df, reference_emotions_df, threshold=0.5):
    matched_emotions = 0
    total_emotions = len(emotions_df)
    
    for idx, row in emotions_df.iterrows():
        timestamp = row['timestamp']
        emotion = row['emotion']
        ref_emotions = reference_emotions_df['emotion'].tolist()
        ref_times = reference_emotions_df['timestamp'].tolist()

        if any(abs(timestamp - ref_time) < threshold and ref_emotion == emotion 
               for ref_time, ref_emotion in zip(ref_times, ref_emotions)):
            matched_emotions += 1

    return matched_emotions, total_emotions


camera_running = False
@app.route('/upload_and_play_audio', methods=['POST'])
def upload_and_play_audio():
    global camera_running
    if 'audio_file' not in request.files:
        flash('No file part')
        return jsonify({'error': 'No file part'})

    audio_file = request.files['audio_file']
    
    if audio_file.filename == '':
        flash('No selected file')
        return jsonify({'error': 'No selected file'})

    if audio_file and allowed_file(audio_file.filename):
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Start the camera feed
        camera_running = True
        video_url = url_for('comp_live_video')

        return jsonify({'file_url': url_for('uploaded_file', filename=filename), 'video_url': video_url})
    else:
        flash('Invalid file type')
        return jsonify({'error': 'Invalid file type'})



@app.route('/stop_video_feed')
def stop_video_feed():
    global camera_running, live_emotions_data
    if camera_running:
        camera_running = False  # Stop the camera feed
        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'emotion_pred.csv')
        live_df_emotions_data = pd.DataFrame(live_emotions_data, columns=['timestamp', 'emotion'])
        live_df_emotions_data.to_csv(csv_file_path, index=False)

        # Clear the live_emotions_data list after saving
        live_emotions_data.clear()

        # Send the CSV file to the user for download
        return send_file(csv_file_path, as_attachment=True, download_name='emotion_pred.csv')

    # If the camera is not running, return a 204 No Content response
    return '', 204



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__=="__main__":
    app.run(debug=True)




