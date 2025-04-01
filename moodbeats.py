import cv2
import pandas as pd
import webbrowser
import random
import time

def detect_emotion_four(duration=10):
  
    # Haar cascade for face and smile detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    
    # Counters for each emotion
    counts = {"happy": 0, "sad": 0, "energetic": 0, "calm": 0, "angry": 0}
    total_frames = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera access नहीं हो पा रहा!")
        return None

    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        frame_emotion = None
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Check for smile: if a smile is detected, consider the frame as "happy"
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            if len(smiles) > 0:
                frame_emotion = "happy"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Happy", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                break  # Process only first detected face
            
            # If no smile detected, use brightness as heuristic:
            avg_intensity = face_roi.mean()
            # Uncomment the next line for debugging brightness per frame:
            # print("Avg intensity:", avg_intensity)
            if avg_intensity < 80:
                frame_emotion = "sad"
                cv2.putText(frame, "Sad", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            elif avg_intensity > 150:
                frame_emotion = "energetic"
                cv2.putText(frame, "Energetic", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            elif 80 <= avg_intensity <= 120:
                frame_emotion = "calm"
                cv2.putText(frame, "Calm", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,0), 2)
            elif avg_intensity > 120:
                frame_emotion = "angry"
                cv2.putText(frame, "Angry", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            break

        if frame_emotion:
            counts[frame_emotion] += 1

        total_frames += 1
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if total_frames == 0:
        return None

    print("Frame-wise emotion counts:", counts)
    dominant_emotion = max(counts, key=counts.get)
    return dominant_emotion

def select_song(mood):
   
    try:
        df = pd.read_csv("dataset.csv")
        print("Dataset loaded successfully!")
    except Exception as e:
        print("CSV file read करने में error:", e)
        return None

    # Define mood-specific keywords
    mood_keywords = {
        "happy": ["happy", "khushi", "masti", "joy"],
        "sad": ["sad", "dukh", "gham", "blue"],
        "energetic": ["energetic", "dynamic", "zest", "power"],
        "calm": ["calm", "peaceful", "relax", "soothing"],
        "angry": ["angry", "furious", "mad", "rage"],
    }

    if mood in mood_keywords:
        filtered_songs = df[df["Song_Name"].str.lower().apply(
            lambda title: any(kw in title for kw in mood_keywords[mood])
        )]
    else:
        filtered_songs = pd.DataFrame()

    if not filtered_songs.empty:
        return filtered_songs.sample(1).iloc[0]
    else:
        return df.sample(1).iloc[0]

def main():
    # Step 1: Detect overall emotion from webcam feed
    mood = detect_emotion_four(duration=10)
    if mood is None:
        print("Emotion detect नहीं हो पाया, इसलिए random song select होगा।")
        mood = "happy"  # Default mood fallback
    print("Detected Emotion:", mood)
    
    # Step 2: Select a song based on the detected mood from CSV file
    song = select_song(mood)
    if song is not None:
        print(f"Playing: {song['Song_Name']} by {song['Artist']} for mood {mood}")
        webbrowser.open(song["YouTube_URL"])
    else:
        print("Song selection में समस्या है।")

if __name__ == "__main__":
    main()