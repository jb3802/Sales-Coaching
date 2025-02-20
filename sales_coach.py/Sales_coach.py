import os
import openai
import speech_recognition as sr
import pandas as pd
from textblob import TextBlob

# Set up OpenAI API key (ensure you have an API key from OpenAI)
openai.api_key = os.getenv("OPENAI_API_KEY")

class SalesCoachingPlatform:
    def __init__(self):
        self.conversations = []  # Store analyzed conversations
        self.feedback = []  # Store AI-generated feedback
    
    def transcribe_audio(self, audio_file):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Error connecting to speech recognition service"
    
    def analyze_text(self, text):
        sentiment = TextBlob(text).sentiment
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sales coaching assistant providing feedback on sales conversations."},
                {"role": "user", "content": text}
            ]
        )
        feedback = response['choices'][0]['message']['content']
        return sentiment, feedback
    
    def process_conversation(self, audio_file):
        transcript = self.transcribe_audio(audio_file)
        sentiment, feedback = self.analyze_text(transcript)
        
        self.conversations.append({
            "transcript": transcript,
            "sentiment": sentiment.polarity,
            "feedback": feedback
        })
        return transcript, sentiment, feedback
    
    def export_results(self, file_name="sales_feedback.csv"):
        df = pd.DataFrame(self.conversations)
        df.to_csv(file_name, index=False)
        print(f"Results exported to {file_name}")

# Example Usage
if __name__ == "__main__":
    sales_coach = SalesCoachingPlatform()
    audio_path = "sample_sales_call.wav"  # Make sure this file exists in your folder
    transcript, sentiment, feedback = sales_coach.process_conversation(audio_path)
    print("Transcript:", transcript)
    print("Sentiment Score:", sentiment.polarity)
    print("AI Feedback:", feedback)
    sales_coach.export_results()
