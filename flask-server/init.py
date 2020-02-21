from flask import Flask,render_template,send_file
import pymysql
import azure.cognitiveservices.speech as speechsdk
from datetime import date


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download')
def download():
    file_name = "static/files/recording_history.txt"
    attachment_filename = date.today().strftime('%Y-%m-%d') + '.txt'
    return send_file(file_name,
                     attachment_filename=attachment_filename,
                     as_attachment=True)

@app.route('/record')
def record(): 
    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = "3a52eddaf0c745f2858266e6fa82039c", "koreacentral"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Creates a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Say something...")


    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed.  The task returns the recognition text as result. 
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query. 
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()
    print(result)
    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        with open('static/files/recording_history.txt', "w") as test:
            test.write(result.text)
        return render_template('record.html', result = result.text)

    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
        return render_template('index.html', error = 'Please say again')
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))






if __name__ == "__main__":
    app.run()