#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import os
from io import BytesIO
import speech_recognition as sr
from queue import Queue
from time import sleep
import keyboard
import wave
import argparse


exit_program = False
def exit_handler():
    global exit_program
    exit_program = True
    print("Will Finish recording")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--audio_file", required=True, 
                        help="Write audio file to.", type=str)
    parser.add_argument("-d","--audio_device_index", default=10,
                        help="microphone or loopback device to be listening.", type=int)
    args = parser.parse_args()

    filename = os.path.splitext(args.audio_file.lower())[0]
    filename += ".wav"
    if os.path.exists(filename):
        raise FileExistsError(filename)


    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    
    try:
        source = sr.Microphone(device_index=args.audio_device_index)
    except Exception as err:
        print(err)
        return

    print("*You should play some sound to make code found the corrent ambient_noise*")
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data(convert_rate=16000, convert_width=2)
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=10)

    wavfile = wave.open(filename, 'wb')
    wavfile.setframerate(16000)
    wavfile.setsampwidth(2)
    wavfile.setnchannels(1)
    
    print("\n**You can stop recording by press END key**")
    keyboard.add_hotkey('end', exit_handler)

    while True:
        
        if not data_queue.empty():
            last_sample = bytes()
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, 16000, 2, 1)
            wavfile.writeframes(audio_data.get_wav_data())
        if exit_program:
            break
        # Infinite loops are bad for processors, must sleep.
        sleep(0.25)
        
    keyboard.remove_hotkey(exit_handler)
    wavfile.close()


    # calling this function requests that the background listener stop listening
    stop_listening(wait_for_stop=False)

    # do some more unrelated things
    sleep(2)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping


if __name__ == "__main__":
    main()
