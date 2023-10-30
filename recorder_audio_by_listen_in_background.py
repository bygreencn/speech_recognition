#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import os
from io import BytesIO
import speech_recognition as sr
from queue import Queue
from time import sleep

def main():
    
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    

    try:
        source = sr.Microphone(device_index=10)
    except Exception as err:
        print(err)
        return


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

    index = 0
    while True:
        try:
            if not data_queue.empty():
                last_sample = bytes()
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, 16000, 2, 1)
                with open("microphone-recongnition_{:0>3d}.wav".format(index), "wb") as f:
                    index = index + 1
                    f.write(audio_data.get_wav_data(convert_rate=16000))
                
            # Infinite loops are bad for processors, must sleep.
            sleep(0.25)
        except KeyboardInterrupt:
            break

    # calling this function requests that the background listener stop listening
    stop_listening(wait_for_stop=False)

    # do some more unrelated things
    while True:
        time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping


if __name__ == "__main__":
    main()
