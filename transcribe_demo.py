#! python3.7

import argparse
import io
from io import BytesIO
import os
import speech_recognition as sr
import numpy as np
import soundfile as sf
import faster_whisper


from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import keyboard

exit_program = False
def exit_handler():
    global exit_program
    exit_program = True
    print("Will Finish recording")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--audio_device_index", default=10,
                        help="microphone or loopback device to be listening.", type=int)
    parser.add_argument("--sample_rate", default=None,
                        help="microphone can be set if it supported; Loopback device only can not be set.", type=int)
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=5,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=0,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--gpu", action="store_true",
                        help="process the audio with GPU.")
    parser.add_argument("--language", default="zh",
                        help="the language of the audio.", type=str)
    parser.add_argument("--no_translate", action="store_true",
                        help="Not tanslate the language of audio to English.")
    parser.add_argument("--process_max_buffer_count", default=5,
                        help="the language of the audio.", type=int)

    args = parser.parse_args()

    print("*You should play some sound to make code found the corrent ambient_noise*")
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Prevents permanent application hang and crash by using the wrong Microphone

    audio_device_index = args.audio_device_index
    if audio_device_index is None:
        print("Available microphone devices and Loopback devices are: ")
        microphone_name = sr.Microphone.list_working_microphones()
        for index in microphone_name:
            print(f"Microphone with name \"{microphone_name[index]}(device_index={index})\" found")
        loopback_device_name = sr.Microphone.list_loopback_devices()
        for index in loopback_device_name:
            print(f"Loopback Device with name \"{loopback_device_name[index]}(device_index={index})\" found")
        return
    else:
        try:
            source = sr.Microphone(device_index=audio_device_index, sample_rate=args.sample_rate)
        except Exception as err:
            print(err)
            return


    models_path = "F:\\Src.Disk\\speech_recognition\\speech_recognition\\recognizers\\models"
    
    models = {'base': os.path.join(models_path, "base"),
              'tiny': os.path.join(models_path, "tiny"),
              'small': os.path.join(models_path, "small"),
              'medium': os.path.join(models_path, "medium"),
              'large-v2': os.path.join(models_path, "large-v2"),
              }
    # prepare local model path.
    model_size_or_path = models[args.model]
    beam_size = 5
    if args.gpu:
        device = "cuda",
        compute_type = "float16"
    else:
        device = "auto"
        compute_type = "int8"
    language = args.language
    translate = True if not args.no_translate else False
    
    whisper_model = faster_whisper.WhisperModel(model_size_or_path, device=device, compute_type=compute_type)
    

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    #transcription = ['']

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
    stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    phrase_time = datetime.utcnow()
    print("\n**You can stop recording by press ESC key**")
    keyboard.add_hotkey('ESC', exit_handler)

    while True:

        now = datetime.utcnow()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            print("buffer size={}".format(data_queue.qsize()))
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                    # This is the last time we received new audio data from the queue.
                phrase_time = now
                print(phrase_time)
            else:
                continue

            # Concatenate our current audio data with the latest audio data.
            #last_sample = data_queue.get()
            precess_count = 0
            while not data_queue.empty():
                data = data_queue.get()
                last_sample += data
                precess_count += 1
                if precess_count > args.process_max_buffer_count:
                    break

            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, 16000, 2, 1)
            wav_stream = io.BytesIO(audio_data.get_wav_data())
            audio_array, sampling_rate = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)

            # Read the transcription.
            segments, info = whisper_model.transcribe(
                audio_array,
                beam_size=beam_size,
                language=language,
                task="translate" if translate else "transcribe",
            )
            found_text = list()
            for segment in segments:
                found_text.append(segment.text)
            text = '.'.join(found_text).strip()

            print(text)
            # Clear the console to reprint the updated transcription.
            # transcription.append(text)
            #os.system('cls' if os.name=='nt' else 'clear')
            # for line in transcription:
            #    print(line)
            # Flush stdout.
            print('', end='', flush=True)

        if exit_program:
            break
        # Infinite loops are bad for processors, must sleep.
        sleep(0.25)
    keyboard.remove_hotkey(exit_handler)

    # calling this function requests that the background listener stop listening
    stop_listening(wait_for_stop=False)

    # do some more unrelated things
    sleep(1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping
    #print("\n\nTranscription:")
    #for line in transcription:
    #    print(line)


if __name__ == "__main__":
    main()
