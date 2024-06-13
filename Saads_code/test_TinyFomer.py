import pyaudio
import threading
import time
import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import BertModel, BertConfig

# Define the CryTinyFormer model
class CryTinyFormer(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_heads=4, num_layers=2, max_len=160):
        super(CryTinyFormer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Load a pre-trained small BERT model from Hugging Face
        config = BertConfig.from_pretrained("prajjwal1/bert-tiny")
        self.tiny_bert = BertModel.from_pretrained("prajjwal1/bert-tiny", config=config)
        
        self.fc = nn.Linear(config.hidden_size, 1)  # Output a single value for binary classification
        self.max_len = max_len

    def forward(self, x):
        batch_size, num_intervals, seq_length, feature_dim = x.size()
        x = x.view(batch_size * num_intervals, seq_length, feature_dim)
        
        x = self.input_projection(x)
        
        # Handle padding and create attention masks
        input_ids = torch.zeros(batch_size * num_intervals, self.max_len, dtype=torch.long).to(x.device)
        attention_mask = torch.zeros(batch_size * num_intervals, self.max_len, dtype=torch.long).to(x.device)
        
        for i in range(batch_size * num_intervals):
            input_len = min(seq_length, self.max_len)
            input_ids[i, :input_len] = torch.arange(1, input_len + 1)  # Dummy token IDs
            attention_mask[i, :input_len] = 1

        transformer_out = self.tiny_bert(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        
        out = transformer_out[:, 0, :]
        out = self.fc(out)
        
        out = out.view(batch_size, num_intervals, 1)
        return out
    
    # Utility functions for audio processing
def pad_and_split_audio(y, sr, target_length=8, interval_length=0.05):
    target_samples = target_length * sr
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    else:
        y = y[:target_samples]
    interval_samples = int(sr * interval_length)
    intervals = [y[i:i + interval_samples] for i in range(0, len(y), interval_samples)]
    return intervals

def extract_mel_spectrogram(y, sr=8000, n_mels=32, n_fft=64, hop_length=8):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram.T


# Listener class to capture audio from the microphone
class Listener:
    def __init__(self, sample_rate=8000, record_seconds=8):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nCry Detection Engine is now listening... \n")

# CryDetectionEngine class to handle real-time inference
class CryDetectionEngine:
    def __init__(self, model_file, sample_rate=8000, record_seconds=2):
        self.listener = Listener(sample_rate=sample_rate, record_seconds=record_seconds)
        self.model = CryTinyFormer(input_dim=32)
        self.model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        self.model.eval().to('cpu')
        self.sample_rate = sample_rate
        self.audio_q = list()

    def predict(self, audio):
        with torch.no_grad():
            buffer = np.concatenate([np.frombuffer(a, dtype=np.int16) for a in audio])
            waveform = buffer.astype(np.float32) / 32768.0

            if np.mean(np.abs(waveform)) < 0.0001:
                return None

            if len(waveform) != self.sample_rate * self.listener.record_seconds:
                waveform = np.pad(waveform, (0, max(0, self.sample_rate * self.listener.record_seconds - len(waveform))), 'constant')[:self.sample_rate * self.listener.record_seconds]

            intervals = pad_and_split_audio(waveform, self.sample_rate, target_length=self.listener.record_seconds, interval_length=0.05)
            feature_sequence = [extract_mel_spectrogram(interval, sr=self.sample_rate) for interval in intervals]

            max_len = 160
            padded_features = []
            for feature in feature_sequence:
                if feature.shape[0] < max_len:
                    pad_width = max_len - feature.shape[0]
                    feature = np.pad(feature, ((0, pad_width), (0, 0)), mode='constant')
                else:
                    feature = feature[:max_len, :]
                padded_features.append(feature)

            padded_features = np.array(padded_features)
            feature_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0)

            outputs = self.model(feature_tensor)
            outputs = outputs.view(-1)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            return preds.mean().item()

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 10:
                diff = len(self.audio_q) - 10
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 10:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=True)
        thread.start()

# DemoAction class to define the action taken when baby cry is detected
class DemoAction:
    def __init__(self, sensitivity=10):
        self.detect_in_row = 0
        self.sensitivity = sensitivity

    def __call__(self, prediction):
        if prediction is None:
            return

        if prediction >= 0.5:
            self.detect_in_row += 1
            if self.detect_in_row >= self.sensitivity:
                self.alert()
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0

    def alert(self):
        print("Baby Crying detected!")

# Main script to load the model and run the detection
if __name__ == "__main__":
    model_file = 'cry_tinyformer_model.pth'  # Update model file name
    cry_detection_engine = CryDetectionEngine(model_file=model_file)
    action = DemoAction(sensitivity=5)
    cry_detection_engine.run(action)
    threading.Event().wait()
