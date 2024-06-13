import torch

# Define the path to the saved model
model_save_path = 'cry_transformer_model_modified.pth'

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CryTransformer(num_classes=2, input_dim=32)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")


import numpy as np

# Get one batch from the test data loader
inputs, labels = next(iter(test_loader))

# Move the inputs and labels to the correct device
inputs, labels = inputs.to(device), labels.to(device)

# Take the first sample from the batch
sample_input = inputs[0].unsqueeze(0)  # Adding batch dimension
sample_label = labels[0]

# Convert the tensor to numpy for further processing
sample_input_np = sample_input.cpu().numpy()
sample_label_np = sample_label.cpu().numpy()

print("Sample input and label extracted!")


with torch.no_grad():
    sample_output = model(sample_input)
    # Flatten the output and labels for evaluation
    sample_output = sample_output.view(-1, sample_output.size(-1))
    _, sample_pred = torch.max(sample_output, 1)

# Convert predictions to numpy
sample_pred_np = sample_pred.cpu().numpy()

print("Predictions obtained from the model!")


def plot_audio_with_labels(signal, sr, ground_truth_labels, predicted_labels, interval_length=0.05, title="Ground Truth and Predicted Labels"):
    # Time axis for the signal
    time_axis = np.arange(len(signal)) / sr
    
    # Time axis for the labels
    label_time_axis = np.arange(len(ground_truth_labels)) * interval_length
    
    plt.figure(figsize=(20, 5))
    plt.plot(time_axis, signal, label='Signal')
    plt.plot(label_time_axis, ground_truth_labels, label='Ground Truth', color='orange')
    plt.plot(label_time_axis, predicted_labels, label='Predicted', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Load the original audio signal corresponding to the sample (assuming the same preprocessing steps)
# Here, we simulate loading and processing the original audio signal for demonstration
# Make sure to replace this with the actual loading and preprocessing of the audio file
sample_audio_path = '/path/to/your/audio/file.wav'  # replace with your actual audio file path
y, sr = librosa.load(sample_audio_path, sr=8000)
y = pad_and_split_audio(y, sr, target_length=8, interval_length=0.05)
y = np.concatenate(y)

# Plot the audio signal with ground truth and predicted labels
plot_audio_with_labels(y, sr, sample_label_np, sample_pred_np, interval_length=0.05)



