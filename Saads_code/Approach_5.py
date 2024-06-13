import json
import numpy as np

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_binary_labels(data, sampling_rate=8000):
    binary_labels = {}
    for entry in data:
        audio_file = entry['audio_files'][:47]
        duration = max([interval[1] for interval in entry['intervals']])
        label_array = np.zeros(int(duration * sampling_rate))

        for interval, label in zip(entry['intervals'], entry['labels']):
            if label == "cry":
                start_idx = int(interval[0] * sampling_rate)
                end_idx = int(interval[1] * sampling_rate)
                label_array[start_idx:end_idx] = 1

        if audio_file not in binary_labels:
            binary_labels[audio_file] = []
        binary_labels[audio_file].append(label_array)

    return binary_labels

def majority_vote(binary_labels):
    majority_labels = {}
    for audio_file, labels_list in binary_labels.items():
        labels_stack = np.stack(labels_list)
        majority_label = np.round(np.mean(labels_stack, axis=0)).astype(int)
        majority_labels[audio_file] = majority_label
    return majority_labels

def convert_to_intervals(binary_label, sampling_rate=8000):
    intervals = []
    current_state = binary_label[0]
    start_idx = 0

    for idx, value in enumerate(binary_label):
        if value != current_state:
            end_idx = idx
            intervals.append((start_idx / sampling_rate, end_idx / sampling_rate, current_state))
            current_state = value
            start_idx = idx

    intervals.append((start_idx / sampling_rate, len(binary_label) / sampling_rate, current_state))
    return [(start, end, int(state)) for start, end, state in intervals]

def process_files(file_paths, sampling_rate=8000):
    combined_binary_labels = {}
    for file_path in file_paths:
        data = read_json(file_path)
        binary_labels = generate_binary_labels(data, sampling_rate)
        
        for audio_file, labels_list in binary_labels.items():
            if audio_file not in combined_binary_labels:
                combined_binary_labels[audio_file] = []
            combined_binary_labels[audio_file].extend(labels_list)
    
    majority_labels = majority_vote(combined_binary_labels)
    intervals_dict = {audio_file: convert_to_intervals(label, sampling_rate) for audio_file, label in majority_labels.items()}

    return intervals_dict

# File paths for the JSON files
file_paths = [
    'mohammed_and_aryan.json',
    'danning_mohab.json',
    'ben_saad_data_v3.json'
]

# Process the files and get the intervals
intervals_dict = process_files(file_paths)

# Prepare the output format
output_data = []
for audio_file, intervals in intervals_dict.items():
    interval_only = [(start, end) for start, end, _ in intervals]
    labels = ["cry" if state == 1 else "-" for _, _, state in intervals]
    output_data.append({
        "audio_files": audio_file,
        "intervals": interval_only,
        "labels": labels
    })




# Save the results to a JSON file
output_file = 'combined_annotations_Approach5.json'
with open(output_file, 'w') as file:
    json.dump(intervals_dict, file, indent=4)

print(f"Combined annotations saved to {output_file}")
