import json
import random

# Load JSON data from files
with open('mohammed_and_aryan.json', 'r') as file:
    mohammed_and_aryan = json.load(file)

with open('danning_mohab.json', 'r') as file:
    danning_mohab = json.load(file)

with open('ben_saad_data_v3.json', 'r') as file:
    ben_saad_data_v3 = json.load(file)

# Function to trim the audio filenames
def trim_audio_filenames(data):
    for item in data:
        item['audio_files'] = item['audio_files'][:47]
    return data

# Trim audio filenames in each JSON list
mohammed_and_aryan = trim_audio_filenames(mohammed_and_aryan)
danning_mohab = trim_audio_filenames(danning_mohab)
ben_saad_data_v3 = trim_audio_filenames(ben_saad_data_v3)

# Create a dictionary to store all unique audio files
audio_files_dict = {}

# Function to process a single JSON list and add to dictionary
def process_json_list(json_list, annotator):
    for item in json_list:
        audio_file = item['audio_files']
        if audio_file not in audio_files_dict:
            audio_files_dict[audio_file] = []
        audio_files_dict[audio_file].append({
            'intervals': item['intervals'],
            'labels': item['labels'],
            'annotator': annotator
        })

# Process each JSON list
process_json_list(mohammed_and_aryan, 'mohammed_and_aryan')
process_json_list(danning_mohab, 'danning_mohab')
process_json_list(ben_saad_data_v3, 'ben_saad_data_v3')

# Create output list with unique audio files selected randomly from the available options
output_list = []
for audio_file, annotations in audio_files_dict.items():
    random.shuffle(annotations)  # Shuffle the list of annotations to avoid bias
    selected_annotation = random.choice(annotations)
    output_list.append({
        'audio_files': audio_file,
        'intervals': selected_annotation['intervals'],
        'labels': selected_annotation['labels'],
        'annotator': selected_annotation['annotator']
    })

# Ensure output contains all samples exactly once
assert len(output_list) == len(audio_files_dict), "The output JSON does not contain all samples exactly once."

# Ensure output contains all samples exactly once with audio file names
output_audio_files = [item['audio_files'] for item in output_list]
input_audio_files = list(audio_files_dict.keys())
assert set(output_audio_files) == set(input_audio_files), "The output JSON does not contain all samples exactly once."

# Write output to a JSON file
with open('combined_annotations_Approach3.json', 'w') as outfile:
    json.dump(output_list, outfile, indent=4)

print("Output JSON file has been created successfully!")
print("Done for approach THREE!")
