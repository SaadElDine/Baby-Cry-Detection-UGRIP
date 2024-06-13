import json

# Load the JSON files
with open('ben_saad_data_v3.json') as f1, open('danning_mohab.json') as f2, open('mohammed_and_aryan.json') as f3:
    team1 = json.load(f1)
    team2 = json.load(f2)
    team3 = json.load(f3)

# Get the audio file names and trim them to the first 47 characters
def trim_audio_filenames(data):
    for item in data:
        item['audio_files'] = item['audio_files'][:47]
    return data

team1 = trim_audio_filenames(team1)
team2 = trim_audio_filenames(team2)
team3 = trim_audio_filenames(team3)

# Function to order and check missing samples
def check_missing_samples(team_data):
    audio_files = [item['audio_files'] for item in team_data]
    return set(audio_files)

# Order the data by audio file names and print missing samples
def order_and_check_teams(team1, team2, team3):
    team1_files = check_missing_samples(team1)
    team2_files = check_missing_samples(team2)
    team3_files = check_missing_samples(team3)
    
    all_files = sorted(list(team1_files | team2_files | team3_files))
    
    def reorder_data(data, all_files):
        data_dict = {item['audio_files']: item for item in data}
        reordered_data = [data_dict[file] for file in all_files if file in data_dict]
        missing_files = set(all_files) - set(data_dict.keys())
        return reordered_data, missing_files
    
    team1_ordered, team1_missing = reorder_data(team1, all_files)
    team2_ordered, team2_missing = reorder_data(team2, all_files)
    team3_ordered, team3_missing = reorder_data(team3, all_files)
    
    print(f"Team 1 missing samples: {team1_missing}")
    print(f"Team 2 missing samples: {team2_missing}")
    print(f"Team 3 missing samples: {team3_missing}")
    print("------------------------------------------------")
    print("################################################")
    print("------------------------------------------------")
    print("################################################")
    print("------------------------------------------------")

    return team1_ordered, team2_ordered, team3_ordered

team1_ordered, team2_ordered, team3_ordered = order_and_check_teams(team1, team2, team3)

# Function to check if two values are within a given threshold
def values_within_threshold(value1, value2, threshold=0.5):
    return abs(value1 - value2) <= threshold

# Edit distance-like algorithm to align intervals
def align_intervals(intervals1, intervals2, intervals3, threshold=0.025):
    all_intervals = sorted(intervals1 + intervals2 + intervals3)
    aligned_intervals = []
    
    while all_intervals:
        current = all_intervals.pop(0)
        overlapping = [current]

        for other in all_intervals[:]:
            if values_within_threshold(current[0], other[0], threshold) and values_within_threshold(current[1], other[1], threshold):
                overlapping.append(other)
                all_intervals.remove(other)

        avg_start = sum([interval[0] for interval in overlapping]) / len(overlapping)
        avg_end = sum([interval[1] for interval in overlapping]) / len(overlapping)

        if aligned_intervals and aligned_intervals[-1][1] != avg_start:
            avg_start = aligned_intervals[-1][1]  # Ensure the end of the last interval is the start of the new one

        if avg_start < avg_end:  # Ensure the start time is smaller than the end time
            aligned_intervals.append((avg_start, avg_end))

    return aligned_intervals

# Function to align and merge annotations
def align_and_merge_annotations(team1, team2, team3, threshold=0.025):
    combined_annotations = []

    for item1, item2, item3 in zip(team1, team2, team3):
        assert item1['audio_files'] == item2['audio_files'] == item3['audio_files']
        
        audio_file = item1['audio_files']
        intervals1 = item1['intervals']
        intervals2 = item2['intervals']
        intervals3 = item3['intervals']
        
        labels1 = item1['labels']
        labels2 = item2['labels']
        labels3 = item3['labels']
        
        aligned_intervals = align_intervals(intervals1, intervals2, intervals3, threshold)
        combined_labels = []

        for interval in aligned_intervals:
            overlapping_labels = []

            for intervals, labels in zip([intervals1, intervals2, intervals3], [labels1, labels2, labels3]):
                for i, original_interval in enumerate(intervals):
                    if values_within_threshold(interval[0], original_interval[0], threshold) and values_within_threshold(interval[1], original_interval[1], threshold):
                        overlapping_labels.append(labels[i])

            if overlapping_labels:
                final_label = max(set(overlapping_labels), key=overlapping_labels.count)  # Majority vote
            else:
                final_label = '-'
            
            combined_labels.append(final_label)

        combined_annotations.append({
            'audio_files': audio_file,
            'intervals': aligned_intervals,
            'labels': combined_labels
        })

    return combined_annotations

combined_annotations = align_and_merge_annotations(team1_ordered, team2_ordered, team3_ordered, threshold=0.025)

# Save the combined annotations to a new JSON file
with open('combined_annotations_Approach4.json', 'w') as f:
    json.dump(combined_annotations, f, indent=4)
print("Done for approach FOUR!")