import json
from collections import Counter

def trim_audio_filenames(data):
    for item in data:
        item['audio_files'] = item['audio_files'][:47]
    return data

def merge_intervals(intervals):
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    merged = []
    
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged

def get_overlapping_labels(interval, intervals, labels):
    overlapping_labels = []
    for i, inter in enumerate(intervals):
        if inter[0] < interval[1] and inter[1] > interval[0]:
            overlapping_labels.append(labels[i])
    return overlapping_labels

def consensus_labels(labels):
    label_count = Counter(labels)
    return label_count.most_common(1)[0][0]

def merge_annotations_consensus(annotation_files):
    merged_annotations = []
    
    for audio_file in annotation_files[0]:
        all_intervals = []
        all_labels = []
        
        # Collect all intervals and labels from each annotator
        for team_annotations in annotation_files:
            for annotation in team_annotations:
                if annotation['audio_files'] == audio_file:
                    all_intervals.extend(annotation['intervals'])
                    all_labels.extend(annotation['labels'])
                    break
        
        # Merge intervals
        merged_intervals = merge_intervals(all_intervals)
        
        # Assign labels to merged intervals
        final_intervals = []
        final_labels = []
        for interval in merged_intervals:
            overlapping_labels = get_overlapping_labels(interval, all_intervals, all_labels)
            final_label = consensus_labels(overlapping_labels)
            final_intervals.append(interval)
            final_labels.append(final_label)
        
        merged_annotations.append({
            'audio_file': audio_file,
            'intervals': final_intervals,
            'labels': final_labels
        })
    
    return merged_annotations

# Load annotations from JSON files
with open('mohammed_and_aryan.json') as f:
    team1 = json.load(f)
with open('danning_mohab.json') as f:
    team2 = json.load(f)
with open('ben_saad_data_v3.json') as f:
    team3 = json.load(f)

# Trim audio filenames
team1 = trim_audio_filenames(team1)
team2 = trim_audio_filenames(team2)
team3 = trim_audio_filenames(team3)

annotation_files = [team1, team2, team3]

# Merge annotations using consensus intervals approach
merged_annotations = merge_annotations_consensus(annotation_files)

# Save merged annotations to a new JSON file
with open('combined_annotations_Approach2.json', 'w') as f:
    json.dump(merged_annotations, f, indent=2)

print("Merged annotations have been saved to 'combined_annotations_Approach2.json'.")
print("Done for approach TWO!")
