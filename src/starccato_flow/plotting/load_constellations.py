"""Load custom constellations from .txt file."""

import os

def load_custom_constellations(filepath):
    """Load custom constellations from .txt file.
    
    File format:
        Abbr NumSegments HIP1 HIP2 HIP2 HIP3 HIP3 HIP4 ...
    
    Returns dict:
        {"ConstellationName": [hip1, hip2, hip3, ...], ...}
    """
    constellations = {}
    abbr_to_name = {
        "NZLB": "NZ Long-tailed Bat",
        "Swan": "Swan",
        "Gibb": "Gibbon",
        "EQCr": "Earthquake Crack",
        "Mayf": "Mayfly",
        "Cadd": "Caddyfly",
        "Ston": "Stonefly"
    }
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Custom constellations file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            
            parts = line.split()
            abbr = parts[0]
            num_segments = int(parts[1])
            hips = list(map(int, parts[2:]))
            
            # Convert from overlapping pairs back to sequence
            # File format: HIP1 HIP2 HIP2 HIP3 HIP3 HIP4
            # We want: [HIP1, HIP2, HIP3, HIP4]
            sequence = [hips[0]]  # Start with first HIP
            for i in range(1, len(hips), 2):
                if i + 1 < len(hips):
                    sequence.append(hips[i])
                if i + 1 < len(hips) and hips[i+1] != hips[i]:
                    sequence.append(hips[i+1])
            
            # Use full name if available, otherwise use abbreviation
            const_name = abbr_to_name.get(abbr, abbr)
            constellations[const_name] = sequence
    
    return constellations
