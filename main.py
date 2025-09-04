import cv2
import os
import numpy as np
import csv
import json
from pathlib import Path

def load_templates(folder_path):
    """Load all template images from a folder"""
    templates = []
    template_files = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Template folder {folder_path} does not exist")
        return templates, template_files
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append(img)
                template_files.append(filename)
                print(f"Loaded template: {filename} - Shape: {img.shape}")
    
    return templates, template_files

def apply_non_max_suppression(matches, overlap_threshold=0.3):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(matches) == 0:
        return []
    
    # Convert matches to format needed for NMS
    boxes = []
    scores = []
    
    for match in matches:
        # Format: [x, y, x+width, y+height]
        box = [match['x'], match['y'], match['x'] + match['width'], match['y'] + match['height']]
        boxes.append(box)
        scores.append(match['confidence'])
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=overlap_threshold)
    
    # Handle different return types from cv2.dnn.NMSBoxes safely
    filtered_matches = []
    if indices is not None and len(indices) > 0:
        # Safely extract indices regardless of format
        indices_flat = []
        for idx in indices:
            if isinstance(idx, (list, tuple, np.ndarray)):
                # Handle case where idx is [0] or (0,) or array([0])
                indices_flat.append(int(idx[0]))
            else:
                # Handle case where idx is already an integer: 0
                indices_flat.append(int(idx))
        
        # Filter matches based on NMS results
        for i in indices_flat:
            if 0 <= i < len(matches):  # Safety check for valid index
                filtered_matches.append(matches[i])
    
    return filtered_matches

def find_template_matches(image, templates, threshold=0.8):
    """Find all matches for templates with improved duplicate filtering"""
    all_matches = []
    max_confidence = 0.0
    
    for i, template in enumerate(templates):
        # Get template dimensions
        h, w = template.shape
        
        # Skip if template is larger than image
        if h > image.shape[0] or w > image.shape[1]:
            continue
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations where matching result is above threshold
        locations = np.where(result >= threshold)
        
        # Store match information with bounding boxes
        template_matches = []
        for pt in zip(*locations[::-1]):  # Switch x and y coordinates
            x, y = pt
            confidence = result[y, x]
            max_confidence = max(max_confidence, confidence)
            
            bbox = {
                'x': int(x),
                'y': int(y), 
                'width': int(w),
                'height': int(h),
                'confidence': float(confidence),
                'template_idx': i
            }
            template_matches.append(bbox)
        
        # Apply NMS to this template's matches
        if template_matches:
            filtered_matches = apply_non_max_suppression(template_matches, overlap_threshold=0.3)
            all_matches.extend(filtered_matches)
            print(f"    Template {i}: {len(template_matches)} raw matches -> {len(filtered_matches)} after NMS")
    
    # Apply final NMS across all templates
    if all_matches:
        final_matches = apply_non_max_suppression(all_matches, overlap_threshold=0.2)
        print(f"    Final: {len(all_matches)} total matches -> {len(final_matches)} after cross-template NMS")
        return final_matches, max_confidence
    
    return [], max_confidence

def format_bounding_boxes(matches):
    """Format bounding boxes as string for CSV storage"""
    if not matches:
        return ""
    
    bbox_strings = []
    for match in matches:
        bbox_str = f"{match['x']},{match['y']},{match['width']},{match['height']}"
        bbox_strings.append(bbox_str)
    
    return ";".join(bbox_strings)

def format_positions_json(matches):
    """Format positions as JSON string for more detailed storage"""
    if not matches:
        return ""
    
    positions = []
    for match in matches:
        position = {
            'x': match['x'],
            'y': match['y'],
            'width': match['width'], 
            'height': match['height'],
            'confidence': round(match['confidence'], 4),
            'center_x': match['x'] + match['width'] // 2,
            'center_y': match['y'] + match['height'] // 2
        }
        positions.append(position)
    
    return json.dumps(positions, separators=(',', ':'))

def crop_and_save_glyphs(screenshot, matches, screenshot_file, output_folder, glyph_type):
    """Crop detected glyphs and save to output folder"""
    cropped_files = []
    
    for idx, bbox in enumerate(matches):
        # Extract crop coordinates
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Crop the glyph area
        cropped_glyph = screenshot[y:y+h, x:x+w]
        
        # Generate filename for cropped glyph
        base_name = Path(screenshot_file).stem
        crop_filename = f"{base_name}_{glyph_type}_{idx:02d}_conf{bbox['confidence']:.3f}.png"
        crop_path = os.path.join(output_folder, crop_filename)
        
        # Save cropped glyph
        success = cv2.imwrite(crop_path, cropped_glyph)
        if success:
            cropped_files.append(crop_filename)
            print(f"    Saved crop: {crop_filename} (pos: {x},{y}, size: {w}x{h})")
        else:
            print(f"    Failed to save: {crop_filename}")
    
    return cropped_files

def process_screenshots():
    """Main function to process all screenshots and save results with bounding boxes and crops"""
    
    # Paths
    templates_e_path = 'templates/e'
    templates_q_path = 'templates/q'
    screenshots_path = 'screenshots'
    output_folder = os.path.join(screenshots_path, 'output')
    results_file = '_results.csv'
    
    print("=== Python Image Classifier with NMS and Bounding Box Detection ===")
    print("Template matching threshold: 0.8 (increased for better precision)")
    print("Non-Maximum Suppression: Enabled")
    print()
    print("Loading templates...")
    
    # Load templates
    templates_e, e_files = load_templates(templates_e_path)
    templates_q, q_files = load_templates(templates_q_path)
    
    print(f"Loaded {len(templates_e)} 'e' templates and {len(templates_q)} 'q' templates")
    
    if len(templates_e) == 0 and len(templates_q) == 0:
        print("Error: No templates loaded. Please check template folders.")
        return
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder, exist_ok=True)
    else:
        print(f"Output folder exists: {output_folder}")
        # Clear existing files in output folder
        for file in os.listdir(output_folder):
            if file.endswith('.png'):
                os.remove(os.path.join(output_folder, file))
        print("Cleared previous output files")
    
    # Check if screenshots folder exists
    if not os.path.exists(screenshots_path):
        print(f"Error: Screenshots folder {screenshots_path} does not exist")
        return
    
    # Get all screenshot files
    screenshot_files = []
    for filename in os.listdir(screenshots_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            screenshot_files.append(filename)
    
    screenshot_files.sort()
    print(f"Found {len(screenshot_files)} screenshots to process")
    
    if len(screenshot_files) == 0:
        print("No screenshot files found in screenshots folder")
        return
    
    results = []
    total_crops_saved = 0
    
    # Process each screenshot
    for screenshot_file in screenshot_files:
        print(f"\nProcessing: {screenshot_file}")
        
        img_path = os.path.join(screenshots_path, screenshot_file)
        screenshot = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if screenshot is None:
            print(f"Warning: Could not load {screenshot_file}")
            continue
        
        print(f"  Screenshot size: {screenshot.shape[1]}x{screenshot.shape[0]}")
        
        # Find matches for 'e' templates
        print("  Searching for 'e' glyphs...")
        e_matches, e_max_conf = find_template_matches(screenshot, templates_e)
        
        # Find matches for 'q' templates  
        print("  Searching for 'q' glyphs...")
        q_matches, q_max_conf = find_template_matches(screenshot, templates_q)
        
        # Determine if glyphs are present
        has_e = len(e_matches) > 0
        has_q = len(q_matches) > 0
        
        # Format bounding box data
        e_bboxes_str = format_bounding_boxes(e_matches)
        q_bboxes_str = format_bounding_boxes(q_matches)
        e_positions_json = format_positions_json(e_matches)
        q_positions_json = format_positions_json(q_matches)
        
        # Crop and save detected glyphs
        print("  Cropping detected glyphs...")
        e_crop_files = crop_and_save_glyphs(screenshot, e_matches, screenshot_file, output_folder, 'e')
        q_crop_files = crop_and_save_glyphs(screenshot, q_matches, screenshot_file, output_folder, 'q')
        
        total_crops_saved += len(e_crop_files) + len(q_crop_files)
        
        result_row = {
            'screenshot': screenshot_file,
            'contains_e': has_e,
            'contains_q': has_q,
            'e_count': len(e_matches),
            'q_count': len(q_matches),
            'e_confidence': round(e_max_conf, 4) if has_e else 0.0,
            'q_confidence': round(q_max_conf, 4) if has_q else 0.0,
            'e_bboxes': e_bboxes_str,
            'q_bboxes': q_bboxes_str,
            'e_positions_json': e_positions_json,
            'q_positions_json': q_positions_json,
            'e_crops': ';'.join(e_crop_files),
            'q_crops': ';'.join(q_crop_files),
            'status': 'processed'
        }
        
        results.append(result_row)
        
        print(f"  RESULTS:")
        print(f"    - Contains 'e': {has_e} (max confidence: {e_max_conf:.4f}, unique detections: {len(e_matches)})")
        print(f"    - Contains 'q': {has_q} (max confidence: {q_max_conf:.4f}, unique detections: {len(q_matches)})")
        print(f"    - Cropped files: {len(e_crop_files)} 'e' + {len(q_crop_files)} 'q'")
    
    # Save results to CSV
    print(f"\nSaving results to {results_file}")
    
    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'screenshot', 'contains_e', 'contains_q', 'e_count', 'q_count',
            'e_confidence', 'q_confidence', 'e_bboxes', 'q_bboxes', 
            'e_positions_json', 'q_positions_json', 'e_crops', 'q_crops', 'status'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    total_screenshots = len(results)
    successful_processed = len([r for r in results if r['status'] == 'processed'])
    screenshots_with_e = len([r for r in results if r['contains_e']])
    screenshots_with_q = len([r for r in results if r['contains_q']])
    total_e_detections = sum([r['e_count'] for r in results])
    total_q_detections = sum([r['q_count'] for r in results])
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total screenshots: {total_screenshots}")
    print(f"Successfully processed: {successful_processed}")
    print(f"Screenshots containing 'e': {screenshots_with_e}")
    print(f"Screenshots containing 'q': {screenshots_with_q}")
    print(f"Total unique 'e' detections: {total_e_detections}")
    print(f"Total unique 'q' detections: {total_q_detections}")
    print(f"Total cropped files saved: {total_crops_saved}")
    print(f"Results saved to: {results_file}")
    print(f"Cropped glyphs saved to: {output_folder}")

if __name__ == "__main__":
    try:
        process_screenshots()
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
