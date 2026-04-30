import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os.path
import scipy.interpolate
import re
import sys
import pandas as pd

# Sort lines into two groups: full lines and interlinear insertions
def sort_lines(baselines):
    # Look at first x coordinate of each baseline to distinguish
    first_xs = [l.T[0][0] for l in baselines]
    x_diffs = np.diff(first_xs)
    
    # Hardcoding the threshold for an interlinear insertion at a difference of 100 pixels
    # TODO: Replace with some per-document computed quantity
    # NOTE: This method may not properly classify multiple insertions in a single line, or insertions too close to the beginning of a line, and if there is an insertion into the first line of a case, as in JUST1-633m48d, it will declare that a full line.
    full_line_indices = []
    threshold = 100
    full_line_indices.append(0)
    for diff_index, diff in enumerate(x_diffs):
        if diff < threshold:
            full_line_indices.append(diff_index+1)
    
    # Return indices of baselines that fall into each category
    return full_line_indices



def get_line_spacing(baselines):
    center_x = np.median([(l.T[0][0] + l.T[0][-1])/2 for l in baselines])
    
    center_ys = []
    for line in baselines:
        xs, ys = line.T
        center_ys.append(np.interp(center_x, xs, ys))

    med_spacing = np.median(np.diff(center_ys))
    return int(round(med_spacing))

def get_namespace(element):
    m = re.match(r'\{.*\}', element.tag)
    return m.group(0)[1:-1] if m else ''    


def subtract_bkd(image):
    bkd = np.percentile(image, 80, axis=0)
    cols = np.arange(image.shape[1])
    coeffs = np.polyfit(cols, bkd, 2)
    smoothed_bkd = np.polyval(coeffs, cols)
    new_image = np.copy(image) / smoothed_bkd * np.median(smoothed_bkd)
    return new_image

def extract_line_image(image_filename, output_filename, baseline, spacing, scaled_height=128, pad=10, dirname="./"):
    baseline = np.unique(baseline, axis=0) #remove repeat points
    lower_spacing = int(round(0.23 * spacing))
    upper_spacing = int(round(0.77 * spacing))
    height = lower_spacing + upper_spacing
    image = cv2.imread(image_filename, 0)
    xs, ys = baseline.T
    
    #Vertical pad everything to make the math easier
    top_pad = np.ones((spacing, image.shape[1])) * np.percentile(image[0:2], 80)
    bot_pad = np.ones((spacing, image.shape[1])) * np.percentile(image[-2:], 80)
    image = np.vstack((top_pad, image, bot_pad))
    ys += spacing
    
    dense_xs = np.arange(max(0, xs.min() - pad), 
                         min(image.shape[1]-1, xs.max() + pad))
    
    rectified = np.zeros((height, len(dense_xs)))
    dense_ys = scipy.interpolate.interp1d(xs, ys, bounds_error=False, kind="linear", fill_value="extrapolate")(dense_xs)
    
    for i,x in enumerate(dense_xs):
        y_min = int(round(dense_ys[i] - upper_spacing))
        y_max = y_min + height
        rectified[:,i] = image[y_min:y_max, x]

    scaled_width = int(round(scaled_height / height * rectified.shape[1]))
    rectified = subtract_bkd(rectified)
    result = cv2.resize(rectified, (scaled_width, scaled_height), cv2.INTER_CUBIC)
    result = (np.median(result) - result) / (np.percentile(result, 90) - np.percentile(result, 10))
    
    cv2.imwrite(output_filename, 255 * (result - result.min()) / (result.max() - result.min()))


all_line_images = []
all_texts = []
all_xml_files = []
        
for filename in sys.argv[1:]:
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    if dirname == "": dirname = "."
    dirname += "/"

    baselines = []
    tree = ET.parse(filename)
    root = tree.getroot()
    ns = {"ns": get_namespace(root)}
    ET.register_namespace('', ns['ns'])

    image_filename = os.path.join(dirname, root.find('ns:Page', ns).get('imageFilename'))

    # Extract baselines
    for text_region in root.findall('.//ns:TextRegion', ns):
        for lineno, text_line in enumerate(text_region.findall('.//ns:TextLine', ns)):
            baseline = text_line.find('ns:Baseline', ns).get('points')
            baselines.append(np.array([p.split(",") for p in baseline.split(" ")], dtype=int))

    #First iteration: calculate average line spacing for both types of lines
    full_line_baseline_indices = sort_lines(baselines)
    med_spacing_full_line = get_line_spacing([baselines[index] for index in full_line_baseline_indices])
    # TODO: replace this hardcoded ratio with a per-document computed quantity
    med_spacing_insertions = med_spacing_full_line//2

    # Form list of med_spacings that corresponds to the list of baselines
    med_spacings_list = [med_spacing_full_line if index in full_line_baseline_indices else med_spacing_insertions for index, _ in enumerate(baselines)]

    #Second iteration: update bounding boxes
    for text_region in root.findall('.//ns:TextRegion', ns):
        for lineno, (text_line, med_spacing) in enumerate(zip(text_region.findall('.//ns:TextLine', ns), med_spacings_list)):
            baseline = text_line.find('ns:Baseline', ns).get('points')      
            line_im_filename = dirname + "line_{}_{}".format(lineno, os.path.basename(image_filename))
            line_im_filename, _ = os.path.splitext(line_im_filename)
            line_im_filename += ".png"

            #breakpoint()
            #line_im_filename = f"{os.path.splitext(image_filename)[0]}_line_{lineno}.png"
            extract_line_image(image_filename, line_im_filename, baselines[lineno], med_spacing, dirname=dirname)
            all_line_images.append(line_im_filename)
            try:
                text = text_line.find('.//ns:TextEquiv', ns).find('.//ns:Unicode', ns).text.strip()
                text = text.replace(",", ".").replace(";", "") #Replacing unsupported characters
            except AttributeError:
                text = ""
            all_texts.append(text)
            all_xml_files.append(filename)


df = pd.DataFrame(data={"file_name": all_line_images, "xml_filename": all_xml_files, "text": all_texts})
df.to_csv(sys.stdout)
