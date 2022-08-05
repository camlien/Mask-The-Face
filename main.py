# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import dlib
from utils.aux_functions import *
import glob

# Command-line input setup
parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--path",
    type=str,
    default="",
    help="Path to either the folder containing images or the image itself",
)
parser.add_argument(
    "--mask_type",
    type=str,
    default="random",
    choices=["surgical", "N95", "KN95", "cloth", "gas", "inpaint", "random", "all"],
    help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
)

parser.add_argument(
    "--pattern",
    type=str,
    default="",
    help="Type of the pattern. Available options in masks/textures",
)

parser.add_argument(
    "--pattern_weight",
    type=float,
    default=0.5,
    help="Weight of the pattern. Must be between 0 and 1",
)

parser.add_argument(
    "--color",
    type=str,
    default="#0473e2",
    help="Hex color value that need to be overlayed to the mask",
)

parser.add_argument(
    "--color_weight",
    type=float,
    default=0.5,
    help="Weight of the color intensity. Must be between 0 and 1",
)

parser.add_argument(
    "--code",
    type=str,
    # default="cloth-masks/textures/check/check_4.jpg, cloth-#e54294, cloth-#ff0000, cloth, cloth-masks/textures/others/heart_1.png, cloth-masks/textures/fruits/pineapple.png, N95, surgical_blue, surgical_green",
    default="",
    help="Generate specific formats",
)


parser.add_argument(
    "--verbose", dest="verbose", action="store_true", help="Turn verbosity on"
)
parser.add_argument(
    "--write_original_image",
    dest="write_original_image",
    action="store_true",
    help="If true, original image is also stored in the masked folder",
)
parser.set_defaults(feature=False)

args = parser.parse_args()
args.write_path = args.path + "_masked"

# Set up dlib face detector and predictor
args.detector = dlib.get_frontal_face_detector()
path_to_dlib_model = "F:/GUI-main/GUI-main/MaskTheFace-master/dlib_models/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(path_to_dlib_model):
    download_dlib_model()

args.predictor = dlib.shape_predictor(path_to_dlib_model)

# Extract data from code
mask_code = "".join(args.code.split()).split(",")
args.code_count = np.zeros(len(mask_code))
args.mask_dict_of_dict = {}


for i, entry in enumerate(mask_code):
    mask_dict = {}
    mask_color = ""
    mask_texture = ""
    mask_type = entry.split("-")[0]
    if len(entry.split("-")) == 2:
        mask_variation = entry.split("-")[1]
        if "#" in mask_variation:
            mask_color = mask_variation
        else:
            mask_texture = mask_variation
    mask_dict["type"] = mask_type
    mask_dict["color"] = mask_color
    mask_dict["texture"] = mask_texture
    args.mask_dict_of_dict[i] = mask_dict

# Check if path is file or directory or none
display_MaskTheFace()
print("display")
allID = glob.glob('F:\\GUI-main\\GUI-main\\training_dataset\\*')

folderWrite = 'F:\\GUI-main\\GUI-main\\training_mask\\'


for genID in tqdm(allID):
    id = genID.split('\\')[-1].split('/')[-1]
    #print(id)
    write_path = folderWrite + id
    #print(write_path)
    if not os.path.isdir(write_path):
            os.makedirs(write_path)
    allImage = glob.glob(genID+'\\*.jpg') 
    allImage += glob.glob(genID+'\\*.png') 
    #print(len(allImage),'-------')
    for img in allImage:
        imgName = img.split('\\')[-1].split('/')[-1][:-4]
        masked_image, mask, mask_binary_array, original_image = mask_image(img, args)
        for i in range(len(mask)):
            #w_path = write_path + "/" +imgName+ "_" + mask[i] + "." + img[-3:]
            w_path = write_path + "/" +imgName+ "." + img[-3:]
            cv2.imwrite(w_path,  masked_image[i])
        #print(len(mask))
        #exit()

print("Processing Done")

''''
conda activate py36

'''
