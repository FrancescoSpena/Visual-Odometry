import utils as u 

path = u.generate_path(0)
result = u.extract_measurements(path)

point_id = result['Point_IDs']
actual_id = result['Actual_IDs']
image_x = result['Image_X']
image_y = result['Image_Y']
app = result['Appearance_Features']

for elem in app: 
    print(len(elem))