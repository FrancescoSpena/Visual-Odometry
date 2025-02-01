import utils as u 

path = u.generate_path(0)
result = u.extract_measurements(path)

point_id = result['Point_IDs']
actual_id = result['Actual_IDs']
image_x = result['Image_X']
image_y = result['Image_Y']
app = result['Appearance_Features']


for i in range(0, 120):
    path = u.generate_path(i)
    result = u.extract_measurements(path)

    point_id = result['Point_IDs']
    app = result['Appearance_Features']

    len_point_id = len(point_id)
    len_app_id = len(app)

    if(len_point_id != len_app_id):
        print(f"len of point id in frame {i} : {len_point_id}")
        print(f"len of actual id in frame {i} : {len_app_id}")
        print("===========")

