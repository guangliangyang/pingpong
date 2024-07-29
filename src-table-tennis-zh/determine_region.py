import matplotlib.path as mplPath
def determine_region(prev_box):
    x = prev_box[0]  # Extract x-coordinates from the box
    y = prev_box[1]  # Extract y-coordinates from the box

    region_coordinates = {}
    with open('Region_division.txt', 'r') as file:
        for line in file:
            point, coord = line.strip().split(':')
            x_coord, y_coord = map(int, coord.strip().replace('(', '').replace(')', '').split(','))
            region_coordinates[point] = (x_coord, y_coord)

    region_mappings_L = {
        'L1': ['A5', 'B', 'B1', 'A5C1_1'],
        'L2': ['A5C1_1', 'B1', 'B2', 'A5C1_2'],
        'L3': ['A5C1_2', 'B2', 'C', 'C1'],
        'L4': ['A4C2_2', 'A5C1_2', 'C1', 'C2'],
        'L5': ['A3C3_2', 'A4C2_2', 'C2', 'C3'],
        'L6': ['A3C3_1', 'A4C2_1', 'A4C2_2', 'A3C3_2'],
        'L7': ['A3', 'A4', 'A4C2_1', 'A3C3_1'],
        'L8': ['A4', 'A5', 'A5C1_1', 'A4C2_1'],
        'L9': ['A4C2_1', 'A5C1_1', 'A5C1_2', 'A4C2_2'],
    }
    region_mappings_R = {
        'R1': ['A', 'A1', 'A1C5_1', 'D2'],
        'R2': ['D2', 'A1C5_1', 'A1C5_2', 'D1'],
        'R3': ['D1', 'A1C5_2', 'C5', 'D'],
        'R4': ['A1C5_2', 'A2C4_2', 'C4', 'C5'],
        'R5': ['A2C4_2', 'A3C3_2', 'C3', 'C4'],
        'R6': ['A2C4_1', 'A3C3_1', 'A3C3_2', 'A2C4_2'],
        'R7': ['A2', 'A3', 'A3C3_1', 'A2C4_1'],
        'R8': ['A1', 'A2', 'A2C4_1', 'A1C5_1'],
        'R9': ['A1C5_1', 'A2C4_1', 'A2C4_2', 'A1C5_2']
    }

    print("Region Coordinates:")
    for point, coords in region_coordinates.items():
        print(point, ":", coords)

    # Iterate over the region coordinates and check if the point lies within the bounding box
    for region, points in region_mappings_L.items():
        region_polygon = mplPath.Path([region_coordinates[point] for point in points])
        if region_polygon.contains_point((x, y)):
            return region
    for region, points in region_mappings_R.items():
        region_polygon = mplPath.Path([region_coordinates[point] for point in points])
        if region_polygon.contains_point((x, y)):
            return region


    return None  # If the region is not found or point is not within the bounding box, return None

