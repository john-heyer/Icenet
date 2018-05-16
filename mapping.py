import numpy as np

def map_hexagonal_to_grid():
	height, width = 10, 10
	string_to_coordinates = {}
	string_n = 0

	# Bottom trapezoid
	start_end_x = [4,3,2,1,0, 10, 9, 8, 7, 4]
	for y in range(height/2):
		x_i = start_end_x[y]
		for x in range(x_i, width):
			string_to_coordinates[string_n] = (x, y)
			string_n += 1

	# Top trapezoid
	for y in range(height/2, height):
		x_f = start_end_x[y]
		for x in range(x_f):
			string_to_coordinates[string_n] = (x, y)
			string_n += 1

	string_ind = [string_to_coordinates[key][1]*10 + string_to_coordinates[key][0] for key in string_to_coordinates]
	return string_ind

print(map_hexagonal_to_grid())


