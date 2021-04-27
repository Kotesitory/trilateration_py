import numpy as np
import matplotlib.pyplot as plt
import intersect_spheres
from intersect_spheres import SphereOperations

np.random.seed(42)

CUTTOF_VAL =.00001

'''
	3-dimensional point container class
	Contains x, y and z coordinate
'''
class Point3D:
	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	def __repr__(self):
		return f"Point3D [{self.x}, {self.y}, {self.y}]"

	def as_numpy(self):
		return np.array([self.x, self.y, self.z])

	def __eq__(self, other):
		if isinstance(other, Point3D):
			return other.x == self.x and other.y == self.y and other.z == self.z

		return False

	def __add__(self, other):
		x = self.x + other.x
		y = self.y + other.y
		z = self.z + other.z
		return Point3D(x, y, z)

	def __sub__(self, other):
		x = self.x - other.x
		y = self.y - other.y
		z = self.z - other.z
		return Point3D(x, y, z)

'''
	Sensor class contains information about a 
	sensor in 3D space.
'''
class Sensor3D:
	def __init__(self, location, radius, is_ancor = False):
		self.location = location
		self.radius = radius
		self.estimated_location = None
		self.is_ancor = is_ancor
		self.degree = 0

	def __repr__(self):
		return f"Sensor3D {self.location} {self.radius} estimated: {self.estimated_location} {'Ancor' if self.is_ancor else 'Not Ancor'} degree: {self.degree}\n"

	def localization_error(self):
		if self.is_ancor:
			return 0.0

		if self.estimated_location:
			return distance(self.location, self.estimated_location)

		return None

'''
	Sphere class representig a sphere with
	radius and center Point3D
'''
class Sphere:
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def __repr__(self):
		return f"Sphere {self.center} {self.radius}\n"

"""
	Calculates eucledian distance between two Point3D objects
"""
def distance(point1, point2):
	a = np.array([point1.x, point1.y, point1.z])
	b = np.array([point2.x, point2.y, point2.z])
	return np.linalg.norm(b - a)

'''
	Trillaterates an intersection point of 4 spheres.
	The spheres need not itersect in a single point, however they
	must all itersect somewhere whith each other
'''
def trilaterate_with_noise(s1, s2, s3, s4):
	points = []
	int1 = get_three_spheres_intersections(s1, s2, s3)
	int2 = get_three_spheres_intersections(s1, s2, s4)
	int3 = get_three_spheres_intersections(s1, s3, s4)
	int4 = get_three_spheres_intersections(s2, s3, s4)
	if int1 and int2 and int3 and int4:
		points.extend(int1)
		points.extend(int2)
		points.extend(int3)
		points.extend(int4)
	else:
		return None

	spheres = [s1, s2, s3, s4]
	points = [point for point in points if point_in_spheres(point, spheres)]
	if len(points) > 0:
		for point in points:
			if point_on_spheres(point, spheres):
				return point

		result = get_polygon_centroid(points)
		return result
	else:
		return None

'''
	Check is a point is contained in a list of Sphere objects
'''
def point_in_spheres(point, spheres):
	for s in spheres:
		d = distance(point, s.center)
		diff = d - s.radius
		if diff > CUTTOF_VAL:
			return False

	return True

'''
	Check is a point is contained on a list of Sphere objects' surface
'''
def point_on_spheres(point, spheres):
	for s in spheres:
		d = distance(point, s.center)
		diff = np.abs(d - s.radius)
		if diff > CUTTOF_VAL:
			return False

	return True

'''
	Calculates centroid point of a polygon
	by averaging all points coordinates
'''
def get_polygon_centroid(points):
	center = Point3D(0.0, 0.0, 0.0)
	for point in points:
		center.x += point.x
		center.y += point.y
		center.z += point.z

	center.x /= len(points)
	center.y /= len(points)
	center.z /= len(points)
	return center

'''
	Calculates intersection pointes of 3 Spheres objects
	I used: https://github.com/vvhitedog/three_sphere_intersection
'''
def get_three_spheres_intersections(s1, s2, s3):
	spheres = [SphereOperations(s.center.as_numpy(), s.radius) for s in [s1, s2, s3]]
	if not spheres[0].check_intersection(spheres[1]):
		return None
	# Get the circle of intersection of first and second sphere
	circle = spheres[0].get_circle_of_intersection(spheres[1])
	# Get the two points of intersection of the circle with the third sphere
	result = spheres[2].find_intersection_with_circle(circle)
	if result:
		p1 = Point3D(*result[0][:3])
		p2 = Point3D(*result[1][:3])
		return p1, p2

	return None

'''
	L - Width and height of the area where sensors are placed
	N - Number of sensors placed
	R - radio range (radius)
	Fa - Fraction of ancor sensors [0.0, 1.0]
	Ferr - noise ratio of signal [0.0, 1.0]
'''
def generate_sensors(L, N, R, Fa):
	assert(Fa >= 0 and Fa <= 1.0)
	assert(L > 0)
	assert(R > 0)
	assert(N > 0)

	ancor_count = int(N * Fa)
	ancor_sensors = []
	nancor_sensors = []
	for i in range(N):
		x, y, z = L * np.random.rand(3)
		if i < ancor_count:
			sensor = Sensor3D(Point3D(x, y, z), R, True)
			ancor_sensors.append(sensor)
		else:
			sensor = Sensor3D(Point3D(x, y, z), R, False)
			nancor_sensors.append(sensor)

	return ancor_sensors, nancor_sensors

'''
	Adds Gaussian noise to a float number.
'''
def add_noise(d, noise_scale):
	noise = noise_scale * np.random.normal(0.0, 0.3)
	while noise > noise_scale or noise < noise_scale * (-1):
		noise = noise_scale * np.random.normal(0.0, 0.3)

	if d + noise < 0:
		return d

	else:
		return d + noise

'''
	Noniterative Localizaztion algorithm usin 2D trilateration.
	Localizes all non_ancors sensors if possible
'''
def localize_sensors(ancors, non_ancors, Ferr):
	localized = []
	for sensor in non_ancors:
		spheres = [Sphere(ancor.location, add_noise(distance(sensor.location, ancor.location), ancor.radius * Ferr)) for ancor in ancors if distance(sensor.location, ancor.location) <= sensor.radius]
		if len(spheres) >= 4:
			spheres.sort(key=lambda s: s.radius)
			s1, s2, s3, s4 = spheres[:4]
			result = trilaterate_with_noise(s1, s2, s3, s4)
			if result:
				sensor.estimated_location = result
				localized.append(sensor)

	return localized

'''
	Iterative Localizaztion algorithm usin 2D trilateration.
	Localizes all non_ancors sensors if possible
	[heuristic] parameter determines which heuristic is used {"degree", "distance"}
'''
def localize_sensors_iterative(ancors, non_ancors, Ferr, heuristic = "distance"):
	new_ancors = ancors.copy()
	unlocalized = non_ancors.copy()
	previous_len = 0
	while previous_len != len(new_ancors):
		previous_len = len(new_ancors)
		for sensor in [sen for sen in unlocalized if not sen.is_ancor]:
			distances = []
			for ancor in new_ancors:
				d = add_noise(distance(sensor.location, ancor.location), Ferr * sensor.radius)
				if d <= sensor.radius:
					distances.append((d, ancor))

			if len(distances) < 4:
				continue

			if heuristic == "degree":
				distances.sort(key= lambda d: d[1].degree)
			else:
				distances.sort(key= lambda d: d[0])

			s1, s2, s3, s4 = [Sphere(dist[1].location, dist[0]) for dist in distances[:4]][:4]
			result = trilaterate_with_noise(s1, s2, s3, s4)
			if result:
				sensor.estimated_location = result
				sensor.degree = distances[0][1].degree + distances[1][1].degree + distances[2][1].degree + distances[3][1].degree + 1
				new_ancors.append(sensor)

		unlocalized = [sensor for sensor in non_ancors if not sensor.estimated_location]

	localized = [sensor for sensor in new_ancors if not sensor.is_ancor and sensor.estimated_location]
	return localized

if __name__ == '__main__':
	L = 200
	N = 100
	R = 100
	Fa = 0.25
	Ferr = 0.1
	ancor_sensors, nancor_sensors = generate_sensors(L, N, R, Fa)
	localized = localize_sensors_iterative(ancor_sensors, nancor_sensors, Ferr, "degree")
	print(localized)
	print(len(localized))
	errors = [s.localization_error() for s in localized]
	print(np.average(errors))