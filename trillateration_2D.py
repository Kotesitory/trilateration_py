import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

CUTTOF_VAL =.00001

'''
	2-dimensional point container class
	Contains x and y coordinate
'''
class Point2D:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __repr__(self):
		return f"Point2D [{self.x}, {self.y}]"

	def as_numpy(self):
		return np.array([self.x, self.y])

	def __eq__(self, other):
		if isinstance(other, Point2D):
			return other.x == self.x and other.y == self.y

		return False

	def __add__(self, other):
		x = self.x + other.x
		y = self.y + other.y
		return Point2D(x, y)

	def __sub__(self, other):
		x = self.x - other.x
		y = self.y - other.y
		return Point2D(x, y)

'''
	Sensor class contains information about a 
	sensor in 2D space.
'''
class Sensor2D:
	def __init__(self, location, radius, is_ancor = False):
		self.location = location
		self.radius = radius
		self.estimated_location = None
		self.is_ancor = is_ancor
		self.degree = 0

	def __repr__(self):
		return f"Sensor2D {self.location} {self.radius} estimated: {self.estimated_location} {'Ancor' if self.is_ancor else 'Not Ancor'} degree: {self.degree}\n"

	def localization_error(self):
		if self.is_ancor:
			return 0.0

		if self.estimated_location:
			return distance(self.location, self.estimated_location)

		return None

'''
	Circle class representig a circle with
	radius and center Point2D
'''
class Circle:
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

	def __repr__(self):
		return f"Circle {self.center} {self.radius}\n"

"""
	Calculates eucledian distance between two Point2D objects
"""
def distance(point1, point2):
	a = np.array([point1.x, point1.y])
	b = np.array([point2.x, point2.y])
	return np.linalg.norm(b - a)

'''
	Trillaterates an intersection point of 3 circles.
	The circles need not itersect in a single point, however they
	must all itersect somewhere whith each other
'''
def trilaterate_with_noise(c1, c2, c3):
	points = []
	int1 = get_two_circle_intersections(c1, c2)
	int2 = get_two_circle_intersections(c2, c3)
	int3 = get_two_circle_intersections(c1, c3)
	if int1 and int2 and int3:
		points.extend(int1)
		points.extend(int2)
		points.extend(int3)
	else:
		return None

	circles = [c1, c2, c3]
	points = [point for point in points if point_in_circles(point, circles)]
	if len(points) > 0:
		for point in points:
			if point_on_circles(point, circles):
				return point

		result = get_polygon_centroid(points)
		return result
	else:
		return None

'''
	Check is a point is contained in a list of Cicle objects
'''
def point_in_circles(point, circles):
	for c in circles:
		d = distance(point, c.center)
		diff = d - c.radius
		if diff > CUTTOF_VAL:
			return False

	return True

'''
	Check is a point is contained on a list of Cicle objects' perimiter
'''
def point_on_circles(point, circles):
	for c in circles:
		d = distance(point, c.center)
		diff = np.abs(d - c.radius)
		if diff > CUTTOF_VAL:
			return False

	return True

'''
	Calculates centroid point of a polygon
	by averaging all points coordinates
'''
def get_polygon_centroid(points):
	center = Point2D(0.0, 0.0)
	for point in points:
		center.x += point.x
		center.y += point.y

	center.x /= len(points)
	center.y /= len(points)
	return center

'''
	Calculates intersection points of 2 Circle objects
'''
def get_two_circle_intersections(c1, c2):
	x0, y0 = c1.center.x, c1.center.y
	x1, y1 = c2.center.x, c2.center.y
	r0, r1 = c1.radius, c2.radius

	d = np.sqrt((x1-x0)**2 + (y1-y0)**2)

	# non intersecting
	if d > r0 + r1 :
		return None

	# One circle within other
	if d < abs(r0-r1):
		return None

	# coincident circles
	if d == 0 and r0 == r1:
		return None
	else:
		a=(r0**2-r1**2+d**2)/(2*d)
		h=np.sqrt(r0**2-a**2)
		x2=x0+a*(x1-x0)/d   
		y2=y0+a*(y1-y0)/d   
		x3=x2+h*(y1-y0)/d     
		y3=y2-h*(x1-x0)/d 

		x4=x2-h*(y1-y0)/d
		y4=y2+h*(x1-x0)/d

		return [Point2D(x3, y3), Point2D(x4, y4)]

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
		x, y = L * np.random.rand(2)
		if i < ancor_count:
			sensor = Sensor2D(Point2D(x, y), R, True)
			ancor_sensors.append(sensor)
		else:
			sensor = Sensor2D(Point2D(x, y), R, False)
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
		circles = [Circle(ancor.location, add_noise(distance(sensor.location, ancor.location), ancor.radius * Ferr)) for ancor in ancors if distance(sensor.location, ancor.location) <= sensor.radius]
		if len(circles) >= 3:
			circles.sort(key=lambda c: c.radius)
			c1, c2, c3 = circles[:3]
			result = trilaterate_with_noise(c1, c2, c3)
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

			if len(distances) < 3:
				continue

			if heuristic == "degree":
				distances.sort(key= lambda d: d[1].degree)
			else:
				distances.sort(key= lambda d: d[0])

			c1, c2, c3 = [Circle(dist[1].location, dist[0]) for dist in distances[:3]][:3]
			result = trilaterate_with_noise(c1, c2, c3)
			if result:
				sensor.estimated_location = result
				sensor.degree = distances[0][1].degree + distances[1][1].degree + distances[2][1].degree + 1
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
	localized = localize_sensors(ancor_sensors, nancor_sensors, Ferr)
	print(localized)
	print(len(localized))
	errors = [s.localization_error() for s in localized]
	print(np.average(errors))