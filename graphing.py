import matplotlib.pyplot as plt
import numpy as np
from trillateration_2D import generate_sensors as generate_sensors_2D, localize_sensors as localize_sensors_2D, \
localize_sensors_iterative as localize_sensors_iterative_2D
from trillateration_3D import generate_sensors as generate_sensors_3D, localize_sensors as localize_sensors_3D, \
localize_sensors_iterative as localize_sensors_iterative_3D

np.random.seed(42)

NUM_OF_ITERATIONS = 15


L = 200
N = 100
Rs = [50, 75, 100, 125, 150]
Fas = [0.1, 0.3, 0.5]
Ferrs = [0.1, 0.2, 0.3, 0.4, 0.50]

def degree_heuristic_2D(ancor_sensors, nancor_sensors, Ferr):
	return localize_sensors_iterative_2D(ancor_sensors, nancor_sensors, Ferr, "degree")

def degree_heuristic_3D(ancor_sensors, nancor_sensors, Ferr):
	return localize_sensors_iterative_3D(ancor_sensors, nancor_sensors, Ferr, "degree")

def distance_heuristic_2D(ancor_sensors, nancor_sensors, Ferr):
	return localize_sensors_iterative_2D(ancor_sensors, nancor_sensors, Ferr, "distance")

def distance_heuristic_3D(ancor_sensors, nancor_sensors, Ferr):
	return localize_sensors_iterative_3D(ancor_sensors, nancor_sensors, Ferr, "distance")

def do_experiments(L, N, R, Fa, Ferr, generation, localization):
	iter_ale = []
	f_localized = []
	for i in range(NUM_OF_ITERATIONS):
		ancor_sensors, nancor_sensors = generation(L, N, R, Fa)
		localized = localization(ancor_sensors, nancor_sensors, Ferr)
		errors = [s.localization_error() for s in localized]
		iter_ale.append(np.average(errors))
		f_localized.append(int(len(localized) / len(nancor_sensors) * 100))

	f_loc = round(np.average(f_localized), 2)
	avg_ale = round(np.average(iter_ale), 2)
	return f_loc, avg_ale

def draw_curves(curves_y, curves_x, x_label, y_label, title, legend_strs, file_name=None):
	colors = ["red", "blue", "green", "purple", "orange"]
	fig, ax = plt.subplots(1)
	for i in range(len(curves_y)):
		ax.plot(curves_x, curves_y[i], color=colors[i], label=legend_strs[i])

	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend(loc="best")
	if file_name:
		fig.savefig(file_name)

	plt.show()


Fl_curves_2D = []
Fl_curves_3D = []
Fl_2D = []
Fl_3D = []
for Fa in Fas:
	Fl_2D = []
	Fl_3D = []
	err_curves = []
	err_curves_ni_2D = []
	err_curves_ni_3D = []
	err_curves_i_dist_2D = []
	err_curves_i_deg_2D = []
	err_curves_i_dist_3D = []
	err_curves_i_deg_3D = []
	for Ferr in Ferrs:
		ales_ni_2D = []
		ales_ni_3D = []
		ales_i_2D_dist = []
		ales_i_2D_deg = []
		ales_i_3D_dist = []
		ales_i_3D_deg = []
		for R in Rs:
			fl_2D, ale_2D_ni = do_experiments(L, N, R, Fa, Ferr, generate_sensors_2D, localize_sensors_2D)
			_, ale_2D_i_dist = do_experiments(L, N, R, Fa, Ferr, generate_sensors_2D, distance_heuristic_2D)
			_, ale_2D_i_deg = do_experiments(L, N, R, Fa, Ferr, generate_sensors_2D, degree_heuristic_2D)
			fl_3D, ale_3D_ni = do_experiments(L, N, R, Fa, Ferr, generate_sensors_3D, localize_sensors_3D)
			_, ale_3D_i_dist = do_experiments(L, N, R, Fa, Ferr, generate_sensors_3D, distance_heuristic_3D)
			_, ale_3D_i_deg = do_experiments(L, N, R, Fa, Ferr, generate_sensors_3D, degree_heuristic_3D)

			Fl_2D.append(fl_2D)
			Fl_3D.append(fl_3D)
			ales_ni_2D.append(ale_2D_ni)
			ales_i_2D_dist.append(ale_2D_i_dist)
			ales_i_2D_deg.append(ale_2D_i_deg)
			ales_ni_3D.append(ale_3D_ni)
			ales_i_3D_dist.append(ale_3D_i_dist)
			ales_i_3D_deg.append(ale_3D_i_deg)

		err_curves_ni_2D.append(ales_ni_2D)
		err_curves_i_dist_2D.append(ales_i_2D_dist)
		err_curves_i_deg_2D.append(ales_i_2D_deg)
		err_curves_ni_3D.append(ales_ni_3D)
		err_curves_i_dist_3D.append(ales_i_3D_dist)
		err_curves_i_deg_3D.append(ales_i_3D_deg)

	draw_curves(err_curves_ni_2D, Rs, "Range", "ALE", f"ALE for Ancor sensor frequency: {int(Fa * 100)}% (Noniterative 2D algorithm)", [f"Noise: {int(fer * 100)}%" for fer in Ferrs], f"graphs/err_curves_ni_2D_Fa_{Fa}.png")
	draw_curves(err_curves_i_dist_2D, Rs, "Range", "ALE", f"ALE for Ancor sensor frequency: {int(Fa * 100)}% \n(Iterative 2D algorithm - distance heuristic)", [f"Noise: {int(fer * 100)}%" for fer in Ferrs], f"graphs/err_curves_i_dist_2D_Fa_{Fa}.png")
	draw_curves(err_curves_i_deg_2D, Rs, "Range", "ALE", f"ALE for Ancor sensor frequency: {int(Fa * 100)}% \n(Iterative 2D algorithm - degree heuristic)", [f"Noise: {int(fer * 100)}%" for fer in Ferrs], f"graphs/err_curves_i_deg_2D_Fa_{Fa}.png")
	draw_curves(err_curves_ni_3D, Rs, "Range", "ALE", f"ALE for Ancor sensor frequency: {int(Fa * 100)}% \n(Noniterative 3D algorithm)", [f"Noise: {int(fer * 100)}%" for fer in Ferrs], f"graphs/err_curves_ni_3D_Fa_{Fa}.png")
	draw_curves(err_curves_i_dist_3D, Rs, "Range", "ALE", f"ALE for Ancor sensor frequency: {int(Fa * 100)}% \n(Iterative 3D algorithm - distance heuristic)", [f"Noise: {int(fer * 100)}%" for fer in Ferrs], f"graphs/err_curves_i_dist_3D_Fa_{Fa}.png")
	draw_curves(err_curves_i_deg_3D, Rs, "Range", "ALE", f"ALE for Ancor sensor frequency: {int(Fa * 100)}% \n(Iterative 3D algorithm - degree heuristic)", [f"Noise: {int(fer * 100)}%" for fer in Ferrs], f"graphs/err_curves_i_deg_3D_Fa_{Fa}.png")
	Fl_curves_2D.append(Fl_2D[:len(Rs)])
	Fl_curves_3D.append(Fl_3D[:len(Rs)])

draw_curves(Fl_curves_2D, Rs, "Range", "Localization freq", f"Localization frequency (Noniterative 2D algorithm)", [f"Ancor feq: {int(fa * 100)}%" for fa in Fas], "graphs/lf_2D.png")
draw_curves(Fl_curves_3D, Rs, "Range", "Localization freq", f"Localization frequency (Noniterative 3D algorithm)", [f"Ancor feq: {int(fa * 100)}%" for fa in Fas], "graphs/lf_3D.png")
print("DONE")
