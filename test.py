import numpy as np
from pyknotid.spacecurves import Knot
from pyknotid.visualise import plot_line

points = np.array([[9.0, 0.0, 0.0],
                   [0.781, 4.43, 2.6],
                   [-4.23, 1.54, -2.6],
                   [-4.5, -7.79, -7.35e-16],
                   [3.45, -2.89, 2.6],
                   [3.45, 2.89, -2.6],
                   [-4.5, 7.79, 0.0],
                   [-4.23, -1.54, 2.6],
                   [0.781, -4.43, -2.6]])

# k = Knot(points)
# # k.plot(mode="vispy")
# plot_line(points)

from pyknotid.catalogue.getdb import download_database
download_database()
from pyknotid.make import trefoil
kt = Knot(trefoil())
print(kt.identify())