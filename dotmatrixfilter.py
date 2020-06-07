"""
================
RGB to grayscale
================

This example converts an image with RGB channels into an image with a single
grayscale channel.

The value of each grayscale pixel is calculated as the weighted sum of the
corresponding red, green and blue pixels as::

        Y = 0.2125 R + 0.7154 G + 0.0721 B

These weights are used by CRT phosphors as they better represent human
perception of red, green and blue than equal weights. [1]_

References
----------
.. [1] http://poynton.ca/PDFs/ColorFAQ.pdf

"""
import matplotlib.pyplot as plt
import numpy as np
import sys

from skimage.filters import unsharp_mask
from skimage.filters import gaussian
from skimage import io
from skimage import data
from skimage.color import rgb2gray

sourcefolder = "source"
processedfolder = "processed"
filename = ""
outputname = ""

sw = 0
ol = 0
th = 0

adj1 = 45
adj2 = .265
contrastmod = 1.0

# Parse command line args
if(len(sys.argv) > 1):
	filename = str(sys.argv[1])
if(len(sys.argv) > 2):
	th = float(sys.argv[2])
if(len(sys.argv) > 3):
	contrastmod = float(sys.argv[3])
if(len(sys.argv) > 4):
	adj1 = float(sys.argv[4])
if(len(sys.argv) > 5):
	adj2 = float(sys.argv[5])
if(len(sys.argv) > 6):
	sw = int(sys.argv[6])
if(len(sys.argv) > 7):
	ol = int(sys.argv[7])
if(len(sys.argv) > 8):
	outputname = str(sys.argv[8])
if(len(sys.argv) > 9):
	sourcefolder = str(sys.argv[9])
if(len(sys.argv) > 10):
	processedfolder = str(sys.argv[10])

# Create output filename
def parsefile(filepath):
	a = filepath.split('.')
	return a[0] + "_" + str(th) + "_" + str(adj1) + "_" + str(adj2) + "_" + str(contrastmod) + "." + a[1]

outputname = parsefile(filename)

# Calculate intensity of a pixel based on average pixel intensity of sample square and distance from center
def validpoint(indices, samplewidth, ratio, r, threshold):

	# Distance from center
	d = np.sqrt((np.abs(indices[0] - samplewidth/2.0)**2) + (np.abs(indices[1] - samplewidth/2.0)**2))

	# Extent based on average pixel intensity of sample square + brightness modifier
	extent = (max(0,ratio+threshold)) * r
	if extent <= 1: return 0

	# Establish where pixel intensity should fall off
	minfalloff = extent*.5
	if d < minfalloff:
		return 1.0
	else:
		d -= minfalloff
		return max(0, 1 - d/(extent-minfalloff))

# Dot matrix filter an input image(arr)
# Assumes input image (arr) is in grayscale
def dotmatrixfilter(arr, samplewidth, overlap, threshold, contrast = 1.0):

	# Initialize default values
	if(samplewidth == 0):
		samplewidth = int(len(arr[0])/adj1)
	if(overlap == 0):
		overlap = int((np.sqrt(2*(samplewidth/2.0)**2)) - (samplewidth/2.0))
		overlap = int(samplewidth * adj2)
	r = samplewidth/2.0
	margin = (samplewidth-(overlap*2))
	w = 1 + int((len(arr)-samplewidth)/margin)
	h = 1 + int((len(arr[0])-samplewidth)/margin)

	# Create duplicate image
	dotmatrix = arr.copy()

	# Chunking for optimization
	chunkedarray = []
	chunkedwidths = [samplewidth-overlap, overlap] + [samplewidth-overlap*2, overlap]*w-1
	chunkedheights = [samplewidth-overlap, overlap] + [samplewidth-overlap*2, overlap]*h-1
	cumulativewidth = 0
	cumulativeheight = 0
	for height in chunkedheights:
		chunkedrow = []
		for width in chunkedwidths:
			total = 0
			for i in range(height):
				y = cumulativeheight + i
				for j in range(width):
					x = cumulativewidth + j
					total += arr[x][y]
			cumulativewidth += width
			chunkedrow += [total]
		cumulativeheight += height
		chunkedarray += [chunkedrow]

	print("chunks: ")
	print(chunkedwidths)
	print(chunkedheights)
	print(chunkedarray)

	# Debug info
	print("samplewidth: " + str(samplewidth))
	print("overlap: " + str(overlap))
	print("r: " + str(r))
	print("margin: " + str(margin))
	print("w: " + str(w))
	print("h: " + str(h))
	print("len(dotmatrix): " + str(len(dotmatrix)))
	print("len(dotmatrix[0]): " + str(len(dotmatrix[0])))

	# Clear out arr
	for i in range(len(arr)):
		for j in range(len(arr[0])):
			arr[i][j] = 0

	# Iterate over each sample square
	for i in range(w):
		for j in range(h):

			# Calculate average pixel intensity
			averagesample = 0
			for k in range(samplewidth):
				for l in range(samplewidth):
					x = k + i * margin
					y = l + j * margin
					averagesample += dotmatrix[x][y]
			
			# Normalize the average
			averagesample /= (samplewidth**2)

			# Contrast modifier
			averagesample -= .5
			averagesample *= contrast
			averagesample += .5
			averagesample = max(min(averagesample, 1.0), 0.0)

			# Iterate over sample square again, additively populating arr w/ values based on distance from center of sample square and average pixel intensity
			for k in range(samplewidth):
				for l in range(samplewidth):
					x = k + i * margin
					y = l + j * margin
					arr[x][y] += validpoint([k, l], samplewidth, averagesample, r, threshold)
					arr[x][y] = min(1.0, arr[x][y])
	return arr

def processimage(filepath, output, samplewidth, overlap, threshold, contrast):

	# Read in image
	original = io.imread(sourcefolder + "/" + filepath)

	# Make it grayscale
	grayscale = rgb2gray(original)

	# Apply dot matrix filter
	dotmatrixfilter(grayscale, samplewidth, overlap, threshold, contrast)

	# Apply gaussian blur
	grayscale = gaussian(grayscale)

	# Save image
	io.imsave('./' + processedfolder + '/' + output, grayscale)

processimage(filename, outputname, sw, ol, th, contrastmod)

#fig, ax = plt.subplots()
#ax.imshow(grayscale, cmap=plt.cm.gray)
#ax.set_title("Pointilist")
#fig.tight_layout()
#plt.show()
