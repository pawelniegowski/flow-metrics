# python -m pip install pillow opencv-python numpy scipy pytesseract tabulate
# For text recognition, install Tesseract and add it to path: https://tesseract-ocr.github.io/tessdoc/Installation.html
# On Linux, install xdg-open to see images
from PIL import Image, ImageDraw, ImageFont, ImageShow
import cv2 as cv
import numpy as np
import numpy.ma as ma
import pytesseract
from scipy import ndimage
from tabulate import tabulate

import glob, os, sys, math, csv, subprocess

PIXELS_PER_UM = float(12)/10  

SHADED_MARGIN = 5
BRIGHTFIELD_BG_AVERAGE = 142
BRIGHTFIELD_NOISE = 5 # deviation in +/- for brightfield
FONT = ImageFont.load_default()
COLUMN_NAMES = [ 'BF1', 'CK', 'SMA', 'SSC', 'Nucleus', 'BF2', 'CD29', 'VIM', 'CD45_CD31' ]
NUCLEUS_COLUMN = COLUMN_NAMES.index('Nucleus')

DEFECT_DEPTH_THRESHOLD = 0
DEFECT_DEPTH_MULTIPLIER = 2
MAX_DEFECT_DIST = 999
DEFECT_MAX_ITERATIONS = 30



interactiveMode = False
interactiveRow = None

if len(sys.argv) < 2:
    print("""Usage: 
python flow-metrics.py <directory with .tifs>
    Measures all TIF images in batch mode, writing to measurements.csv.
python flow-metrics.py <single .tif file> [--interactive] [--irow=n]
    Measures a single file, showing final graphical result and printing metrics.
    --interactive - display step-by-step graphical output
    --irow=3 - display step-by-step graphical output for row number 3
""")

    sys.exit(1)


def isBrightfieldBG(color):
    for i in range(3):
        if color[i] < 120 or color[i] > 160:
            return False
    return True

def findRows(px, height, column):
    bgcolor = px[column,0]

    rowsFound = []
    
    brightfieldColor = None
    
    inRow = False
    rowStart = 0
    
    for i in range(1,height):
        if inRow:
            if px[column,i] == bgcolor:
                inRow = False
                rowsFound.append( (rowStart,i) )
        else:
            if px[column,i] != bgcolor and isBrightfieldBG(px[column,i]):
                inRow = True
                rowStart = i        
    return rowsFound
    
def findColumnWidth(im, width, row):
    for i in range(SHADED_MARGIN,width):
        if not isBrightfieldBG(im[i,row]):
            return i+2 # TODO: better
    return None
    
    
def dbgShow(cvImg):
    if not interactiveMode:
        return
    cv.imshow('img',cvImg)
    if cv.waitKey(0) == 27: # esc
        sys.exit(0)

def findCluster(im, nucIm):
    ocvIm = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    grayscale = cv.cvtColor(ocvIm, cv.COLOR_BGR2GRAY)
    nucleus = cv.cvtColor(np.array(nucIm), cv.COLOR_RGB2GRAY)
    dbgShow(nucleus)
    
    grayscale_processed = grayscale
    _, nucThresh  = cv.threshold(nucleus,50,255, cv.THRESH_BINARY)
    dbgShow(nucThresh)
    
    # remove white aura, make background uniform
    dbgShow(grayscale_processed)
    floodfilled = grayscale_processed.copy()
    for i in range(0,255-BRIGHTFIELD_BG_AVERAGE,2):
        ffmask = np.zeros( (im.height+2, im.width+2), np.uint8)
        cv.floodFill(floodfilled, ffmask, (im.width - SHADED_MARGIN*2,SHADED_MARGIN*2), BRIGHTFIELD_BG_AVERAGE+i, 1, 1)
        
    ffmask = np.zeros( (im.height+2, im.width+2), np.uint8)        
    cv.floodFill(floodfilled, ffmask, (im.width - SHADED_MARGIN*2,SHADED_MARGIN*2), BRIGHTFIELD_BG_AVERAGE, 0, 0)
    grayscale_processed = floodfilled
    
    
    dbgShow(grayscale_processed)
    grayscale_processed = (((grayscale_processed.astype(float)-BRIGHTFIELD_BG_AVERAGE)/BRIGHTFIELD_NOISE)**4).clip(0,255).astype(np.uint8)#
    dbgShow(grayscale_processed)
    
    ret, thresh = cv.threshold(grayscale_processed, 50, 255, cv.THRESH_BINARY)
    dbgShow(thresh)
    
    thresh = cv.addWeighted(thresh,1, nucThresh, 1, 0)
    dbgShow(thresh)
    
    kernel = np.ones((5,5),np.uint8)
    kernel3 = np.ones((3,3),np.uint8)
    
    edt = ndimage.distance_transform_edt(255-thresh)
    edt = 255 - (edt*10).clip(0,255).astype(np.uint8)
    dbgShow(edt)
    edt = cv.blur(edt, (7,7))
    dbgShow(edt)
    
    ret, thresh = cv.threshold(edt, 200,255,cv.THRESH_BINARY)
    dbgShow(thresh)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel3, iterations=4)
    
    dbgShow(thresh)
    
    
    ffmask = np.zeros( (im.height+2, im.width+2), np.uint8)
    cv.floodFill(thresh, ffmask, (0,0), 0, 10, 3)
    
    dbgShow(thresh)
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, 3)
    
    bestContour = None
    bestDistance = 99999
    bestArea = 0
    
    for i in range(len(contours)):
        M = cv.moments(contours[i])
        if M['m00'] == 0: # zero area, ignore
            continue
        
        areaPercent = M['m00'] / (im.width*im.height)
        if areaPercent < 0.01 or areaPercent >= 0.8:
            continue
        
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        dx = im.width/2 - cx
        dy = im.height/2 - cy
        if math.sqrt(dx*dx + dy*dy) < bestDistance:
            bestContour = contours[i]
            bestDistance = math.sqrt(dx*dx + dy*dy)
        cv.drawContours(grayscale,[contours[i]],-1, 255, 2)
        
    if bestContour is not None:
        cv.drawContours(grayscale,[bestContour],-1, 0, 3)        
        dbgShow(grayscale)
    
    return bestContour

def removeDefectsFromContour(contour, defectIds):
    minDist = MAX_DEFECT_DIST**2
    start = 0
    end = 0
    if interactiveMode:
        print("Defects:" + str(defectIds))
    for i in range(0,len(defectIds)):
        for j in range(i+1, len(defectIds)):
            defectI = contour[defectIds[i][0]][0]
            defectJ = contour[defectIds[j][0]][0]
            dx = defectI[0] - defectJ[0]
            dy = defectI[1] - defectJ[1]
            distSq = dx*dx + dy*dy
            
            totalDepth = defectIds[i][1] + defectIds[j][1]
            if distSq > totalDepth*totalDepth * (DEFECT_DEPTH_MULTIPLIER*DEFECT_DEPTH_MULTIPLIER):
                continue

            if minDist > distSq:
                minDist = distSq
                start = defectIds[i][0]
                end = defectIds[j][0]
                
    if start == 0 and end == 0:
        return contour

    if start <= end:
        inside = contour[start:end]
        lenInside = 0
        if inside.size != 0:
            lenInside = cv.arcLength(inside, False)
            
        outside1 = contour[0:start]
        outside2 = contour[end:len(contour)]
        
        lenOutside = 0
        if outside1.size != 0:
            lenOutside += cv.arcLength(outside1, False)
        if outside2.size != 0:
            lenOutside += cv.arcLength(outside2, False)
        if lenOutside < lenInside:
            start,end = end,start     
    else:
        inside = contour[end:start]
        lenInside = 0
        if inside.size != 0:
            lenInside = cv.arcLength(inside, False)
                        
        outside1 = contour[0:end]
        outside2 = contour[start:len(contour)]
        
        lenOutside = 0
        if outside1.size != 0:
            lenOutside += cv.arcLength(outside1, False)
        if outside2.size != 0:
            lenOutside += cv.arcLength(outside2, False)
        if lenInside < lenOutside:
            start,end = end,start

    if start <= end:
        out = np.concatenate((contour[0:start], contour[end:len(contour)]), axis=0)
    else:
        out = contour[end:start]
    return out
    
    
    
def removeDefects(contour):
    defectNum = 9999
    for i in range(DEFECT_MAX_ITERATIONS):
        hull = cv.convexHull(contour, returnPoints=False)
        try:
            defects = cv.convexityDefects(contour, hull)
        except cv.error as e:
            print("Couldn't find convexity defects! Working with unmodified contour...")
            print(e)
            return contour
        defectIds = []
        if defects is None:
            return contour
        for d in range(defects.shape[0]):
            defectStart, defectEnd, furthestPoint, depth = defects[d,0]
            depth = depth/256.0
            if depth > DEFECT_DEPTH_THRESHOLD:
                defectIds.append( (furthestPoint, depth) )
        contour = removeDefectsFromContour(contour, defectIds)
        
        defectNum = len(defectIds)
        if defectNum < 2:
            break
    return contour
        
        

    

def separateCells(im, nucIm, clusterContour):
    ocvIm = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    grayscale = cv.cvtColor(ocvIm, cv.COLOR_BGR2GRAY)
    nucRGB = np.array(nucIm)
    nucleus = nucRGB[:,:,0]
    nucleus = cv.addWeighted(nucRGB[:,:,0], 1, nucRGB[:,:,1], 1, 0.0)
    nucleus = cv.addWeighted(nucleus, 1, nucRGB[:,:,2], 1, 0.0)
    #cv.cvtColor(np.array(nucIm), cv.COLOR_RGB2GRAY)
    #nucleus = cv.cvtColor(np.array(nucIm), cv.COLOR_RGB2GRAY)
    
    nucleus = cv.blur(nucleus,(5,5))

    areaForMax = nucleus[10:(nucIm.height-10),10:(nucIm.width-10)]
    maxPixel = (areaForMax.max() + np.percentile(areaForMax,95)) / 2.0
    
    dbgShow(nucleus)    
    _, thresh  = cv.threshold(nucleus,int(maxPixel * 0.7),255, cv.THRESH_BINARY)
    #_, thresh  = cv.threshold(nucleus,int(maxPixel * 0.6),255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    dbgShow(thresh)
    kernel = np.ones((5,5),np.uint8)

    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel, iterations=2)
    
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    thresh = cv.morphologyEx(thresh, cv.MORPH_ERODE, kernel, iterations=1)    

    dbgShow(thresh)
    nCells, markers = cv.connectedComponents(thresh,_,8,cv.CV_32S)

    mask = np.zeros((im.height, im.width),np.uint8)
    cv.drawContours(mask, [clusterContour], -1, 1, -1)
    markers[mask==0] = 128
   
    edt = ndimage.distance_transform_edt(255-thresh)
    edt = (edt).clip(0,255).astype(np.uint8)
    edt = edt * mask
    dbgShow(edt)

    markers8 = cv.normalize(markers,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
    print("\tFound %d cells" % (nCells-1))
    dbgShow(cv.applyColorMap(markers8,cv.COLORMAP_VIRIDIS))
    
    # watershed is described in detail here:
    # https://people.cmm.minesparis.psl.eu/users/beucher/wtshed.html
    markers = cv.watershed(cv.cvtColor(edt,cv.COLOR_GRAY2BGR), markers)
    markers[markers==-1] = 255
    dbgShow(markers.astype(np.uint8) * 60)
    
    retContours = []
    for i in range(1,nCells):
        marker = (markers == (i)).astype(np.uint8)
        dbgShow(marker.astype(np.uint8) * 60)
        contours, hierarchy = cv.findContours(marker * 255, cv.RETR_TREE, 3)
        if contours:
            retContours.append(contours[0])
        else: 
            print("Warning: contour %d not found" % (i))

        
    
    return retContours

    

    
def findID(im):
    ocvIm = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    grayscale = cv.cvtColor(ocvIm, cv.COLOR_BGR2GRAY)
    grayscaleCropped = grayscale[SHADED_MARGIN:SHADED_MARGIN+40,  SHADED_MARGIN:SHADED_MARGIN+140]
    _, thresh = cv.threshold(grayscaleCropped, 100, 255, cv.THRESH_BINARY_INV)
    thresh = 255 - thresh
    sampleID = pytesseract.image_to_string(thresh, lang='eng', config='--psm 7')
    sampleID = sampleID.replace('T','7') # tesseract sometimes reads 7 as T in this font (?)
    return sampleID.strip()
    
def getIntensityInContour(pilIm, contour):
    mask = np.ones((pilIm.height, pilIm.width),np.uint8)
    cv.drawContours(mask, [contour], -1, 0, -1)
    
    intensityMap = cv.cvtColor(np.array(pilIm), cv.COLOR_RGB2GRAY)
    maskedMap = ma.masked_array(intensityMap, mask=mask)

    mean = maskedMap.mean()
    median = np.ma.median(maskedMap)
    maxv = maskedMap.max()
    return (mean, median, maxv)

def processImage(imgPath):
    global interactiveMode
    origIm = Image.open(imgPath,formats=['tiff'])
    rgbIm = origIm.convert('RGB')
    im = rgbIm.load()

    rows = findRows(im, rgbIm.height, 347) # TODO: combine results from multiple columns, to be sure

    columnWidth = findColumnWidth(im, rgbIm.width,rows[0][0]+SHADED_MARGIN)
    columnNumber = int(rgbIm.width / (columnWidth+5))
    columnWidth = int(rgbIm.width / (columnNumber+1))
   
    outputCells = [ ] 
    
   
    draw = ImageDraw.Draw(rgbIm)
    
    
    for nrow, row in enumerate(rows):
        sampleID = 0
        cvContours = None
        measurements = {}
        if interactiveRow != None:
            interactiveMode = interactiveRow == nrow
        
        for column, i in enumerate(range(0,rgbIm.width, columnWidth)):
            rect = [ (i,row[0]), (i+columnWidth, row[1]) ]
            draw.rectangle( rect, outline="red")
            rectIm = rgbIm.crop(rect[0]+rect[1])
            

                        
            if i==0: # brightfield 1
                sampleID = findID(rectIm)
                print("Starting sample %s..." % sampleID)
                
                nucleusRect = [ (NUCLEUS_COLUMN*columnWidth,row[0]), ((NUCLEUS_COLUMN+1)*columnWidth, row[1]) ]
                nucleusIm = rgbIm.crop(nucleusRect[0]+nucleusRect[1])
                
                clusterContour = findCluster(rectIm, nucleusIm)
                if clusterContour is None:
                    print("\tWarning: cluster contour not found, skipping row")
                    break
                separatedContours = separateCells(rectIm, nucleusIm, clusterContour)
                
                separatedContours = [ removeDefects(x) for x in separatedContours ]
                cellContours = [ clusterContour ] + separatedContours 
                measurements[sampleID] = {}
                for n,cell in enumerate(cellContours):
                    measurements[sampleID][n] = {}
                
            
            if i != 0: # average intensity per contour
                for n,contour in enumerate(cellContours): # TODO : can we get input images in grayscale?
                    if n==0 or column>=len(COLUMN_NAMES): # skip BF1 and extra columns at the end
                        continue
                    intensityMetrics = getIntensityInContour(rectIm, contour)
                    intensityMetrics = [ "%.02f" % (x*100/255) for x in intensityMetrics ] # rescale to 0-100
                    draw.text( (i,row[0]+(n-1)*10), intensityMetrics[0], (255,255,0), font=FONT )
                    measurements[sampleID][n][COLUMN_NAMES[column]+'_mean'] = intensityMetrics[0]
                    measurements[sampleID][n][COLUMN_NAMES[column]+'_median'] = intensityMetrics[1]
                    measurements[sampleID][n][COLUMN_NAMES[column]+'_max'] = intensityMetrics[2]
                
            
            # draw all cells here
            for contourNum, contour in enumerate(cellContours):
                layer = np.zeros((row[1]-row[0], columnWidth,4),np.uint8)
                green = 0
                if contourNum>=1:
                    green = 255
                color = (contourNum*80,green,255,255)
                cv.drawContours(layer,[contour],-1, color, min(contourNum+1,2))
                if contourNum>=1:
                    ellipse = cv.fitEllipse(contour)
                    cv.ellipse(layer,ellipse, (255,255,0,255), 1)
                
                
                pilLayer = Image.fromarray(layer)
                rgbIm.paste(pilLayer, (i,row[0]), mask=pilLayer)

                
            
            
        if not measurements:
            print("...done, but nothing written")
            continue
                
        for n,contour in enumerate(cellContours):
            if n == 0:
                continue
            M = cv.moments(contour)
            area = cv.contourArea(contour)
            # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
            ellipse = cv.fitEllipse(contour)
            
            #maxDiameter = radius*2
            maxDiameter = ellipse[1][0]
            minDiameter = ellipse[1][1]
            if maxDiameter < minDiameter:
                maxDiameter, minDiameter = minDiameter, maxDiameter
            
            measurements[sampleID][n].update({
                'readID': sampleID,
                'cellsInCluster': len(cellContours)-1,
                'indexInCluster': n,
                'area': area / (PIXELS_PER_UM**2),
                'diameterMax': maxDiameter / PIXELS_PER_UM,
                'diameterMin': minDiameter / PIXELS_PER_UM
                #'diameter_min': minDiameter
            })
            outputCells.append(measurements[sampleID][n])
        
        print("...done.")
   
  
    return (outputCells, rgbIm)



target = sys.argv[1]
intensityMetricHeaders = [[ x+'_mean', x+'_median', x+'_max'] for x in COLUMN_NAMES[1:]]
intensityMetricHeaders = [x for sub in intensityMetricHeaders for x in sub] #flatten
headers = ['filename','patientID','readID', 'indexInCluster', 'cellsInCluster', 'area','diameterMin','diameterMax'] + intensityMetricHeaders


if target.endswith(".tif"):
    irow = [x for x in sys.argv if x.startswith("--irow=")]
    if irow != []:
        interactiveRow = int(irow[0][len("--irow="):])-1
    
    if "--interactive" in sys.argv:
        interactiveMode = True
    
    output, im = processImage(target)
    
    print("Printing shortened table (use directory mode to get all metrics in CSV format)...")
    output = [ { k:v for k, v in row.items() if not k.endswith("_mean") and not k.endswith('_median') and not k.endswith('_max')} for row in output ]
    
    print(tabulate(output,headers="keys"))
    if sys.platform=='linux' or sys.platform=='linux2':
        tmpPath ='tmp.png'
        im.save(tmpPath)
        if '--noimage' not in sys.argv:
            subprocess.Popen(['xdg-open',tmpPath])
    else:
        im.show()
    
else:

    with open('measurements.csv','w',newline='') as f:
        wrt = csv.DictWriter(f,fieldnames=headers)
        wrt.writeheader()
        
        searchdir = os.path.join(target,"**")
        print("Searching for TIF files in %s and subdirs..." % searchdir)
        for path in glob.glob(searchdir, recursive=True):
            if not path.endswith(".tif") or os.path.split(path)[1].startswith('.'):
                continue
            print(path)
            output, im = processImage(path)
            for row in output:
                directory, fname = os.path.split(path)
                row['filename'] = fname
                row['patientID'] = fname.split("_")[0]
                wrt.writerow(row)
