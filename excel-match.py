# pip install pyexcel pyexcel-ods
import sys, os
import pyexcel



if len(sys.argv) < 5:
    print("""Usage: 
python xlsx-match.py [ .xlsx or .ods file ] [number of columns] [indexLeft] [indexRight]
    Matches first [number of columns] with all the other columns based on matching two indices.
    Indices may have suffixes starting with "_" that will be ignored for matching purposes.
    Output is written to another file (test.xlsx -> test.matched.xlsx).
    Example: 
        python excel-match.py MyTest.ods 4 "readID" "Cell number"
""")
    sys.exit(1)

def processIndex(idx):
    if isinstance(idx,str) and '_' in idx:
        return idx.split('_')[0]
    else:
        return str(idx)


    
filename, nColumns, indexL, indexR = sys.argv[1:]
nColumns = int(nColumns)

records = pyexcel.get_records(file_name=filename)
leftLookup = {}
rightLookup = {}
for row in records:
    lindex = processIndex(row[indexL])
    rindex = processIndex(row[indexR])
    leftLookup[lindex] = True
    rightLookup[rindex] = True


output = []
output.append(list(records[0].keys()))

totalColumns = len(records[0].values())

currentLeft = 0 
currentRight = 0
prevLindex = ''
prevRindex = ''
lindex = 'ERROR'
rindex = 'ERROR'
prevRowL = None
prevRowR = None
rowL = None
rowR = None
while currentLeft < len(records) or currentRight < len(records):
    try:
        rowL = records[currentLeft]
        if currentLeft>0:
            prevLindex = processIndex(records[currentLeft-1][indexL])
        lindex = processIndex(rowL[indexL])
    except IndexError:
        rowL = {x:'' for x in range(totalColumns)}
        lindex = None
        print("left side ended")
    try:
        rowR = records[currentRight]
        if currentRight>0:
            prevRindex = processIndex(records[currentRight-1][indexR])
        rindex = processIndex(rowR[indexR])
    except IndexError:
        rowL = {x:'' for x in range(totalColumns)}
        rindex = None
        print("right side ended")
        
    if not lindex and not rindex: # ???
        break
        
    if lindex == rindex:
        print("Matched: %s" % lindex)
        currentLeft += 1
        currentRight += 1
        output.append( list(rowL.values())[0:nColumns] + list(rowR.values())[nColumns:])
        continue
        
        
    if not lindex: # ran out of records on the left
        output.append( ([''] * nColumns) + list(rowR.values())[nColumns:] )    
        print("lindex is none")
        currentRight+=1
        continue
        
    if not rindex: # ran out of records on the right
        output.append( list(rowL.values())[0:nColumns] + ([''] * (totalColumns-nColumns)) )
        print("rindex is none")
        currentLeft+=1
        continue        
        
    if rindex not in leftLookup: # complete mismatch, insert blank+R
        output.append( ([''] * nColumns) + list(rowR.values())[nColumns:] )    
        print("Mismatch: %s not present on the left" % rindex)
        currentRight += 1
        continue
    if lindex not in rightLookup: # complete mismatch, insert L+blank
        output.append( list(rowL.values())[0:nColumns] + ([''] * (totalColumns-nColumns)) )
        currentLeft += 1
        print("Mismatch: %s not present on the right" % lindex)
        continue
       
    if prevRindex == rindex: # extra element on the left, insert blank+R
        output.append( ([''] * nColumns) + list(rowR.values())[nColumns:] )
        currentRight += 1
        continue

    if prevLindex == lindex: # extra element on the right, insert L+blank
        output.append( list(rowL.values())[0:nColumns] + ([''] * (totalColumns-nColumns)) )
        currentLeft += 1
        continue
        
    
    print(rowL)
    print(rowR)
    print("ERROR: malformed data? (rows above) check sorting in your excel file")
    sys.exit(1)
 
splitPath = os.path.splitext(filename)
destPath = splitPath[0] + '.matched' + splitPath[1]
pyexcel.save_as(array=output, dest_file_name=destPath)
    

