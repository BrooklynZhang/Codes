import sys
import numpy as np
import copy
'''
Report reflexive vertices
'''
def findReflexiveVertices(polygons):
    vertices=[]
    # Getting the slope of two points
    def getAngle(pointP, pointX):
        x = pointX[0] - pointP[0]
        y = pointX[1] - pointP[1]
        angle = np.rad2deg(np.arctan2(y, x))
        #print angle
        return angle

    #Sort the vertices for Graham Scan
    def GSsort(polygon):
        temp = polygon[0]
        resultIndex = 0
        result = []
        #list = []

        # Finding the vertex with smallest y
        for index in range(0, len(polygon)):
            # If temp is bigger, update temp
            if temp[1] > polygon[index][1]:
                temp = polygon[index]
                resultIndex = index
            # If equals temp, get the one with smaller x cordinate
            if temp[1] == polygon[index][1]:
                if temp[0] > polygon[index][0]:
                    temp = polygon[index]
                    resultIndex = index

        result.append(polygon[resultIndex])
        polygon.remove(polygon[resultIndex])
        #print result[0]
        #print polygon

        # Sort the rest of the vertices based on their slope with result[0]
        temp = result[0]
        tempIndex = 0
        for i in range(0, len(polygon)):
            angle = 180
            temp = result[0]
            for j in range(0, len(polygon)):
                
                if angle > getAngle(result[0], polygon[j]):
                    angle = getAngle(result[0], polygon[j])
                    temp = polygon[j]
                    #print temp
                    tempIndex = j
            result.append(temp)
            polygon.remove(polygon[tempIndex])
        #print result
        return result 

    def boxPoints(polygon):
        results = []
        results.append(polygon[0])

        #Traverse the polygon clockwise, ignore concave points
        #Go till len(polygon)-1 bc p3 needs to go back to the starting point
        polygon.append(polygon[0])
        for i in range(0, len(polygon)-2):
            p1 = polygon[i]
            p2 = polygon[i + 1]
            p3 = polygon[i + 2]

            z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if z > 0:
                results.append(polygon[i+1])
            #print z
        return results
    wholeThing = []
    for i in range(0, len(polygons)):
        temp = []
        for j in range(0, len(polygons[i])):
            temp.append(polygons[i][j])
        wholeThing.append(temp)

    #print "this should be poly ", wholeThing

    for i in range(0, len(polygons)):
        vertices = vertices + (boxPoints(GSsort(wholeThing[i])))

    #print "-----------------------------------"
    # Your code goes here
    # You should return a list of (x,y) values as lists, i.e.
    # vertices = [[x1,y1],[x2,y2],...]

    return vertices

'''
Compute the roadmap graph
'''
def computeSPRoadmap(polygons, reflexVertices):
    def distance(p1, p2):
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]

        return (x**2 + y**2)**(1.0/2)

    #print polygons
    vertexMap = dict()
    adjacencyListMap = dict()
    
    #remove non-reflexive vertices from polygons
    polyBox = []
    for i in range(0, len(polygons)):
        temp = []
        for j in range(0, len(polygons[i])):
            if polygons[i][j] in reflexVertices:
                temp.append(polygons[i][j])
        polyBox.append(temp)
    #print "polyBox", polyBox

    #build an array consisting all edges of the polybox
    boxEdge = []
    #first add the first vertex again at the end of the list
    for i in range(0, len(polyBox)):
        polyBox[i].append(polyBox[i][0])
        #print polyBox[i]

    #connect adjacent vertices and add edges
    for i in range(0, len(polyBox)):
        temp = []
        for j in range(0, len(polyBox[i])-1):
            p1 = polyBox[i][j]
            p2 = polyBox[i][j+1]
            #print [p1, p2]
            temp.append([p1, p2])
        boxEdge.append(temp)
        #print "--------------------------"

    def mult(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
    def intersect3(a, b, c, d):
        if a == c or a == d:
            return False
        if b == c or b == d:
            return False
        if max(a[0], b[0]) < min(c[0], d[0]):
            return False
        if max(a[1], b[1]) < min(c[1], d[1]):
            return False
        if max(c[0], d[0]) < min(a[0], b[0]):
            return False
        if max(c[1], d[1]) < min(a[1], b[1]):
            return False
        if mult(c, b, a) * mult(b, d, a) < 0:
            return False
        if mult(a, d, c) * mult(d, b, c) < 0:
            return False
        return True
    #intersect([0,0,5,0], [0,1,5,1])
    #check if the line segment of two points intersect any polygon
    def check_intersect(p1, p2, boxEdge):
        #temp1 = [p1[0], p1[1], p2[0], p2[1]]
        for i in range(0, len(boxEdge)):
            for j in range(0, len(boxEdge[i])):
                #temp2 = [boxEdge[i][j][0][0],boxEdge[i][j][0][1], boxEdge[i][j][1][0], boxEdge[i][j][1][1]]
                if intersect3(p1, p2, boxEdge[i][j][0], boxEdge[i][j][1]):
                    #print "result from intersect3 ", intersect3(p1, p2, boxEdge[i][j][0], boxEdge[i][j][1])
                    return True
        return False
    #print check_intersect([1,0], [2,7], boxEdge)
    polyDiagnal = []
    for i in range(0, len(polyBox)):
        temp = []
        for j in range(0, len(polyBox[i])):
            x = polyBox[i][j]
            for k in range(0, len(polyBox[i])):
                y = polyBox[i][k]
                if x != y:
                    temp.append([x, y])
        polyDiagnal.append(temp)
    #print "diagonals are ", polyDiagnal
    start = []
    end = []
    result = []
    for i in range(0, len(polyBox)):
        for j in range(0, len(polyBox[i])):
            start = polyBox[i][j]
            for x in range(0, len(polyBox)):
                for y in range(0, len(polyBox[x])):
                    end = polyBox[x][y]
                    if (not check_intersect(start, end, boxEdge)) and (not check_intersect(start, end, polyDiagnal)):
                        if [start, end] not in result:
                            result.append([start, end])
    temp = []
    for i in range(0, len(result)):
        if ([result[i][1], result[i][0]] not in temp) and (result[i][0] != result[i][1]):
            temp.append(result[i])
            #print result[i]
    result = temp
    for i in range(0, len(reflexVertices)):
        vertexMap.update({i+1: reflexVertices[i]})
    index = 1
    for i in vertexMap:
        temp = []
        for j in vertexMap:
            if ([vertexMap[i], vertexMap[j]] in result) or ([vertexMap[j], vertexMap[i]] in result):
                temp.append([j, distance(vertexMap[i], vertexMap[j])])
        adjacencyListMap.update({index: temp})
        index = index + 1
    #print adjacencyListMap

        #print i, vertexMap[i], adjacencyListMap[i]
    #for i in range(0, len(result)):



    # Your code goes here
    # You should check for each pair of vertices whether the
    # edge between them should belong to the shortest path
    # roadmap. 
    #
    # Your vertexMap should look like
    # {1: [5.2,6.7], 2: [9.2,2.3], ... }
    #
    # and your adjacencyListMap should look like
    # {1: [[2, 5.95], [3, 4.72]], 2: [[1, 5.95], [5,3.52]], ... }
    #
    # The vertex labels used here should start from 1
    
    return vertexMap, adjacencyListMap

'''
Perform uniform cost search 
'''
def uniformCostSearch(adjListMap, start, goal):
    
    path = []
    pathLength = 0
    path.append(start)
    usc = dict()  
    find = -1
    adj = []
    previous = []
    for i in adjListMap:
        usc.update({i : [pathLength,path]})
        previous.append([])
        find = find + 1
        check = False
    for i in adjListMap:        
        pathLength = usc[i][0]
        path = copy.deepcopy(usc[i][1])
        for j in range(0,len(adjListMap[i])):        
            pathLength = usc[i][0] + adjListMap[i][j][1]
            adj = copy.deepcopy(adjListMap[i][j][0])
            if adj not in path:
                path.append(adj)
                find = adjListMap[i][j][0]
                if usc[find][0] == -1:
                    check = True
                    break
                if usc[find][0] == 0:
                    usc.update({adjListMap[i][j][0]:[pathLength,path]})
                else:
                    if usc[find][0] > pathLength:
                        usc.update({adjListMap[i][j][0] :[pathLength,path]})
            if check:
                break
    print usc
    pathLength = usc[goal][0]
    path = usc[goal][1]
    return path, pathLength
    '''
        
    iteration = [0,[start]]
    global iteration_list
    global passed_list
    global previous
    previous = []
    iteration_list = []
    iteration_list.append(iteration)
    print iteration_list
    print 'xxxxxxxxxxxx'
   # print adjListMap[start]
    for x in range(0, len(adjListMap[start])):
        temp = [0,[start]]
        cost = temp[0] + adjListMap[start][x][1]
        temp[0] = cost
        temp[1].append(adjListMap[start][x][0])
        iteration_list.append(temp)
        previous.append(adjListMap[start][x][0])
    print previous
    
    def iterationnext(adjListMap):
        global iteration_list
        global previous
        previous1 = copy.deepcopy(previous)
        print previous1
        iteration_list1 = copy.deepcopy(iteration_list)
        for x in range(0, len(previous1)):
            temp1 = copy.deepcopy(iteration_list1[len(iteration_list1) - len(previous1) + x])
            #print temp1
            for y in range(0, len(adjListMap[previous1[x]])):
                temp2 =copy.deepcopy(temp1)
                if (adjListMap[previous1[x]][y][0] == -1):
                    break
                if (adjListMap[previous1[x]][y][0] not in temp2[1] and adjListMap[previous1[x]][y][0] != -1):
                    cost = temp2[0] + adjListMap[previous1[x]][y][1]
                    temp2.pop(0)
                    temp2.insert(0,cost)
                    add = adjListMap[previous1[x]][y][0]
                    temp2[1].append(add)
                    iteration_list.append(temp2)
                    previous.append(temp2[1][len(temp2[1])-1])


            previous.remove(previous1[x])

        return
    
    
    print 'start !!!!!!!!!!'
    
    
    for a in range(0, 8):
        print a
        iterationnext(adjListMap)  
    print iteration_list

    for i in range(0,len(iteration_list)):
        if (iteration_list[i][1][len(iteration_list[i][1])-1] == -1):
            pathLength = iteration_list[i][0]
            break;
            
    for y in range(0,len(iteration_list)):
        if (iteration_list[y][1][len(iteration_list[y][1])-1] == -1):
            if (iteration_list[y][0] < pathLength):
               pathLength = iteration_list[y][0]  
               
    for x in range(0,len(iteration_list)):
        if (pathLength == iteration_list[x][0]):
            path = iteration_list[x][1]
            break;
            '''

'''
Agument roadmap to include start and goal
'''
def updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2):
    updatedALMap = dict()
    startLabel = 0
    goalLabel = -1
    
    # Your code goes here. Note that for convenience, we 
    # let start and goal have vertex labels 0 and -1,
    # respectively. Make sure you use these as your labels
    # for the start and goal vertices in the shortest path
    # roadmap. Note that what you do here is similar to
    # when you construct the roadmap. 
    vertices=[]
    # Getting the slope of two points
    def getAngle(pointP, pointX):
        x = pointX[0] - pointP[0]
        y = pointX[1] - pointP[1]
        angle = np.rad2deg(np.arctan2(y, x))
        #print angle
        return angle

    #Sort the vertices for Graham Scan
    def GSsort(polygon):
        temp = polygon[0]
        resultIndex = 0
        result = []
        #list = []

        # Finding the vertex with smallest y
        for index in range(0, len(polygon)):
            # If temp is bigger, update temp
            if temp[1] > polygon[index][1]:
                temp = polygon[index]
                resultIndex = index
            # If equals temp, get the one with smaller x cordinate
            if temp[1] == polygon[index][1]:
                if temp[0] > polygon[index][0]:
                    temp = polygon[index]
                    resultIndex = index

        result.append(polygon[resultIndex])
        polygon.remove(polygon[resultIndex])
        #print result[0]
        #print polygon

        # Sort the rest of the vertices based on their slope with result[0]
        temp = result[0]
        tempIndex = 0
        for i in range(0, len(polygon)):
            angle = 180
            temp = result[0]
            for j in range(0, len(polygon)):
                
                if angle > getAngle(result[0], polygon[j]):
                    angle = getAngle(result[0], polygon[j])
                    temp = polygon[j]
                    #print temp
                    tempIndex = j
            result.append(temp)
            polygon.remove(polygon[tempIndex])
        #print result
        return result 

    def boxPoints(polygon):
        results = []
        results.append(polygon[0])

        #Traverse the polygon clockwise, ignore concave points
        #Go till len(polygon)-1 bc p3 needs to go back to the starting point
        polygon.append(polygon[0])
        for i in range(0, len(polygon)-2):
            p1 = polygon[i]
            p2 = polygon[i + 1]
            p3 = polygon[i + 2]

            z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            if z > 0:
                results.append(polygon[i+1])
            #print z
        return results
    wholeThing = []
    for i in range(0, len(polygons)):
        temp = []
        for j in range(0, len(polygons[i])):
            temp.append(polygons[i][j])
        wholeThing.append(temp)

    #print "this should be poly ", wholeThing

    for i in range(0, len(polygons)):
        vertices = vertices + (boxPoints(GSsort(wholeThing[i])))
#/////////////////////////////////////////////////////////////////////////////////
    def distance(p1, p2):
        x = p2[0] - p1[0]
        y = p2[1] - p1[1]

        return (x**2 + y**2)**(1.0/2)

    #remove non-reflexive vertices from polygons
    polyBox = []
    for i in range(0, len(polygons)):
        temp = []
        for j in range(0, len(polygons[i])):
            if polygons[i][j] in vertices:
                temp.append(polygons[i][j])
        polyBox.append(temp)
    #print "polyBox", polyBox

    #build an array consisting all edges of the polybox
    boxEdge = []
    #first add the first vertex again at the end of the list
    for i in range(0, len(polyBox)):
        polyBox[i].append(polyBox[i][0])
        #print polyBox[i]

    #connect adjacent vertices and add edges
    for i in range(0, len(polyBox)):
        temp = []
        for j in range(0, len(polyBox[i])-1):
            p1 = polyBox[i][j]
            p2 = polyBox[i][j+1]
            #print [p1, p2]
            temp.append([p1, p2])
        boxEdge.append(temp)
        #print "--------------------------"

    def mult(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
    def intersect3(a, b, c, d):
        if a == c or a == d:
            return False
        if b == c or b == d:
            return False
        if max(a[0], b[0]) < min(c[0], d[0]):
            return False
        if max(a[1], b[1]) < min(c[1], d[1]):
            return False
        if max(c[0], d[0]) < min(a[0], b[0]):
            return False
        if max(c[1], d[1]) < min(a[1], b[1]):
            return False
        if mult(c, b, a) * mult(b, d, a) < 0:
            return False
        if mult(a, d, c) * mult(d, b, c) < 0:
            return False
        return True
    def check_intersect(p1, p2, boxEdge):
        for i in range(0, len(boxEdge)):
            for j in range(0, len(boxEdge[i])): 
                if intersect3(p1, p2, boxEdge[i][j][0], boxEdge[i][j][1]):               
                    return True
        return False
    polyDiagnal = []
    for i in range(0, len(polyBox)):
        temp = []
        for j in range(0, len(polyBox[i])):
            x = polyBox[i][j]
            for k in range(0, len(polyBox[i])):
                y = polyBox[i][k]
                if x != y:
                    temp.append([x, y])
        polyDiagnal.append(temp)
    start = []
    end = []
    result = []
    a = []
    a.append([x1, y1])
    polyBox.append(a)
    b = []
    b.append([x2, y2])
    polyBox.append(b)
    #print "polyBox",polyBox
    for i in range(0, len(polyBox)):
        for j in range(0, len(polyBox[i])):
            start = polyBox[i][j]
            for x in range(0, len(polyBox)):
                for y in range(0, len(polyBox[x])):
                    end = polyBox[x][y]
                    if (not check_intersect(start, end, boxEdge)) and (not check_intersect(start, end, polyDiagnal)):
                        if [start, end] not in result:
                            result.append([start, end])
    temp = []
    for i in range(0, len(result)):
        if ([result[i][1], result[i][0]] not in temp) and (result[i][0] != result[i][1]):
            temp.append(result[i])
            #print result[i]
    result = temp

    UpdatedvertexMap = dict()
    UpdatedvertexMap.update({-1: [x2,y2]})
    UpdatedvertexMap.update({0: [x1,y1]})
    #print "lllllllllllllllllll",UpdatedvertexMap
    for i in range(0, len(reflexVertices)):
        UpdatedvertexMap.update({i+1: reflexVertices[i]})
    #print "UpdatedvertexMap",UpdatedvertexMap

    index = -1
    for i in UpdatedvertexMap:
        temp = []
        for j in UpdatedvertexMap:
            if ([UpdatedvertexMap[i], UpdatedvertexMap[j]] in result) or ([UpdatedvertexMap[j], UpdatedvertexMap[i]] in result):
                #print "in result: ", i, j
                temp.append([j, distance(UpdatedvertexMap[i], UpdatedvertexMap[j])])
                #print temp
        #print "llllllllllllllllllll",vertexMap[i], vertexMap[j]
        updatedALMap.update({i : temp})
        index = index + 1
    '''
    print "------------------------------------" 
    for i in updatedALMap:
        print i, updatedALMap[i]
    print "------------------------------------" 
    '''
    return startLabel, goalLabel, updatedALMap

if __name__ == "__main__":
    
    # Retrive file name for input data
    if(len(sys.argv) < 6):
        print "Five arguments required: python spr.py [env-file] [x1] [y1] [x2] [y2]"
        exit()
    
    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])

    # Read data and parse polygons
    lines = [line.rstrip('\n') for line in open(filename)]
    polygons = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            polygon.append(map(float, xys[p].split(',')))
        polygons.append(polygon)

    # Print out the data
    print "Pologonal obstacles:"
    for p in range(0, len(polygons)):
        print str(polygons[p])
    print ""

    # Compute reflex vertices
    reflexVertices = findReflexiveVertices(polygons)
    print "Reflexive vertices:"
    print str(reflexVertices)
    print ""

    # Compute the roadmap 
    vertexMap, adjListMap = computeSPRoadmap(polygons, reflexVertices)
    print "Vertex map:"
    print str(vertexMap)
    print ""
    print "Base roadmap:"
    print str(adjListMap)
    print ""

    # Update roadmap
    start, goal, updatedALMap = updateRoadmap(polygons, vertexMap, adjListMap, x1, y1, x2, y2)
    print "Updated roadmap:"
    print str(updatedALMap)
    print ""

    # Search for a solution     
    path, length = uniformCostSearch(updatedALMap, start, goal)
    print "Final path:"
    print str(path)
    print "Final path length:" + str(length)
    

    # Extra visualization elements goes here
