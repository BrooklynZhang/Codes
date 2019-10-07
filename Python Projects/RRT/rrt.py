#coding:utf-8
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
'''
Set up matplotlib to create a plot with an empty square
'''
def setupPlot():
    fig = plt.figure(num=None, figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
    plt.autoscale(False)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.set_axis_off()
    ax.add_patch(patches.Rectangle(
        (0,0),   # (x,y)
        1,          # width
        1,          # height
        fill=False
        ))
    return fig, ax

'''
Make a patch for a single pology 
'''
def createPolygonPatch(polygon, color):
    verts = []
    codes= []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0]/10., xy[1]/10.))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=1)

    return patch
    

'''
Render the problem  
'''
def drawProblem(robotStart, robotGoal, polygons):
    fig, ax = setupPlot()
    patch = createPolygonPatch(robotStart, 'green')
    ax.add_patch(patch)    
    patch = createPolygonPatch(robotGoal, 'red')
    ax.add_patch(patch)    
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p], 'gray')
        ax.add_patch(patch)    
    plt.show()
                                                                                                                        
                                                                                                            
'''
Grow a simple RRT 
'''
def growSimpleRRT(points):
    
    newPoints = dict()
    adjListMap = dict()
    
    # Your code goes here
    num_points = len(points)
    tmp = []
    
    for i in range (1, len(points)+1): #遍历所有points的点

        distance_1 = 0
        distance_2 = 0
        max_point = 1
        
        newPoints.update({i: points[i]})  #把点放到newpoints中
        #print "i = ", i
        #print "newPoints =", newPoints
        #print "adjListMap = ", adjListMap
        #print "len(adjListMap) = ", len(adjListMap)
        k = 0
        if (i == 1):         #第一第二个点就没必要对比了
            adjListMap.update({1: [2]})
        elif(i == 2):
            adjListMap.update({2: [1]})
        else:
            
            distance_1 = ((points[i][0] - newPoints[1][0])**2 + (points[i][1] - points[1][1])**2 )**(0.5)
            for k in range(0,len(newPoints)-2):  #把所有的points里的点和现有所有的点的距离对比 照出最短的
                j = newPoints.keys()[k]
            
                #比较随机点和两个点的距离，短的距离的点放入max_point
                #print "i = ", i
                #print "k = ", k
                #print "j = ", j
                #print "newPoints.keys()[k] = ", j
                #distance_1 = ((points[i][0] - newPoints[j][0])**2 + (points[i][1] - points[j][1])**2 )**(0.5)
                if(i == j):
                    continue
                #print "i = ", i
                #print "j = ", j
                #print "points[i][0] = ", points[i][0]
                #print "newPoints[newPoints.keys()[k+1]][0] = ", newPoints[newPoints.keys()[k+1]][0]
                #print "points[i][1] = ", points[i][1]
                #print "points[newPoints.keys()[k+1]][1] = ", points[newPoints.keys()[k+1]][1]
                distance_2 = ((points[i][0] - newPoints[newPoints.keys()[k]][0])**2 + (points[i][1] - points[newPoints.keys()[k]][1])**2 )**(0.5)
                #print "distance_1 = ", distance_1
                #print "distance_2 = ", distance_2

                if(distance_1 > distance_2):
                    distance_1 = distance_2
                    #distance_1存着两点间距离的最小值
                    max_point = j
            #print "两点连线的最小值为 = ", distance_1
            #print "最短距离的点为", max_point
        
            #比较点线距离
            distance_3 = 999
            distance_4 = 0
            max_point_2_1 = 0
            max_point_2_2 = 0

            for b in range (0, len(adjListMap)-1):
                #在现存的adjListMap中寻找
                x = adjListMap.keys()[b]
                #print "x = ", x
                #print "adjListMap[x][0] = ", adjListMap[x][0]
                #print "points[adjListMap[x][0]] = ", points[adjListMap[x][0]]
                #print "points[i] = ", points[i]
                #print "points[x] = ", points[x]

                if(len(adjListMap[x]) == 1):
                    #判定在不在线上
                    x_foot, y_foot = foot_point(points[x], points[adjListMap[x][0]], points[i])
                    point_tmp = (x_foot, y_foot)
                    if(is_online(points[x], points[adjListMap[x][0]], point_tmp) == True):
                        distance_4 = point_to_straight(points[x], points[adjListMap[x][0]], points[i])
                        if(distance_3 > distance_4):
                            distance_3 = distance_4
                            max_point_2_1 = x
                            max_point_2_2 = adjListMap[x][0]
                    else:
                        continue
                else:
                    for z in range (0, len(adjListMap[x])):
                        y = newPoints.keys()[z]
                        #distance_3 = point_to_straight(points[x], points[adjListMap[x][y]], points[i])
                        #print "points[x]", points[x]
                        #print "adjListMap[x][y]", adjListMap[x][z]
                        #print "points[adjListMap[x][y]]", points[adjListMap[x][z]]
                        #print "points[i]", points[i]
                        distance_4 = point_to_straight(points[x], points[adjListMap[x][z]], points[i])
                        #print "distance, point_1, point_2", distance_4, x, adjListMap[x][z]
                        #print "distance_3", distance_3
                        #print "distance_4", distance_4
                        if(distance_3 > distance_4):
                            #判定在不在线上
                            x_foot_1, y_foot_1 = foot_point(points[x], points[adjListMap[x][z]], points[i])
                            point_tmp = (x_foot_1, y_foot_1)
                            if(is_online(points[x], points[adjListMap[x][z]], point_tmp) == True):
                                #print "yes"
                                distance_3 = distance_4
                                #distance_3存着点线的最小值
                                max_point_2_1 = x
                                #print "x = ", x
                                max_point_2_2 = adjListMap[x][z]
                                #print "max_point_2_1 = ", max_point_2_1
                                #print "max_point_2_2 = ", max_point_2_2
    
            #print "垂线的最小值为 = ", distance_3
        
            #垂线更短
            if(distance_1 > distance_3):
                #print "max_point_2_1 = ", max_point_2_1
                #print "max_point_2_2 = ", max_point_2_2

                #max_ponit应该换成新的点
                #获得垂足的横纵坐标
                x_foot, y_foot = foot_point(points[max_point_2_1], points[max_point_2_2], points[i])

                #print "points[max_point_2_1][0] = ", points[max_point_2_1][0]
                #print "points[max_point_2_1][1] = ", points[max_point_2_1][1]
                #print "points[max_point_2_2][0] = ", points[max_point_2_2][0]
                #print "points[max_point_2_2][1] = ", points[max_point_2_2][1]
            
                #print "On line"
                
                newPoints.update({num_points + 1: (x_foot, y_foot)})
                points.update({num_points + 1: (x_foot, y_foot)})

                #三个点都加上邻居垂足
                adjListMap.update({i: [num_points + 1]})
                adjListMap[max_point_2_1].append(num_points + 1)
                adjListMap[max_point_2_2].append(num_points + 1)
                #print "adjListMap_1 = ", adjListMap

                #垂足加入adjListMap,并添加三个点作为邻居
                adjListMap.update({num_points + 1: [i]})
                adjListMap[num_points + 1].append(max_point_2_1)
                adjListMap[num_points + 1].append(max_point_2_2)
                
                #print "最近距离的垂足的点为", num_points + 1
                #print "坐标为 ", [x_foot, y_foot]
                #print "adjListMap_online = ", adjListMap
                
                #删除max两点的互相的
                adjListMap[max_point_2_1].remove(max_point_2_2)
                adjListMap[max_point_2_2].remove(max_point_2_1)
                
                num_points = num_points + 1
                            
            #两点直线最短
            else:
                if (i == 1):         #第一第二个点就没必要对比了
                    adjListMap.update({i: []})
                elif (i == 2):
                    adjListMap.update({i: []})
                    adjListMap[j].append(i)
                elif (i == max_point):
                    continue
                else:
                    adjListMap.update({i: [max_point]})
                    adjListMap[max_point].append(i)
                    #print "最近距离的点为 = ", max_point
                    #print "adjListMap_point = ", adjListMap
        #print "\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        # print "newPoints = ", newPoints

    #print "***************************************\nadjListMap: "
    #print adjListMap
    #print "***************************************"
    
    
    return newPoints, adjListMap

def is_online(point_1, point_2, point):
    if(((min(point_1[0], point_2[0]) < point[0]) and (point[0] < max(point_1[0], point_2[0])))
       and ((min(point_1[1], point_2[1]) < point[1]) and (point[1] < max(point_1[1], point_2[1])))):
        return True
    return False


def point_to_straight(point_1, point_2, point):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    if (x2 - x1 == 0):
        distance = np.abs(y2 - y1)
    else:
        A = (y2 - y1) / (x2 - x1)
        B = -1
        C = y1 - A * x1
        distance = np.abs(A*point[0] + B*point[1] + C) / np.sqrt(A**2 + B**2)
    return distance

def foot_point(point_1, point_2, point):
    x0 = point[0]
    y0 = point[1]
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    if (x2 != x1):
        A = (y2 - y1) / (x2 - x1)
        B = -1
        C = y1 - A * x1
        x_new = (B*B*x0 - A*B*y0 - A*C)/(A*A + B*B)
        y_new = (-A*B*x0 + A*A*y0 - B*C)/(A*A + B*B);
    else:
        x_new = x1
        y_new = y2
    return x_new, y_new

def get_index(a, key):
    for i in range (0,len(a)):
        #print "i = ", i
        if(a[i] == key):
            return i
    return -1

        

'''
Perform basic search 
'''
def basicSearch(tree, start, goal):
    path_1 = []
    path_1.append(start)    
    
    

    def bfs(graph_to_search, start, end):
        queue = [[start]]
        visited = set()
        
        while queue:
            path = queue.pop(0)
            vertex = path[-1]
            if vertex == end:
                return path
            elif vertex not in visited:
                for current_neighbour in graph_to_search.get(vertex, []):
                    new_path = list(path)
                    new_path.append(current_neighbour)
                    queue.append(new_path)
                visited.add(vertex)
        return [-1]
    
    path = bfs(tree, start, goal)
    #print path
    return path   
    # Your code goes here. As the result, the function should
    # return a list of vertex labels, e.g.
    #
    # path = [23, 15, 9, ..., 37]
    #
    # in which 23 would be the label for the start and 37 the
    # label for the goal.

'''
Display the RRT and Path
'''


def displayRRTandPath(points, tree, path, robotStart = None, robotGoal = None, polygons = None):
    
    # Your code goes here
    # You could start by copying code from the function
    # drawProblem and modify it to do what you need.
    # You should draw the problem when applicable.
    #print "0000000000"
    #print "points = ", points
    
    
    #display polygons
    #print "00000000000000000000"
    #print "points[1] = ", points[1]
    #print "polygons is ", polygons
    
    
    #dispaly 障碍物
    if(polygons != None):
        for i in range (0, len(polygons)):
            graphx_p = []
            graphy_p = []
            for j in range (0, len(polygons[i])):
                graphx_p.append(polygons[i][j][0])
                graphy_p.append(polygons[i][j][1])
            graphx_p.append(polygons[i][0][0])
            graphy_p.append(polygons[i][0][1])
            #print "graphx_p is ", graphx_p
            #print "graphy_p is ", graphy_p

            plt.plot(graphx_p, graphy_p, 'b')


    #display tree
    #print "tree is ", tree
    #print "vvvvvvvvvvvvvvv"
    for b in range(0, len(tree)):
        i = tree.keys()[b]
    #print "i = ", i
        graphx_p = []
        graphy_p = []
        for j in range(0,len(tree[i])):
            graphx_p.append(points[i][0])
            graphy_p.append(points[i][1])
            graphx_p.append(points[tree[i][j]][0])
            graphy_p.append(points[tree[i][j]][1])
            #print "points is", points
            #print "graphx_p is ", graphx_p
            #print "graphy_p is ", graphy_p
            plt.plot(graphx_p, graphy_p, 'black')
            graphx_p = []
            graphy_p = []

    #display path
    graphx = []
    graphy = []
    #print "path is", path
    #graphx.append(points[path[0]][0])
    #graphy.append(points[path[0]][1])
    for i in path:
        # 在 i index 上实际的 key
        #j = path.keys()[i]
        #print "i = ", i
        #print "point = ", i
        #print "坐标是 ", points[i]
        #print "邻居们是", tree[i]
        #for j in range(0, len(tree[i])):
            #print "邻居为", tree[i][j]
            #print "邻居坐标为", points[tree[i][j]]
            #print "///////"
        graphx.append(points[i][0])
        graphy.append(points[i][1])
    #graphx.append(points[path[len(path)-1]][0])
    #graphy.append(points[path[len(path)-1]][1])
    plt.plot(graphx, graphy, 'orange')

    plt.show()
    return

'''
Collision checking
'''

def isCollisionFree(robot, point, obstacles):

    def inside_polygon(x, y, points):
        x = round(x, 2)
        y = round(y, 2)
        n = len(points)
        inside = False
        p1x, p1y = points[0]
        for i in range(1, n + 1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


    def mult(a, b, c):
        return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
    
    def intersect(a, b, c, d):
        c = [c[0], c[1]]
        d = [d[0], d[1]]
        if(((a[1] - b[1]) != 0) and ((c[1] - d[1]) != 0)):
            slope1 = ((a[0] - b[0])/(a[1] - b[1])) ** 2
            slope2 = ((c[0] - d[0])/(c[1] - d[1])) ** 2

            if (round(slope1,2) == round(slope2,2)):

                return False

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

    robot_curr = []

    for i in range(0, len(robot)):

        robot_curr.append([robot[i][0] + point[0], robot[i][1] + point[1]])

    # Check if there is intersection bw robot and obstacles
    #print "00000000"
    #print "robot_curr = ", robot_curr
    for i in range(0, len(robot_curr)-1):
        # Traverse obstacles
        for j in range(0, len(obstacles)):
            # Check each edge of each obstacles
            for k in range(0, len(obstacles[j])-1):

                if intersect(robot_curr[i], robot_curr[i+1], obstacles[j][k], obstacles[j][k+1]):

                    return False


    # Check if there is a point of robot inside obstacles               
    for i in range(0, len(robot_curr)):

        for j in range(0, len(obstacles)):

            if inside_polygon(robot_curr[i][0], robot_curr[i][1], obstacles[j]):
                return False

    return True

'''
The full RRT algorithm
'''

def closest(point, points):
    dis = distance(point,points[1])
    pt = points[1]
    for i in range(1,len(points)):
        temp = distance(point, points[i])
        if (temp < dis):
            dis = temp
            pt = points[i]
    return pt

def edgeobs(point, points, obstacles):
    check = False
    edgepoint1 = point
    edgepoint2 = closest(point,points)
    for k in range(0, len(obstacles)):
        for l in range(0, len(obstacles[k])-1):
            check =  intersect(edgepoint1, edgepoint2, obstacles[k][l], obstacles[k][l+1])
            if(check == True):
                return check
    return check

def distance(point1,point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**(0.5)

def is_collision(point1,point2,obstacles):
    for i in range(0, len(obstacles)):
        for j in range(0, len(obstacles[i])-1):
            check = intersect(point1, point2, obstacles[i][j], obstacles[i][j+1])
            if (check == True):
                return check
    return False

def RRT(robot, obstacles, startPoint, goalPoint):
    
    def init(robot, obstacles, startPoint, goalPoint):
        points = dict()
        points.update({1:startPoint})
        i = 2
        
        while (i < 400):
            x = round(random.uniform(0,10),1)
            y = round(random.uniform(0,10),1)
            point = (x, y)
            Flag = isCollisionFree(robot, point, obstacles)
            if (Flag == True):
                Flag2 = edgeobs(point, points, obstacles)
                if (Flag2 == False ):
                    points.update({i:point})
                    i = i + 1;
            else:
                continue
        
        points.update({400:goalPoint})
        return points
    
    def removeedge(tree,obstacles):
        tmp = []
        for b in range(0, len(tree)):
            i = tree.keys()[b]
            for j in range(0, len(tree[i])):
                if (is_collision(points[i], points[tree[i][j]], obstacles) == True):
                    tmp.append([i, tree[i][j]])


        for z in range(0, len(tmp)):
            a = tmp[z][0]
            b = tmp[z][1]
            tree[a].remove(b)
        return tree
    
    points = dict()
    tree = dict()
    path = []
    points = init(robot, obstacles, startPoint, goalPoint)
    newpoint, tree = growSimpleRRT(points)
    tree = removeedge(tree,obstacles)
    path = basicSearch(tree, 1 , 400)
    #print "00000000000\npath = ", path
    while (path[0] == -1):
        #print "try again"
        points = dict()
        tree = dict()
        path = []
        points = init(robot, obstacles, startPoint, goalPoint)
        newpoint, tree = growSimpleRRT(points)
        tree = removeedge(tree,obstacles)
        path = basicSearch(tree, 1 , 400)
    
    #print points
    #print ' '
    #print tree
    displayRRTandPath(newpoint,tree, path, robotStart, robotGoal, obstacles)
    return points, tree, path

def is_intersect_line(adjListMap, point_1, point_2, newpoint):
    #判断是否与之前的相交

    for a in range (0, len(adjListMap)):
        x = adjListMap.keys()[a]
        for b in range(0, len(adjListMap[x])):
            #print "x = ", x
            #print "b = ", b
            #print "point_1 = ", x
            #print "zb = ", newpoint[x]
            #print "point_2 = ", tree[x][b]
            #print "zb = ", newpoint[tree[x][b]]
            if (intersect(point_1, point_2, newpoint[x], newpoint[adjListMap[x][b]]) == True):
                return True
    return False


def mult(a, b, c):
    return (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])

def intersect(a, b, c, d):
    
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
    robot = []
    obstacles = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            xy = xys[p].split(',')
            polygon.append((float(xy[0]), float(xy[1])))
        if line == 0 :
            robot = polygon
        else:
            obstacles.append(polygon)

    # Print out the data
    print "Robot:"
    print str(robot)
    print "Pologonal obstacles:"
    for p in range(0, len(obstacles)):
        print str(obstacles[p])
    print ""

    # Visualize
    robotStart = []
    robotGoal = []

    def start((x,y)):
        return (x+x1, y+y1)
    def goal((x,y)):
        return (x+x2, y+y2)
    robotStart = map(start, robot)
    robotGoal = map(goal, robot)
    #drawProblem(robotStart, robotGoal, obstacles)

    # Example points for calling growSimpleRRT
    # You should expect many mroe points, e.g., 200-500
    points = dict()
    points[1] = (5, 5)
    points[2] = (7, 8.2)
    points[3] = (6.5, 5.2)
    points[4] = (0.3, 4)
    points[5] = (6, 3.7)
    points[6] = (9.7, 6.4)
    points[7] = (4.4, 2.8)
    points[8] = (9.1, 3.1)
    points[9] = (8.1, 6.5)
    points[10] = (0.7, 5.4)
    points[11] = (5.1, 3.9)
    points[12] = (2, 6)
    points[13] = (0.5, 6.7)
    points[14] = (8.3, 2.1)
    points[15] = (7.7, 6.3)
    points[16] = (7.9, 5)
    points[17] = (4.8, 6.1)
    points[18] = (3.2, 9.3)
    points[19] = (7.3, 5.8)
    points[20] = (9, 0.6)

    # Printing the points
    print "" 
    print "The input points are:"
    print str(points)
    print ""
    
    points, adjListMap = growSimpleRRT(points)

    # Search for a solution  
    path = basicSearch(adjListMap, 1, 20)

    # Your visualization code 
    displayRRTandPath(points, adjListMap, path)
    print 'next'
    # Solve a real RRT problem
    RRT(robot, obstacles, (x1, y1), (x2, y2))
    
    # Your visualization code
    #displayRRTandPath(points, adjListMap, path, robotStart, robotGoal, obstacles)



