import sys
from datetime import datetime, time, timedelta
import pandas as pd


# C950 - Webinar-1 - Let’s Go Hashing
# W-1_ChainingHashTable_zyBooks_Key-Value.py
# Ref: zyBooks: Figure 7.8.2: Hash table using chaining.
# Modified for Key:Value

# HashTable class using chaining.
# Space Complexity O(N) N is the initial_capacity
class ChainingHashTable:
    # Constructor with optional initial capacity parameter.
    # Assigns all buckets with an empty list.
    # Time complexity: O(1)
    def __init__(self, initial_capacity=10):
        # initialize the hash table with empty bucket list entries.
        self.table = []
        for i in range(initial_capacity):
            self.table.append([])

    # Inserts a new item into the hash table.
    # Time complexity: O(N) Uses chaining
    def insert(self, key, item):  # does both insert and update
        # get the bucket list where this item will go.
        bucket = hash(key) % len(self.table)
        bucket_list = self.table[bucket]

        # update key if it is already in the bucket
        for kv in bucket_list:
            # print (key_value)
            if kv[0] == key:
                kv[1] = item
                return True

        # if not, insert the item to the end of the bucket list.
        key_value = [key, item]
        bucket_list.append(key_value)
        return True

    # Searches for an item with matching key in the hash table.
    # Returns the item if found, or None if not found.
    # Time complexity: O(N) Uses chaining
    def search(self, key):
        # get the bucket list where this key would be.
        bucket = hash(key) % len(self.table)
        bucket_list = self.table[bucket]
        # print(bucket_list)

        # search for the key in the bucket list
        for kv in bucket_list:
            # print (key_value)
            if kv[0] == key:
                return kv[1]  # value
        return None

    # Removes an item with matching key from the hash table.
    # Time complexity: O(N) Uses chaining
    def remove(self, key):
        # get the bucket list where this item will be removed from.
        bucket = hash(key) % len(self.table)
        bucket_list = self.table[bucket]

        # remove the item from the bucket list if it is present.
        for kv in bucket_list:
            # print (key_value)
            if kv[0] == key:
                bucket_list.remove([kv[0], kv[1]])

# Space complexity: O(1)
class Truck:
    # Time complexity: O(1)
    def __init__(self, packages, address, speed, capacity, departureTime):
        self.packages = packages
        self.address = address
        self.speed = speed
        self.departureTime = departureTime
        self.capacity = capacity

# Space complexity: O(1)
class Vertex:
    # Constructor for a new Vertex object. All vertex objects
    # start with a distance of positive infinity.
    # Time complexity: O(1)
    def __init__(self, label):
        self.label = label
        self.distance = float('inf')
        self.pred_vertex = None

# Space complexity: O(N * m) N = number of elements in adjacencyList m = number of elements in adjacency_list[element]
class Graph:
    # Time complexity: O(1)
    def __init__(self):
        self.adjacency_list = {}  # vertex dictionary {key:value}
        self.edge_weights = {}  # edge dictionary {key:value}

    # Time complexity: O(1)
    def add_vertex(self, new_vertex):
        self.adjacency_list[new_vertex] = []  # {vertex_1: [], vertex_2: [], ...}

    # Time complexity: O(1)
    def add_directed_edge(self, from_vertex, to_vertex, weight=1.0):
        self.edge_weights[(from_vertex, to_vertex)] = weight
        # {(vertex_1,vertex_2): 484, (vertex_1,vertex_3): 626, (vertex_2,vertex_6): 1306, ...}
        self.adjacency_list[from_vertex].append(to_vertex)
        # {vertex_1: [vertex_2, vertex_3], vertex_2: [vertex_6], ...}

    # Time complexity: O(1)
    def add_undirected_edge(self, vertex_a, vertex_b, weight=1.0):
        self.add_directed_edge(vertex_a, vertex_b, weight)
        self.add_directed_edge(vertex_b, vertex_a, weight)

# Space complexity: O(1)
class Package:
    # Time complexity: O(1)
    def __init__(self, packageId, address, city, state, zipCode, deliveryDeadline, mass, specialNotes,
                 status='Not delivered', deliveryTime='nan'):
        self.packageId = packageId
        self.address = address
        self.city = city
        self.state = state
        self.zipCode = zipCode
        self.deliveryDeadline = deliveryDeadline
        self.mass = mass
        self.specialNotes = specialNotes
        self.status = status
        self.deliveryTime = deliveryTime

    # Time complexity: O(1)
    def setPackageStatus(self, status):
        self.status = status

    # Time complexity: O(1)
    def setDeliveryTime(self, deliveryTime):
        self.deliveryTime = deliveryTime

    # Time complexity: O(1)
    def __str__(self):
        return '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.packageId, self.address, self.city, self.state,
                                                               self.zipCode, self.deliveryDeadline, self.mass,
                                                               self.specialNotes, self.status, self.deliveryTime)

# Time complexity: O(N) One loop through size of df
# Space complexity: O(N) N = size of df
def loadPackageData(filename):
    df = pd.read_excel(filename)
    df = df.values[7:]

    hashTable = ChainingHashTable(41)
    start = Package(0, '4001 South 700 East', 'Salt Lake City', 'UT', 84107, 'nan', 0, 'nan', 'nan')
    hashTable.insert(0, start)

    for data in df:
        packageId = data[0]
        address = data[1]
        city = data[2]
        state = data[3]
        zipCode = data[4]
        deadline = data[5]
        mass = data[6]
        specialNotes = data[7]

        package = Package(packageId, address, city, state, zipCode, deadline, mass, specialNotes)

        hashTable.insert(packageId, package)

    hashTable.remove(9)
    package = Package(9, '410 S State St', 'Salt Lake City', 'UT', '84111', 'EOD', '2', 'Wrong address listed',
                      'Not delivered')
    hashTable.insert(9, package)
    return hashTable

# Time complexity: O(N^2) first loop through df, second loop through df[data] list
# Space complexity: O(N^2) size of df * size of df[data]
def loadDistanceData(filename):
    df = pd.read_excel(filename)
    df = df.values[7:]
    distanceData = [[] for x in range(len(df))]

    for data in range(len(df)):
        for value in range(2, len(df[data])):
            distanceData[data].append(df[data][value])

    for data in range(len(distanceData)):
        for value in range(len(distanceData[data])):
            distanceData[data][value] = distanceData[value][data]

    return distanceData

# Time complexity: O(N^3) first loop through df, second loop .strip() on data[1], third loop .split()
# Space complexity: O(N * m) N = size of data[1] m = number of \n
def loadAddressData(filename):
    df = pd.read_excel(filename)
    df = df.values[7:]
    addressData = []

    for data in df:
        addressData.append(data[1].strip().split('\n')[0])
    addressData[0] = '4001 South 700 East'
    addressData[24] = '5383 South 900 East #104'

    return addressData

# Time complexity: O(N^2) Nested loops through unvisited_queue and unvisited_queue - 1
# Space complexity: O(N) N = length of adjacency_list
def dijkstra_shortest_path(g, start_vertex):
    # Put all vertices in an unvisited queue.
    unvisited_queue = []

    for current_vertex in g.adjacency_list:
        unvisited_queue.append(current_vertex)
        # unvisited_queue = [vertex_1, vertex_2, ...]

    # Start_vertex has a distance of 0 from itself
    start_vertex.distance = 0

    # One vertex is removed with each iteration; repeat until the list is
    # empty.
    while len(unvisited_queue) > 0:

        # Visit vertex with minimum distance from start_vertex
        smallest_index = 0

        for i in range(1, len(unvisited_queue)):
            if unvisited_queue[i].distance < unvisited_queue[smallest_index].distance:
                smallest_index = i
        current_vertex = unvisited_queue.pop(smallest_index)

        # Check potential path lengths from the current vertex to all neighbors.
        for adj_vertex in g.adjacency_list[current_vertex]:  # values from  dictionary
            # if current_vertex = vertex_1 => adj_vertex in [vertex_2, vertex_3], if vertex_2 => adj_vertex in [vertex_6], ...
            edge_weight = g.edge_weights[(current_vertex, adj_vertex)]  # values from dictionary
            # edge_weight = 484 then 626 then 1306, ...}
            alternative_path_distance = current_vertex.distance + edge_weight

            # If shorter path from start_vertex to adj_vertex is found, update adj_vertex's distance and predecessor
            if alternative_path_distance < adj_vertex.distance:
                adj_vertex.distance = alternative_path_distance
                adj_vertex.pred_vertex = current_vertex

# Time complexity: O(N) N = length of the path from end_vertex to start_vertex
# Space complexity: O(N) N = size of path
def get_shortest_path(start_vertex, end_vertex):
    # Start from end_vertex and build the path backwards.
    path = ""
    current_vertex = end_vertex
    count = 0
    pathList = []
    while current_vertex is not start_vertex:
        path = " -> " + str(current_vertex.label) + path
        current_vertex = current_vertex.pred_vertex
        count += 1
        pathList.append(current_vertex)
    path = start_vertex.label + path

    return path, count, pathList


if __name__ == '__main__':
    # Time complexity: O(N)
    # Space complexity: O(N)
    hashTable = loadPackageData('WGUPS Package File.xlsx')
    distanceData = loadDistanceData('WGUPS Distance Table.xlsx')
    addressData = loadAddressData('WGUPS Distance Table.xlsx')

    truck1Time = timedelta(hours=10, minutes=20, seconds=0)
    truck2Time = timedelta(hours=8, minutes=0, seconds=0)
    truck3Time = timedelta(hours=10, minutes=16, seconds=52)

    truck1 = Truck([0, 1, 4, 5, 7, 8, 9, 10, 11, 12, 17, 21, 22, 23, 24, 26, 27], '4001 South 700 East', 18.0, 16,
                   truck1Time)
    truck2 = Truck([0, 3, 13, 14, 15, 16, 19, 18, 20, 29, 30, 31, 36, 37, 38, 40, 2], '4001 South 700 East', 18.0, 16,
                   truck2Time)
    truck3 = Truck([0, 6, 25, 28, 32, 33, 34, 35, 39], '4001 South 700 East', 18.0, 16, truck3Time)

    # Time complexity: O(NLogN) .sort() = O(NLogN)
    # Space complexity: O(N)
    g = Graph()
    verticesList = []
    verticesList.sort()

    # Time complexity: O(N) length of packagesList
    # Space complexity: O(N)
    def addVertices(packagesList):
        for x in packagesList:
            vertex = Vertex(str(x))
            g.add_vertex(vertex)
            verticesList.append(vertex)

    # Time complexity: O(N^2) first element in adjacency_list * all elements in adjacency_list
    # Space complexity: O(N^2)
    def addEdges():
        for x in g.adjacency_list:
            xIndex = addressData.index(hashTable.search(int(x.label)).address)
            for y in g.adjacency_list:
                yIndex = addressData.index(hashTable.search(int(y.label)).address)
                g.add_undirected_edge(x, y, distanceData[xIndex][yIndex])


    # Time complexity: O(N^2) dijkstra_shortest_path(g, startVertex) = O(N^2), get_shortest_path(startVertex, v) * length of adjacency_list = O(N^2)
    # Space complexity: O(N) N = size of pathList from get_shortest_path(startVertex, v)[2]
    def getShortestPath(startVertex, totalDistance, deliveryTime, packageId=None):
        dijkstra_shortest_path(g, startVertex)
        distance = 100
        shortestEnd = 0
        pathList = []
        pathDistance = 0

        for v in g.adjacency_list:
            if v is not startVertex and v.distance < distance:
                pathList.clear()
                pathList.append(get_shortest_path(startVertex, v)[2])
                distance = v.distance
                shortestEnd = v
                pathDistance = v.distance

        for x in pathList[0]:
            hashTable.search(int(x.label)).setPackageStatus('Delivered')
            hashTable.search(int(x.label)).setDeliveryTime(deliveryTime)
            g.adjacency_list.pop(x)

        timeMinutes = int((pathDistance / 18) * 60)
        timeSeconds = int(((pathDistance / 18) * 60 % 1) * 60)
        timeToDeliver = timedelta(hours=0, minutes=timeMinutes, seconds=timeSeconds)

        deliveryTime += timeToDeliver
        totalDistance += pathDistance

        if int(shortestEnd.label) == packageId:
            packageId = deliveryTime

        if len(g.adjacency_list) == 1:
            hashTable.search(int(shortestEnd.label)).setPackageStatus('Delivered')
            hashTable.search(int(shortestEnd.label)).setDeliveryTime(deliveryTime)
            g.adjacency_list.clear()

            return totalDistance, deliveryTime, packageId

        for y in g.adjacency_list:
            y.pred_vertex = None
            y.distance = float('inf')
        return getShortestPath(shortestEnd, totalDistance, deliveryTime, packageId)


    # Time complexity: O(N^2) addEdges() = O(N^2)
    # Space complexity: O(N^2) addEdges() = O(N^2)
    def loadAllPackages():
        verticesList.clear()
        addVertices(truck1.packages)
        addEdges()
        truck1Distance = getShortestPath(verticesList[0], 0, truck1Time)

        verticesList.clear()
        addVertices(truck2.packages)
        addEdges()
        truck2Distance = getShortestPath(verticesList[0], 0, truck2Time)

        verticesList.clear()
        addVertices(truck3.packages)
        addEdges()
        truck3Distance = getShortestPath(verticesList[0], 0, truck3Time)

        return truck1Distance[0], truck2Distance[0], truck3Distance[0]


    # Time complexity: O(N^2)
    # Space complexity: O(1)
    def getPackageStatus():
        for vertex in range(1, len(hashTable.table)):
            for data in hashTable.table[vertex]:
                print("Package ID: " + str(data[1].packageId) + ', Address: ' + data[1].address + ', Package Status: '
                      + data[1].status + ', Time Delivered: ' + str(data[1].deliveryTime))

    timeHash = ChainingHashTable(41)


    # Time complexity: O(N^2) N = length of truck.packages * hashTable.table[truck.packages[vertex]]
    # Space complexity: O(N) length of truck.packages
    def getAllPackagesByTime(truck, timeToCheck, packageId=None):
        for vertex in range(1, len(truck.packages)):
            for data in hashTable.table[truck.packages[vertex]]:
                if timeToCheck < data[1].deliveryTime and timeToCheck >= truck.departureTime:
                    timeHash.insert(data[1].packageId, "Package ID: " + str(
                        data[1].packageId) + ', Address: ' + data[1].address + ', Package Status: En Route')
                elif timeToCheck <= truck.departureTime:
                    timeHash.insert(data[1].packageId, "Package ID: " + str(
                        data[1].packageId) + ', Address: ' + data[1].address + ', Package Status: At Hub')
                elif timeToCheck >= data[1].deliveryTime:
                    timeHash.insert(data[1].packageId,
                                    "Package ID: " + str(data[1].packageId) + ', Address: ' + data[1].address +
                                    ', Package Status: Delivered' + ', Delivery Time ' + str(data[1].deliveryTime))
        return timeHash.search(packageId)

    print(chr(164) + ' ' + '- ' * 26 + chr(164))
    print(chr(166) + "  Menu:" + ' ' * 46 + chr(166))
    print(chr(166) + "\t 1) Print All Package Status and Total Mileage" + ' ' * 4 + chr(166))
    print(chr(166) + "\t 2) Get a Single Package Status with a Time" + ' ' * 7 + chr(166))
    print(chr(166) + "\t 3) Get All Package Status with a Time" + ' ' * 12 + chr(166))
    print(chr(166) + "\t 4) Exit the Program" + ' ' * 30 + chr(166))
    print(chr(164) + ' ' + '- ' * 26 + chr(164))

    selectedOption = int(input("\nPlease select a menu option: "))

    # Time complexity: O(N^2) getPackageStatus() = O(N^2)
    # Space complexity: O(1)
    if selectedOption == 1:
        distance1, distance2, distance3 = loadAllPackages()
        print()
        getPackageStatus()
        print("\nTotal Distance Traveled (Miles): " + str(round(distance1 + distance2 + distance3, 2)))

    # Time complexity: O(N^2)
    # Space complexity: O(N)
    elif selectedOption == 2:
        packageId = int(input("\nPlease enter the Package ID: "))
        time = input("Please enter a time in HH:MM:SS: ")
        timeToCheck = timedelta(hours=int(time[0:2]), minutes=int(time[3:5]), seconds=int(time[6:8]))
        loadAllPackages()
        if packageId in truck1.packages:
            p = getAllPackagesByTime(truck1, timeToCheck, packageId)
        elif packageId in truck2.packages:
            p = getAllPackagesByTime(truck2, timeToCheck, packageId)
        elif packageId in truck3.packages:
            p = getAllPackagesByTime(truck3, timeToCheck, packageId)
        print()
        print(p)

    # Time complexity: O(N^2) loadAllPackages() = O(N^2) and getAllPackagesByTime() = O(N^2)
    # Space complexity: O(N^2) loadAllPackages() = O(N^2)
    elif selectedOption == 3:
        time = input("\nPlease enter a time in HH:MM:SS: ")
        timeToCheck = timedelta(hours=int(time[0:2]), minutes=int(time[3:5]), seconds=int(time[6:8]))
        loadAllPackages()
        getAllPackagesByTime(truck1, timeToCheck)
        getAllPackagesByTime(truck2, timeToCheck)
        getAllPackagesByTime(truck3, timeToCheck)

        for x in range(len(timeHash.table)):
            for y in timeHash.table[x]:
                print(timeHash.search(x))
    # Time complexity: O(1)
    # Space complexity: O(1)
    elif selectedOption == 4:
        print("\nClosing the program ...")
        exit(1)
    else:
        print("\nNot a valid option")
        exit(1)
