/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cpcs324groupproject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import javafx.util.Pair;

/**
 *
 * @author Admin
 */
public class Cpcs324GroupProject {

    public static void main(String[] args) throws FileNotFoundException {
        Scanner on = new Scanner(System.in);
        System.out.printf("please enter :\n1.Comparison between Horspool and Brute force algorithms "
                + "\n2. Finding minimum spanning tree using Prim’s algorithm "
                + "\n3. Finding minimum spanning tree using Kruskal’s algorithm"
                + "\n4. Finding shortest path using Dijkstra’s algorithm "
                + "\n5. Quit\n ");
        String a = on.next();

        switch (a) {

            case "1":
                Scanner in = new Scanner(System.in);
                System.out.println("How many lines you want to read from the text file?");
                int numOfLines = in.nextInt();
                System.out.println("How many patterns to be generated? ");
                int numOfPatterns = in.nextInt();
                System.out.println("What is the length of each pattern? ");
                int lengthOfPattern = in.nextInt();

                File f = new File("input.txt");
                Scanner read = new Scanner(f);

                String text = "";
                for (int i = 0; i < numOfLines; i++) {
                    text += read.nextLine().toLowerCase(Locale.ROOT) + "";
                }

                PrintWriter output = new PrintWriter("patterns.txt");
                String patterns[] = new String[numOfPatterns];
                int index;

                for (int i = 0; i < numOfPatterns; i++) {
                    index = (int) (Math.random() * (text.length() - lengthOfPattern));
                    String p = "";
                    for (int j = 0; j < lengthOfPattern; j++) {
                        p += text.charAt(index + j) + "";

                    }

                    patterns[i] = p;
                    output.println(p);

                }
                output.close();

                //////////////////////////////////Brute force algorithm ////////////////////////////////////////////////////////
                double startTime = System.nanoTime();

                for (int i = 0; i < numOfPatterns; i++) {

                    BruteForceStringMatch(text, patterns[i], lengthOfPattern);

                }
                double ftime = System.nanoTime();

                double avgTimeForBruteForce = (ftime - startTime) / 1000000;

                //////////////////////////////////Horspool algorithm////////////////////////////////////////////////////////////
                startTime = System.nanoTime();

                for (int i = 0; i < numOfPatterns; i++) {

                    HorspoolMatching(text, patterns[i], lengthOfPattern);

                }
                ftime = System.nanoTime();

                double avgTimeForHorspool = (ftime - startTime) / 1000000;

                /////////////////////////////////////0utput/////////////////////////////////////////////////////////////////////
                System.out.printf("%d patterns, each of length %d have been generated in a file Patterns.txt\n\n", numOfPatterns, lengthOfPattern);
                System.out.printf("Avarage time of search in Brute force Approach: %f \n\n", avgTimeForBruteForce);
                System.out.printf("Avarage time of search in Horspool Approach: %f \n\n", avgTimeForHorspool);

                if (avgTimeForHorspool <= avgTimeForBruteForce) {
                    System.out.println("For this instance Horspool approach is better than Brute Force Approach ");
                } else {
                    System.out.println("For this instance Brute Force approach is better than Horspool Approach ");
                }

                break;

            case "2":

                File g = new File("input1.txt");
                read = new Scanner(g);

                break;
            case "3":
                break;
            case "4":

                File in2 = new File("input2.txt");
                Scanner read2 = new Scanner(in2);

                int V = read2.nextInt();
                Graph graph = new Graph(V);
                Graph1 graph1 = new Graph1(V);
//        System.out.println(V);
                int loob = read2.nextInt();
                int array[][] = new int[V][V];

                int a1,
                 b1,
                 c1;

                int Vinland = V;

                // adjacency list representation of graph
                List<List<Node>> adj_list = new ArrayList<List<Node>>();
                // Initialize adjacency list for every node in the graph 
                for (int i = 0; i < Vinland; i++) {
                    List<Node> item = new ArrayList<Node>();
                    adj_list.add(item);
                }
                for (int i = 0; i < loob; i++) {

                    a1 = read2.nextInt();
                    b1 = read2.nextInt();
                    c1 = read2.nextInt();
                    array[a1][b1] = c1;
                    graph.addEdge(a1, b1, c1);
                    graph1.addEdge(a1, b1, c1);
                    adj_list.get(a1).add(new Node(b1, c1));
                }
                System.out.println("");
                System.out.println("Weight Matrix:\n");
                System.out.print("  ");

                for (int i = 0; i < V; i++) {
                    System.out.print(i + " ");
                }
                System.out.println("");
                for (int i = 0; i < V; i++) {
                    System.out.print(i + " ");
                    for (int j = 0; j < V; j++) {

                        if (array[i][j] != 0) {
                            System.out.print(array[i][j] + " ");

                        } else {
                            System.out.print(0 + " ");
                        }
                    }
                    System.out.println("");
                }

                System.out.println("\n" + "# of vertices is: " + V + " # of edges is: " + loob);
                for (int i = 0; i < V; i++) {
                    System.out.print(i + " :");
                    for (int j = 0; j < V; j++) {

                        if (array[i][j] != 0) {
                            System.out.print(i + "-" + j + " " + array[i][j] + " ");

                        }
                    }
                    System.out.println("");

                }

                System.out.print("Enter Source vertex:");
                Scanner qw = new Scanner(System.in);
                int qwq = qw.nextInt();
                int source = qwq;

//            graph = new Graph(vertices);
//            graph1 = new Graph1(vertices);
                double startTime1 = System.nanoTime();

                graph1.dijkstra_GetMinDistances(qwq);

                double ftime1 = System.nanoTime();

                double avgTimeMH = (ftime1 - startTime1) / 1000000;

                // Input graph edges 
//        adj_list.get(0).add(new Node(3, 7)); 
//        adj_list.get(1).add(new Node(0, 3)); 
//        adj_list.get(1).add(new Node(2, 4));
//        adj_list.get(2).add(new Node(4, 6)); 
//        adj_list.get(3).add(new Node(1, 2));
//        adj_list.get(3).add(new Node(2, 5));
//        adj_list.get(4).add(new Node(3, 4));  
                // call Dijkstra's algo method  
                double startTime2 = System.nanoTime();
                Graph_pq dpq = new Graph_pq(Vinland);
                dpq.algo_dijkstra(adj_list, source);
                double ftime2 = System.nanoTime();

                double avgTimePQ = (ftime2 - startTime2) / 1000000;

                // Print the shortest path from source node to all the nodes 
                System.out.println("Dijkstra using min heap: ");
//        System.out.println("A path from\t\t" + "to #\t\t" + "Distance");
                for (int i = 0; i < dpq.dist.length; i++) {
                    System.out.println("A path from " + source + " To " + i + " (Length:" + dpq.dist[i] + ")");
                }

                System.out.println("Comparison Of the running time : \n Running time of Dijkstra using priority queue is: " + avgTimePQ + " nano seconds");
                System.out.println("  \n Running time of Dijkstra using min Heap is: " + avgTimeMH + " nano seconds");
                if (avgTimeMH > avgTimePQ) {
                    System.out.println("Running time of Dijkstra using priority queue is better");
                } else {
                    System.out.println("Running time of Dijkstra using min Heap is better");
                }

                break;

            default:

        }

    }

    public static int HorspoolMatching(String text, String pattern, int m) {

        int[] shiftTable = ShiftTable(pattern);
        int i = m - 1;     // m is te length of the pattern

        while (i <= text.length() - 10) {
            int k = 0;
            while (k <= m - 1 && pattern.charAt(m - 1 - k) == text.charAt(i - k)) {
                k++;
            }
            if (k == m) {
                return i - m + 1;
            } else {

                i += shiftTable[(text.charAt(i) - 97)];
            }
        }

        return -1;
    }

    public static int[] ShiftTable(String pattern) {
        int table[] = new int[26];
        for (int i = 0; i < table.length; i++) {
            table[i] = pattern.length();
        }
        for (int i = 0; i < pattern.length() - 1; i++) {

            table[(pattern.charAt(i)) - 97] = pattern.length() - i - 1;
        }

        return table;
    }

    public static double BruteForceStringMatch(String text, String pattern, int m) {

        for (int i = 0; i < text.length() - m; i++) {     // m is te length of the pattern
            int j = 0;

            while (j < m && pattern.charAt(j) == text.charAt(i + j)) {
                j++;
            }
            if (j == m) {
                return i;
            }
        }
        return -1;
    }

    static class Edge {

        int source;
        int destination;
        int weight;

        public Edge(int source, int destination, int weight) {
            this.source = source;
            this.destination = destination;
            this.weight = weight;
        }
    }

    static class Graph {

        int vertices;
        LinkedList<Edge>[] adjacencylist;

        Graph(int vertices) {
            this.vertices = vertices;
            adjacencylist = new LinkedList[vertices];
            //initialize adjacency lists for all the vertices
            for (int i = 0; i < vertices; i++) {
                adjacencylist[i] = new LinkedList<>();
            }
        }

        public void addEdge(int source, int destination, int weight) {
            Edge edge = new Edge(source, destination, weight);
            adjacencylist[source].addFirst(edge);

            edge = new Edge(destination, source, weight);
            adjacencylist[destination].addFirst(edge); //for undirected graph
        }

        public void printDijkstra1(int[] distance, int sourceVertex) {
            System.out.println("Dijkstra using priority queue:");
            for (int i = 0; i < vertices; i++) {
                System.out.println("A path from " + sourceVertex + " to  " + +i
                        + " (length: " + distance[i] + ")");
            }
        }
    }

    static class HeapNode {

        int vertex;
        int distance;
    }

    static class Graph1 {

        int vertices;
        LinkedList<Edge>[] adjacencylist;

        Graph1(int vertices) {
            this.vertices = vertices;
            adjacencylist = new LinkedList[vertices];
            //initialize adjacency lists for all the vertices
            for (int i = 0; i < vertices; i++) {
                adjacencylist[i] = new LinkedList<>();
            }
        }

        public void addEdge(int source, int destination, int weight) {
            Edge edge = new Edge(source, destination, weight);
            adjacencylist[source].addFirst(edge);

            edge = new Edge(destination, source, weight);
            adjacencylist[destination].addFirst(edge); //for undirected graph
        }

        public void dijkstra_GetMinDistances(int sourceVertex) {
            int INFINITY = Integer.MAX_VALUE;
            boolean[] SPT = new boolean[vertices];

//          //create heapNode for all the vertices
            HeapNode[] heapNodes = new HeapNode[vertices];
            for (int i = 0; i < vertices; i++) {
                heapNodes[i] = new HeapNode();
                heapNodes[i].vertex = i;
                heapNodes[i].distance = INFINITY;
            }

            //decrease the distance for the first index
            heapNodes[sourceVertex].distance = 0;

            //add all the vertices to the MinHeap
            MinHeap minHeap = new MinHeap(vertices);
            for (int i = 0; i < vertices; i++) {
                minHeap.insert(heapNodes[i]);
            }
            //while minHeap is not empty
            while (!minHeap.isEmpty()) {
                //extract the min
                HeapNode extractedNode = minHeap.extractMin();

                //extracted vertex
                int extractedVertex = extractedNode.vertex;
                SPT[extractedVertex] = true;

                //iterate through all the adjacent vertices
                LinkedList<Edge> list = adjacencylist[extractedVertex];
                for (int i = 0; i < list.size(); i++) {
                    Edge edge = list.get(i);
                    int destination = edge.destination;
                    //only if  destination vertex is not present in SPT
                    if (SPT[destination] == false) {
                        ///check if distance needs an update or not
                        //means check total weight from source to vertex_V is less than
                        //the current distance value, if yes then update the distance
                        int newKey = heapNodes[extractedVertex].distance + edge.weight;
                        int currentKey = heapNodes[destination].distance;
                        if (currentKey > newKey) {
                            decreaseKey(minHeap, newKey, destination);
                            heapNodes[destination].distance = newKey;
                        }
                    }
                }
            }
            //print SPT
            printDijkstra(heapNodes, sourceVertex);
        }

        public void decreaseKey(MinHeap minHeap, int newKey, int vertex) {

            //get the index which distance's needs a decrease;
            int index = minHeap.indexes[vertex];

            //get the node and update its value
            HeapNode node = minHeap.mH[index];
            node.distance = newKey;
            minHeap.bubbleUp(index);
        }

        public void printDijkstra(HeapNode[] resultSet, int sourceVertex) {
            System.out.println("Dijkstra using min heap:");
            for (int i = 0; i < vertices; i++) {
                System.out.println("A path from :" + sourceVertex + " to  " + +i
                        + " (Length: " + resultSet[i].distance + ")");
            }
        }
    }

    static class MinHeap {

        int capacity;
        int currentSize;
        HeapNode[] mH;
        int[] indexes; //will be used to decrease the distance

        public MinHeap(int capacity) {
            this.capacity = capacity;
            mH = new HeapNode[capacity + 1];
            indexes = new int[capacity];
            mH[0] = new HeapNode();
            mH[0].distance = Integer.MIN_VALUE;
            mH[0].vertex = -1;
            currentSize = 0;
        }

        public void display() {
            for (int i = 0; i <= currentSize; i++) {
                System.out.println(" " + mH[i].vertex + "   distance  " + mH[i].distance);
            }
            System.out.println("________________________");
        }

        public void insert(HeapNode x) {
            currentSize++;
            int idx = currentSize;
            mH[idx] = x;
            indexes[x.vertex] = idx;
            bubbleUp(idx);
        }

        public void bubbleUp(int pos) {
            int parentIdx = pos / 2;
            int currentIdx = pos;
            while (currentIdx > 0 && mH[parentIdx].distance > mH[currentIdx].distance) {
                HeapNode currentNode = mH[currentIdx];
                HeapNode parentNode = mH[parentIdx];

                //swap the positions
                indexes[currentNode.vertex] = parentIdx;
                indexes[parentNode.vertex] = currentIdx;
                swap(currentIdx, parentIdx);
                currentIdx = parentIdx;
                parentIdx = parentIdx / 2;
            }
        }

        public HeapNode extractMin() {
            HeapNode min = mH[1];
            HeapNode lastNode = mH[currentSize];
//            update the indexes[] and move the last node to the top
            indexes[lastNode.vertex] = 1;
            mH[1] = lastNode;
            mH[currentSize] = null;
            sinkDown(1);
            currentSize--;
            return min;
        }

        public void sinkDown(int k) {
            int smallest = k;
            int leftChildIdx = 2 * k;
            int rightChildIdx = 2 * k + 1;
            if (leftChildIdx < heapSize() && mH[smallest].distance > mH[leftChildIdx].distance) {
                smallest = leftChildIdx;
            }
            if (rightChildIdx < heapSize() && mH[smallest].distance > mH[rightChildIdx].distance) {
                smallest = rightChildIdx;
            }
            if (smallest != k) {

                HeapNode smallestNode = mH[smallest];
                HeapNode kNode = mH[k];

                //swap the positions 
                indexes[smallestNode.vertex] = k;
                indexes[kNode.vertex] = smallest;
                swap(k, smallest);
                sinkDown(smallest);
            }
        }

        public void swap(int a, int b) {
            HeapNode temp = mH[a];
            mH[a] = mH[b];
            mH[b] = temp;
        }

        public boolean isEmpty() {
            return currentSize == 0;
        }

        public int heapSize() {
            return currentSize;
        }
    }
}

class Graph_pq {

    int dist[];
    Set<Integer> visited;
    PriorityQueue<Node> pqueue;
    int V; // Number of vertices 
    List<List<Node>> adj_list;

    //class constructor
    public Graph_pq(int V) {
        this.V = V;
        dist = new int[V];
        visited = new HashSet<Integer>();
        pqueue = new PriorityQueue<Node>(V, new Node());
    }

    // Dijkstra's Algorithm implementation 
    public void algo_dijkstra(List<List<Node>> adj_list, int src_vertex) {
        this.adj_list = adj_list;

        for (int i = 0; i < V; i++) {
            dist[i] = Integer.MAX_VALUE;
        }

        // first add source vertex to PriorityQueue 
        pqueue.add(new Node(src_vertex, 0));

        // Distance to the source from itself is 0 
        dist[src_vertex] = 0;
        while (visited.size() != V) {

            // u is removed from PriorityQueue and has min distance  
            int u = pqueue.remove().node;

            // add node to finalized list (visited)
            visited.add(u);
            graph_adjacentNodes(u);
        }
    }
    // this methods processes all neighbours of the just visited node 

    private void graph_adjacentNodes(int u) {
        int edgeDistance = -1;
        int newDistance = -1;

        // process all neighbouring nodes of u 
        for (int i = 0; i < adj_list.get(u).size(); i++) {
            Node v = adj_list.get(u).get(i);

            //  proceed only if current node is not in 'visited'
            if (!visited.contains(v.node)) {
                edgeDistance = v.cost;
                newDistance = dist[u] + edgeDistance;

                // compare distances 
                if (newDistance < dist[v.node]) {
                    dist[v.node] = newDistance;
                }

                // Add the current vertex to the PriorityQueue 
                pqueue.add(new Node(v.node, dist[v.node]));
            }
        }
    }
}

class Node implements Comparator<Node> {

    public int node;
    public int cost;

    public Node() {
    } //empty constructor 

    public Node(int node, int cost) {
        this.node = node;
        this.cost = cost;
    }

    @Override
    public int compare(Node node1, Node node2) {
        if (node1.cost < node2.cost) {
            return -1;
        }
        if (node1.cost > node2.cost) {
            return 1;
        }
        return 0;
    }
}
