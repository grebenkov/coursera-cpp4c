// compile with C++11 features enabled, for gcc use -std=c++11 flag
// Note: outside of the MOOC, I would have split this program into
// severall .cpp/.h files, with declarations of classes in headers
// and implementation in .cpp. I decided not to do it so it would be
// necessary to submit only one file.

// Also, no need to write inline functions if we use Java or C#-style
// class definitions (with all code inside class definition) — 
// ISO C++ says that the inline definition of member function in C++ 
// is the same as declaring it with inline

#include <iostream> // std::cout
#include <utility> // std::pair
#include <list> // std::list
#include <limits> // std::numeric_limits
#include <vector> // std::vector
#include <algorithm> // std::push_heap
#include <random> // std::random_device, std::uniform_real_distribution, 
                  // std::mt19937

// I am not "using namespace std", because I prefer to see clearly where my 
// functions and classes come from. It is not hard to type in std:: prefix,
// but it contributes to the code clarity. It may be not a big issue in a 
// small program, but as code gets larger, it becomes a problem.

class Vertex {  
public:  
  // Let's have fun with some typedefs. Advantage: we show that edge number
  // and edge distance are different things. Disadvantage: sometimes it is
  // useful to know underlying type, and it is not obvious to what some_t
  // maps. Using an IDE solves that problem, and I think using custom type
  // names will make code more readable. 
  typedef unsigned int vertex_number_t;
  typedef double distance_t;
  // I use a key-value pair here, because we need to store edge value
  // (which is distance)
  typedef std::pair<vertex_number_t, distance_t> edge_t;
  
  // Use a list instead of a vector. We do not need random access to 
  // neighbors list, but we want to allow to remove neighbors, and it is
  // more efficient using a list.
  typedef std::list<edge_t> neighbors_list_t;
  
  // default constructor
  Vertex() {
    clear_data();
  }
    
  static const distance_t len_infinity;
  static const vertex_number_t maxvector;

  // I use adjancency list approach. It is less space efficient than
  // adjancency matrix in non-sparse lists (density greater than 1/64)
  // (see https://en.wikipedia.org/wiki/Adjacency_list), but it is more
  // time-efficent when we want to get all vertices adjancent to node,
  // and in Dijkstra algorithm we do just that.  
  
  // In Dijkstra algorithm, we need to know vertex distance from source
  // and previous vector in the shortest path. We also should be able
  // to mark vertices as visited
  distance_t get_distance() const { return distance_from_source; }
  void set_distance(distance_t distance_) { distance_from_source = distance_; }
  
  Vertex* get_previous() const { return previous; }
  void set_previous(Vertex* previous_) { previous = previous_; }
  
  bool get_marked() const {return marked; }
  void mark() { marked = true; }
  void unmark() { marked = false; }
    
  // Clear all additional data — set it to defaults
  void clear_data() {
    distance_from_source = Vertex::len_infinity;
    previous = nullptr;
    marked = false;
  }
  
  // Return a reference to neighbors list. It is const reference, because
  // we do not want external users to change list. Whole function is marked
  // const because it does not change the Vertex object. 
  const neighbors_list_t& get_neighbors() const { return neighbors; }
  
  // Add an edge to node with value distance.
  // Default distance is 0.
  void add_neighbor(vertex_number_t node, distance_t distance = 0.0) {
    neighbors.push_back(std::make_pair(node, distance));
  }
  
  // Get edge value (distance) to the neighbor with a specific number
  // Return infinity if edge does not exist.
  distance_t get_neighbor(vertex_number_t node) {
    // Idiomatic way of iterating over list in C++. Note the auto keyword
    // form C++11 standard: it allows shorter declarations: I would have
    // to write neighbors_list_t::iterator previously.
    for (auto it = neighbors.begin(); it != neighbors.end(); it++) {
      if (std::get<0>(*it) == node) {
	      return std::get<1>(*it);
      }
    }
    // Edge does not exist
    return len_infinity;
  }
  
  // Remove neighbor with a specific number. If no such neighbor, do nothing
  void remove_neighbor(vertex_number_t node) {
    for (auto it = neighbors.begin(); it != neighbors.end(); it++) {
      if (std::get<0>(*it) == node) {
	      neighbors.erase(it);
	      break;
      }
    }
  }
  
private:  
  neighbors_list_t neighbors;  
  distance_t distance_from_source; 
  Vertex* previous; 
  bool marked;
};

// init vertex static consts
const Vertex::distance_t Vertex::len_infinity = std::numeric_limits<double>::infinity();
const Vertex::vertex_number_t Vertex::maxvector = std::numeric_limits<unsigned int>::max();

class Graph {
public:
  // Constructor
  // Parameters:
  // node_number — number of nodes in the graph
  Graph(Vertex::vertex_number_t node_number) {
    // Initialize vertices vector
    vertices.resize(node_number);
  }
  
  // check for existence of edge between nodes a and b, true if exists
  // node indexes are zero-based in Graph!
  bool has_edge(Vertex::vertex_number_t a, Vertex::vertex_number_t b)
  {
    if (vertices[a].get_neighbor(b) != Vertex::len_infinity) {
      return true;
    }
    else {
      return false;
    }
  }
  // create an edge between nodes a and b with specified distance
  void create_edge(Vertex::vertex_number_t a, Vertex::vertex_number_t b, 
		               Vertex::distance_t distance = 0.0) {
    vertices[a].add_neighbor(b, distance);
  }
  
  // clear additional data in vertices (such as distance, etc.)
  void clear_data() {
    // C++11 for_each loop
    for (Vertex& v : vertices) {
      v.clear_data();
    }
  }
  
  // get node bu number
  Vertex& get_node(Vertex::vertex_number_t node_number) {
    return vertices[node_number];
  }
  
private:
  std::vector<Vertex> vertices;
};

// We need this helper class because STL heap functions help retrieve 
// element with _highest_ value (use < operator to compare), and we need 
// reverse of that.
template <class T>
struct GreaterThanComparer {
   bool operator()(const T& s1, const T& s2)
   {
       return s1 > s2;
   }
};

// Priority queue. 
// See http://marknelson.us/1996/01/01/priority-queues/ for further
// reference
template <class T>
class PriorityQueue {
public:
  void push(T elem) {
    heap.push_back(elem);
    std::push_heap(heap.begin(), heap.end(), GreaterThanComparer<T>());
  }
  
  bool empty() const {
    return heap.empty();
  }
  
  T min() const {
    return heap.front();
  }
  
  void pop() {
    std::pop_heap(heap.begin(), heap.end(), GreaterThanComparer<T>());
    heap.pop_back();
  }
private:
  std::vector<T> heap;
};

class ShortestPath {
public:
  // Constructor. Parameters:
  // graph — graph to operate on
  // start_ — starting node
  // end_ — target node  
  //
  // No error checking for now! It is up to user to provide correct input
  // Constructor performs only initialization. Actual pathfinding is done
  // by path_length method.
  ShortestPath(Graph& graph_, Vertex::vertex_number_t start_, 
	             Vertex::vertex_number_t end_) : graph(graph_), 
	             start(start_), end(end_) { 
	  length_in_nodes = 0;
	}
	       
  // Implementation of algorithm from 
  // https://en.wikipedia.org/wiki/Dijkstra's_algorithm	       
  Vertex::distance_t path_length() {
    // Initialize
    graph.clear_data();
    PriorityQueue<pathlen_t> pq;
    
    Vertex& source = graph.get_node(start);
    // Distance from source to source is 0
    source.set_distance(0);
    // Insert source in priority queue
    pq.push(std::make_pair(0, start));
    // Target vector. 
    Vertex* target_node = &(graph.get_node(end));
    
    // The main loop
    while(!pq.empty()) {
      // get vertex in queue with smallest distance 
      pathlen_t closest_vertex_pathlen = pq.min();      
      // remove it from the queue
      pq.pop();
      // get vertex corresponding to it      
      Vertex& vertex = graph.get_node(std::get<1>(closest_vertex_pathlen));
      // mark as visited
      vertex.mark();
      
      // check if we got to target
      if (&vertex == target_node)
      {
	      length_in_nodes = 0;
	V     Vertex* p = &vertex;
	      // descend down the path until we get to nullptr
	      while (p = p->get_previous()) {
	        length_in_nodes++;
	      }	
	      return vertex.get_distance();	
      }
      
      // for each neighbor of vertex
      for (Vertex::edge_t edge : vertex.get_neighbors()) 
      {
 	      Vertex::vertex_number_t neighbor_no = std::get<0>(edge);
	      Vertex& neighbor = graph.get_node(neighbor_no);
	      // accumulate shortest distance from source
	      Vertex::distance_t dist = vertex.get_distance() + std::get<1>(edge);
	      // if found a shorter path (at init — shorter than infinity)
	      if (dist < neighbor.get_distance() && !neighbor.get_marked()) {
	        // update everything and add vector to queue
	        neighbor.set_distance(dist);
	        neighbor.set_previous(&vertex);
	        pq.push(std::make_pair(dist, neighbor_no));
	      }	
      }
    }
    
    // haven't got to target from source
    return Vertex::len_infinity;
  }
  
  Vertex::vertex_number_t get_length_in_nodes() { return length_in_nodes; }
  
private:
  Graph& graph;
  Vertex::vertex_number_t start, end;
  
  Vertex::vertex_number_t length_in_nodes;
  
  // type definition to represent node's distance from source
  // order is reversed because pairs compare lexicografically
  // see http://www.cplusplus.com/reference/utility/pair/operators/
  typedef std::pair<Vertex::distance_t, Vertex::vertex_number_t> pathlen_t;
};

class MonteCarloSimulation {
public:
  // Constructor. Parameters:
  // node_number_ — number of nodes in graph, > 10 recommended
  // density_ — percentage of the edges to be created / 100, from 0.0 to 1.0.
  // min_distance_ — minimal edge value, should be greater than 0.0
  // max_distance_ — minimal edge value, should be greater than min_distance_
  //
  // No error checking for now! It is up to user to provide correct input
  // Constructor performs only initialization.
  MonteCarloSimulation(int node_number_, double density_, 
			    double min_distance_, double max_distance_) : 
			    node_number(node_number_), density(density_),
			    min_distance(min_distance_), 
			    max_distance(max_distance_) { } 
  
  // Actually run the simulation. Can be called multiple times, each times
  // a new graph is created.
  void RunSimulation() {
    Graph g(node_number);    
    
    fill_random_edges(g);
    
    // average path length = total path length / number of paths
    // assuming number of paths = node number - 1, will decrement if no path
    Vertex::distance_t total_path_length = 0.0;
    int number_of_paths = node_number - 1;
    Vertex::vertex_number_t total_length_in_nodes = 0;
    
    for (int i = 1; i < node_number; ++i) {
      ShortestPath path(g, 0, i);
      Vertex::distance_t length = path.path_length();      
      if (length != Vertex::len_infinity) {
	      total_path_length += length;
	      total_length_in_nodes += path.get_length_in_nodes();	
      } else {
	      number_of_paths -= 1;
      }      
    }
    
    // starting vertex is not connected to other vectors
    if (number_of_paths == 0) {      
      shortest_path_in_nodes = 0;
      shortest_path_in_distance = 0;
    }
    else
    { 
      shortest_path_in_distance = total_path_length / number_of_paths;
      shortest_path_in_nodes = total_length_in_nodes / number_of_paths;
    }       
  }
  
  double get_shortest_path_in_distance() {    
    return shortest_path_in_distance;
  }
  
  int get_shortest_path_in_nodes() {    
    return shortest_path_in_nodes;
  }

private:
  int node_number;
  double density;
  double min_distance, max_distance;
  double shortest_path_in_distance;
  int shortest_path_in_nodes;    
  
  // helper function to fill a graph with edges. 
  void fill_random_edges(Graph& g)
  {
    // C++11 way of getting random numbers, see 
    // http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    // std::random_device uses hardware RNG when available. It is not guaranteed
    // to work fast, so we use only one value from it to seed regular PRNG.
    std::random_device randomdevice;
    std::mt19937 generator(randomdevice());
    std::uniform_real_distribution<> distance_distribution(min_distance,
							                                             max_distance);
    // interval of this distribution is closed: [a,b]
    std::uniform_int_distribution<> node_distribution(0, node_number-1);
    
    // number of edges in complete graph = n(n-1)/2, see 
    // https://en.wikipedia.org/wiki/Complete_graph
    int edges_to_create = static_cast<int>(node_number * (node_number - 1) / 2
                                                       * density);    
    while (edges_to_create > 0) {
      // pick two random nodes
      int a = node_distribution(generator);
      int b = node_distribution(generator);
      // check for an edge between a and b, no loops
      if (a == b || g.has_edge(a, b)) {
	      continue; // try another edge
      }      
      // create undirected edge with random distance
      auto distance = distance_distribution(generator);
      g.create_edge(a, b, distance);
      g.create_edge(b, a, distance);            
      // one less to go
      edges_to_create -= 1;
    }
  }
};

void perform_simulation(int node_number, double density, double min_distance, double max_distance,
			int iterations)
{
  MonteCarloSimulation sim(node_number, density, min_distance, max_distance);
  std::cout << "Simulation: " << node_number << " nodes, density " << density 
            << ", distance range " << min_distance << " to " << max_distance 
            << ", " << iterations << " iterations." << std::endl;
  double total_length = 0.0;	   
  int total_length_in_nodes = 0;
  int numtries = iterations;
  
  for (int i=0; i<iterations; ++i) {        
    sim.RunSimulation();
    if (sim.get_shortest_path_in_distance() > 0) {
      total_length += sim.get_shortest_path_in_distance();    
      total_length_in_nodes += sim.get_shortest_path_in_nodes();
    } else {
      numtries--;
    }
  }
	    
  std::cout << "Average path length in distance: " << total_length / numtries << std::endl;
  std::cout << "Average path length in nodes: " 
            << static_cast<double>(total_length_in_nodes) / numtries << std::endl;
}

int main(int argc, char **argv) {  
  perform_simulation(50, 0.2, 1.0, 10.0, 10000);
  perform_simulation(50, 0.4, 1.0, 10.0, 10000);
  return 0;
}
