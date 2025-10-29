#include "graph_utils.h"
#include "mvc_env.h"
#include "graph.h"
#include <cassert>
#include <ctime>
#include <random>
#include <algorithm>
#include <set>
#include "stdio.h"
#include <queue>
#include <vector>
#include <stack>



MvcEnv::MvcEnv(double _norm)
{
    norm = _norm;
    graph = nullptr;
    numCoveredEdges = 0;
    CcNum = 1.0;
    state_seq.clear();
    act_seq.clear();
    action_list.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
    // Initialize episode truncation parameters
    trunc_threshold = 0.0;
    rollout_return = 0.0;

}

MvcEnv::~MvcEnv()
{
    norm = 0;
    graph = nullptr;
    numCoveredEdges = 0;
    state_seq.clear();
    act_seq.clear();
    action_list.clear();
    reward_seq.clear();
    sum_rewards.clear();
    covered_set.clear();
    avail_list.clear();
}

void MvcEnv::s0(std::shared_ptr<Graph> _g)
{
    graph = _g;
    covered_set.clear();
    action_list.clear();
    numCoveredEdges = 0;
    CcNum = 1.0;
    state_seq.clear();
    act_seq.clear();
    reward_seq.clear();
    sum_rewards.clear();
    // Initialize episode truncation parameters for new episode
    trunc_threshold = 0.2 * graph->num_nodes;  // N_trunc = 50% of original graph size
    rollout_return = 0.0;
}

double MvcEnv::step(int a)

{
    assert(graph);
    assert(covered_set.count(a) == 0);
    state_seq.push_back(action_list);
    act_seq.push_back(a);
    covered_set.insert(a);
    action_list.push_back(a);

    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;

//    double oldCcNum = CcNum;
//    CcNum = getNumofConnectedComponents();    
//    double r_t = getReward(oldCcNum);
    double r_t = getReward();
    reward_seq.push_back(r_t);
    sum_rewards.push_back(r_t);  // same as reward_seq in this script, re caluculated in nstep_replay_mem.cpp

    return r_t;
}

double MvcEnv::stepRollout(int a)

{
    assert(graph);
    assert(covered_set.count(a) == 0);
    // state_seq.push_back(action_list);
    // act_seq.push_back(a);
    covered_set.insert(a);
    // action_list.push_back(a);

    for (auto neigh : graph->adj_list[a])
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;

//    double oldCcNum = CcNum;
//    CcNum = getNumofConnectedComponents();    
//    double r_t = getReward(oldCcNum);

    double r_t = getReward();
    return r_t;
}

void MvcEnv::stepWithoutReward(int a)

{
    assert(graph);
    assert(covered_set.count(a) == 0);
    covered_set.insert(a);
    action_list.push_back(a);
    for (auto neigh : graph->adj_list[a])
        {
        if (covered_set.count(neigh) == 0)
            numCoveredEdges++;
        }
}

// delete a in g, g will be changed
void MvcEnv::stepDelete(int a)

{
    assert(graph);
    // std::shared_ptr<GraphUtil> graphutil =std::shared_ptr<GraphUtil>(new GraphUtil());
    // graphutil->deleteNode(graph->adj_list,a);
    int node = a;
    // modify adjlist/edgelist
    for (int i=0; i<(int)graph->adj_list[node].size(); ++i)
    {
        int neighbour = graph->adj_list[node][i];
        graph->adj_list[neighbour].erase(remove(graph->adj_list[neighbour].begin(), graph->adj_list[neighbour].end(), node), graph->adj_list[neighbour].end());

        std::vector<std::pair<int,int>>::iterator iter = graph->edge_list.begin();
        while (iter!=graph->edge_list.end())
        {
            if ((*iter).first==neighbour || (*iter).second==neighbour) 
            {
                iter=graph->edge_list.erase(iter);
            }
            else
                iter++;
        }
    }
    // update edgelist: keep 0-num_nodes index
    std::set<int> nodes_set;
    std::vector<int> nodes_vec;
    for (auto iter = graph->edge_list.begin(); iter != graph->edge_list.end(); iter++)
    {
        nodes_set.insert((*iter).first);
        nodes_set.insert((*iter).second);
    }
    nodes_vec.assign(nodes_set.begin(), nodes_set.end());
    for (auto iter = graph->edge_list.begin(); iter != graph->edge_list.end(); iter++)
    {
        int f = (*iter).first;
        int s = (*iter).second;
        auto it_f = std::find(nodes_vec.begin(),nodes_vec.end(),f);
        auto it_s = std::find(nodes_vec.begin(),nodes_vec.end(),s);
        (*iter).first = it_f - nodes_vec.begin();
        (*iter).second = it_s - nodes_vec.begin();
    }
    

    graph->adj_list[node].clear();

    //modify node_num/edge_num
    graph->num_nodes--;
    graph->num_edges = graph->edge_list.size();
}

// random
int MvcEnv::randomAction()
{
    assert(graph);
    avail_list.clear();

    for (int i = 0; i < graph->num_nodes; ++i)
        if (covered_set.count(i) == 0)
        {
            bool useful = false;
            for (auto neigh : graph->adj_list[i])
                if (covered_set.count(neigh) == 0)
                {
                    useful = true;
                    break;
                }
            if (useful)
                avail_list.push_back(i);
        }

    assert(avail_list.size());
    srand(std::time(0));
    int idx = rand() % avail_list.size();
    return avail_list[idx];
}

////degree
int MvcEnv::degreeAction()
{
   assert(graph);
   avail_list.clear();

   int maxID = -1;
   int maxDegree = 0;
   for (int i = 0; i < graph->num_nodes; ++i)
   {
       int degree = 0;
       if (covered_set.count(i) == 0)
       {
           for (auto neigh : graph->adj_list[i])
               if (covered_set.count(neigh) == 0)
               {
                   degree++;
               }
       }
       if(degree>maxDegree){
           maxDegree = degree;
           maxID = i;
       }
   }
   return maxID;
}

// betweenness
int MvcEnv::betweenAction()
{
    assert(graph);

    std::map<int,int> id2node;
    std::map<int,int> node2id;

    std::map <int,std::vector<int>> adj_dic_origin;
    std::vector<std::vector<int>> adj_list_reID;


    for (int i = 0; i < graph->num_nodes; ++i)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                   if(adj_dic_origin.find(i) != adj_dic_origin.end())
                   {
                       adj_dic_origin[i].push_back(neigh);
                   }
                   else{
                       std::vector<int> neigh_list;
                       neigh_list.push_back(neigh);
                       adj_dic_origin.insert(std::make_pair(i,neigh_list));
                   }
                }
            }
        }

    }


     std::map<int, std::vector<int>>::iterator iter;
     iter = adj_dic_origin.begin();

     int numrealnodes = 0;
     while(iter != adj_dic_origin.end())
     {
        id2node[numrealnodes] = iter->first;
        node2id[iter->first] = numrealnodes;
        numrealnodes += 1;
        iter++;
     }

     adj_list_reID.resize(adj_dic_origin.size());

     iter = adj_dic_origin.begin();
     while(iter != adj_dic_origin.end())
     {
        for(int i=0;i<(int)iter->second.size();++i){
            adj_list_reID[node2id[iter->first]].push_back(node2id[iter->second[i]]);
        }
        iter++;
     }


    std::vector<double> BC = Betweenness(adj_list_reID);
    std::vector<double>::iterator biggest_BC = std::max_element(std::begin(BC), std::end(BC));
    int maxID = std::distance(std::begin(BC), biggest_BC);
    int idx = id2node[maxID];
//    printGraph();
//    printf("\n maxBetID:%d, value:%.6f\n",idx,BC[maxID]);
    return idx;
}

// all edges are covered
bool MvcEnv::isTerminal()
{
    assert(graph);
    return graph->num_edges <= numCoveredEdges;
}

// all nodes are covered
bool MvcEnv::isNoNodes()
{
    assert(graph);
    printf("in mvc_env.cpp: %d,%d",graph->num_nodes,covered_set.size());
    return graph->num_nodes <= covered_set.size();
}

bool MvcEnv::isTruncated()
{
    assert(graph);

    // Check if the largest connected component size is below the truncation threshold
    double lcc_size = getMaxConnectedNodesNum();
    if (lcc_size < trunc_threshold) {
        return true;
    }
    
    return false;
}

// estimate remaining discount return with rollout strategy
double MvcEnv::estimateRemainingReturn(int rollout, double gamma)
{   
    // Create a copy of the current state for heuristic rollout
    std::set<int> covered_set_copy = covered_set;
    std::vector<int> action_list_copy = action_list;
    int numCoveredEdges_copy = numCoveredEdges;

    assert(graph);
    
    // Auto-select rollout method based on graph fragmentation
    if (rollout == -1) {
        double lcc_size = getMaxConnectedNodesNum();
        double lccRatio = lcc_size / (double)graph->num_nodes;
        
        // If graph is highly fragmented (LCC < 30% of total nodes), use component-based estimation
        // This is much faster than full heuristic rollout
        if (lccRatio < 0.3 || lcc_size <= 10) {
            rollout_return =  componentRollout(gamma);
        } else {
            rollout_return =  HeuristicRollout(gamma);
        }
    }
    
    // Manual selection
    if (rollout == 0)
        rollout_return = HeuristicRollout(gamma);
    else if (rollout == 1)
        rollout_return = componentRollout(gamma);
    else
        rollout_return = 0.0; // Invalid rollout type

    // Restore original state
    covered_set = covered_set_copy;
    action_list = action_list_copy;
    numCoveredEdges = numCoveredEdges_copy;

    return rollout_return;
}

double MvcEnv::componentRollout(double gamma)
{
    assert(graph);
    
    // Use the ultra-fast size-only version for better performance
    std::vector<int> component_sizes = getCcSizesDescending();
    
    double cumulative_return = 0.0;
    double lcc_size = getMaxConnectedNodesNum();
    int total_nodes = graph->num_nodes;
    
    
    // If graph is highly fragmented (LCC <= 3 and most components are tiny)
    if (lcc_size <= 3.0) {
        // Calculate more accurate expected steps for fragmented graph
        double total_steps = 0.0;
        
        // For each component, estimate steps needed
        for (int comp_size : component_sizes) {
            if (comp_size <= 1) {
                // Isolated nodes need no more steps
                continue;
            } else if (comp_size == 2) {
                // Size-2 components need 1 step to break
                total_steps += 1.0;
            } else if (comp_size == 3) {
                // Size-3 components need 1-2 steps to break
                total_steps += 1.66667;
            } else {
                // Larger components (shouldn't happen when lcc_size <= 3, but safety check)
                total_steps += comp_size * 0.6;
            }
        }
        double final_reward = -(1.0) / (total_nodes * total_nodes); // Target: LCC size = 1
        double current_reward = -lcc_size / (total_nodes * total_nodes);
        double reward_improvement = final_reward - current_reward;
        cumulative_return = reward_improvement * (1.0 - std::pow(gamma, total_steps)) / (1.0 - gamma);

        // More sophisticated approach: model the sequential dismantling process
        // Sort components by priority (larger components first for better reward improvement)
        // Calculate step-by-step reward improvement
        double current_lcc = lcc_size;
        cumulative_return = 0.0;
        total_steps = 0.0;
        
        for (int comp_size : component_sizes) {
            if (comp_size <= 1) {
                // Isolated nodes contribute no future reward
                continue;
            }
            double step = 0.0;
            if (comp_size == 2) {
                step = 1.0;
            } else if (comp_size == 3) {
                step = 1.5;
            } else {
                step = comp_size * 0.6;
            }
            
            // Reward improvement when this component is dismantled
            double reward_before = -current_lcc / (total_nodes * total_nodes);
            double reward_after = -std::max(1.0, current_lcc - comp_size + 1.0) / (total_nodes * total_nodes);
            double reward_improvement = reward_after - reward_before;
            
            // Add discounted reward for this component
            double discount_factor = std::pow(gamma, total_steps);
            cumulative_return += discount_factor * reward_improvement;
            
            // Update state for next iteration
            current_lcc = std::max(1.0, current_lcc - comp_size + 1.0);
            total_steps += step;
        }
        
        return cumulative_return;
    }
    
    // For larger components, use the original detailed estimation
    for (int comp_size : component_sizes) {
        if (comp_size <= 1) {
            // Isolated nodes contribute no future reward
            continue;
        }
        
        // Estimate reward improvement from dismantling this component
        // Current contribution: -comp_size^2 / total_nodes^2 (approximately)
        // Final contribution: -1 / total_nodes^2 (when reduced to isolated nodes)
        double current_comp_reward = -(double)(comp_size * comp_size) / (total_nodes * total_nodes);
        double final_comp_reward = -(double)comp_size / (total_nodes * total_nodes); // comp_size isolated nodes
        double reward_improvement = final_comp_reward - current_comp_reward;
        
        // Estimate steps needed to dismantle this component
        // For a connected component of size n, we typically need to remove about 60-80% of nodes
        // to break it into small fragments. Use a more conservative estimate.
        double dismantling_ratio = 0.7; // Need to remove 70% of nodes
        if (comp_size <= 5) dismantling_ratio = 0.6; // Smaller components easier to break
        if (comp_size <= 3) dismantling_ratio = 0.5; // Very small components
        
        double expected_steps = std::ceil(comp_size * dismantling_ratio);
        
        // Calculate discounted return for this component using geometric series
        // V = r + γr + γ²r + ... + γ^(k-1)r = r * (1 - γ^k) / (1 - γ)
        // But we need to model that reward improves gradually as component breaks down
        
        double step_reward = reward_improvement / expected_steps; // Average reward per step
        double component_return = 0.0;
        
        if (gamma < 1.0) {
            component_return = step_reward * (1.0 - std::pow(gamma, expected_steps)) / (1.0 - gamma);
        } else {
            component_return = step_reward * expected_steps;
        }
        
        cumulative_return += component_return;
    }
    
    return cumulative_return;
}

double MvcEnv::HeuristicRollout(double gamma)
{
    assert(graph);
    
    int numUnCoveredNodes = graph->num_nodes - covered_set.size();
    
    int rollout_steps = 0;
    std::vector<double> rollout_reward_seq;
    
    double cumulative_return = 0.0;
    // Execute Degree Attack heuristic until graph is completely dismantled
    while (true) {
        // Check if graph is completely dismantled (LCC size <= 1)
        if(isTerminal()) break;
        
        // Get the node with highest degree using degreeAction
        int a = degreeAction();
        if (a == -1) {
            // No valid action available, break
            break;
        }
        
        // Execute the action without reward (just update state)
        rollout_reward_seq.push_back(stepRollout(a));
        rollout_steps++;
        
        // Safety check to prevent infinite loops
        if (rollout_steps > numUnCoveredNodes) break;
    }
    
    // Handle edge case where no steps were taken
    if (rollout_steps == 0) {
        cumulative_return = 0.0;
    } else {
        // Calculate discounted return by working backwards
        for(int i = rollout_steps-2; i >= 0; i--){
            rollout_reward_seq[i] = rollout_reward_seq[i] + gamma * rollout_reward_seq[i+1];
        }
        cumulative_return = rollout_reward_seq[0];
    }

    return cumulative_return;
}

double MvcEnv::getReward()
{
    return -(double)getMaxConnectedNodesNum()/(graph->num_nodes*graph->num_nodes);
}


//double MvcEnv::getReward(double oldCcNum)
//{
//    return (CcNum - oldCcNum) / CcNum*graph->num_nodes ;
//
//}

void MvcEnv::printGraph()
{   
    printf("node_num: %d\n",graph->num_nodes);
    printf("edge_num: %d\n",graph->num_edges);
    // printf("edge_list:\n");
    // printf("[");
    // for (int i = 0; i < (int)graph->edge_list.size();i++)
    // {
    // printf("[%d,%d],",graph->edge_list[i].first,graph->edge_list[i].second);
    // }
    // printf("]\n");


    printf("covered_set:\n");

    std::set<int>::iterator it;
    printf("[");
    for (it=covered_set.begin();it!=covered_set.end();it++)
    {
        printf("%d,",*it);
    }
    printf("]\n");

    std::vector<int> ccSizes = getCcSizesDescending();
    printf("connected component sizes descending:\n");
    printf("[");
    for(int i =0;i<(int)ccSizes.size();++i){
        printf("%d,",ccSizes[i]);
    }
    printf("]\n");  

}

std::vector<int> MvcEnv::getCcSizesDescending()
{
    return graph->getCcSizesDescending(covered_set);
}

double MvcEnv::getNumofConnectedComponents()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }
    std::set<int> lccIDs;
    for(int i =0;i< graph->num_nodes; i++){
        lccIDs.insert(disjoint_Set.unionSet[i]);
    }
    return (double)lccIDs.size();
}

double MvcEnv::getMaxConnectedNodesNum()
{
    assert(graph);
    Disjoint_Set disjoint_Set =  Disjoint_Set(graph->num_nodes);

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (covered_set.count(i) == 0)
        {
            for (auto neigh : graph->adj_list[i])
            {
                if (covered_set.count(neigh) == 0)
                {
                    disjoint_Set.merge(i, neigh);
                }
            }
        }
    }
    return (double)disjoint_Set.maxRankCount;
}


std::vector<double> MvcEnv::Betweenness(std::vector< std::vector <int> > adj_list) {

	int i, j, u, v;
	int Long_max = 4294967295;
	int nvertices = adj_list.size();	// The number of vertices in the network
	std::vector<double> CB;
    double norm=(double)(nvertices-1)*(double)(nvertices-2);

	CB.resize(nvertices);

	std::vector<int> d;								// A vector storing shortest distance estimates
	std::vector<int> sigma;							// sigma is the number of shortest paths
	std::vector<double> delta;							// A vector storing dependency of the source vertex on all other vertices
	std::vector< std::vector <int> > PredList;			// A list of predecessors of all vertices

	std::queue <int> Q;								// A priority queue soring vertices
	std::stack <int> S;								// A stack containing vertices in the order found by Dijkstra's Algorithm

	// Set the start time of Brandes' Algorithm

	// Compute Betweenness Centrality for every vertex i
	for (i=0; i < nvertices; i++) {
		/* Initialize */
		PredList.assign(nvertices, std::vector <int> (0, 0));
		d.assign(nvertices, Long_max);
		d[i] = 0;
		sigma.assign(nvertices, 0);
		sigma[i] = 1;
		delta.assign(nvertices, 0);
		Q.push(i);

		// Use Breadth First Search algorithm
		while (!Q.empty()) {
			// Get the next element in the queue
			u = Q.front();
			Q.pop();
			// Push u onto the stack S. Needed later for betweenness computation
			S.push(u);
			// Iterate over all the neighbors of u
			for (j=0; j < (int) adj_list[u].size(); j++) {
				// Get the neighbor v of vertex u
				// v = (ui64) network->vertex[u].edge[j].target;
				v = (int) adj_list[u][j];

				/* Relax and Count */
				if (d[v] == Long_max) {
					 d[v] = d[u] + 1;
					 Q.push(v);
				}
				if (d[v] == d[u] + 1) {
					sigma[v] += sigma[u];
					PredList[v].push_back(u);
				}
			} // End For

		} // End While

		/* Accumulation */
		while (!S.empty()) {
			u = S.top();
			S.pop();
			for (j=0; j < (int)PredList[u].size(); j++) {
				delta[PredList[u][j]] += ((double) sigma[PredList[u][j]]/sigma[u]) * (1+delta[u]);
			}
			if (u != i)
				CB[u] += delta[u];
		}

		// Clear data for the next run
		PredList.clear();
		d.clear();
		sigma.clear();
		delta.clear();
	} // End For

	// End time after Brandes' algorithm and the time difference

    for(int i =0; i<nvertices;++i){
        if (norm == 0)
        {
            CB[i] = 0;
        }
        else
        {
            CB[i]=CB[i]/norm;
        }
    }

	return CB;

} // End of BrandesAlgorithm_Unweighted