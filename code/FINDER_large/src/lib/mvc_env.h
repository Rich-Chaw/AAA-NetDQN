#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include <memory>
#include "graph.h"
#include "graph_utils.h"
#include "disjoint_set.h"

class MvcEnv
{
public:
    MvcEnv(double _norm);

    ~MvcEnv();

    void s0(std::shared_ptr<Graph> _g);

    double step(int a);

    void stepWithoutReward(int a);

    void stepDelete(int a);

    double stepRollout(int a);

    std::vector<double> Betweenness(std::vector< std::vector <int> > adj_list);

    // action strategy
    int randomAction();
    int degreeAction();
    int betweenAction();

    bool isTerminal();
    // Episode Truncation with Early Termination
    bool isTruncated();

    bool isNoNodes();

    // Reward Reshaping via Heuristic Rollout
    double HeuristicRollout(double gamma);
    // Reward Reshaping via estimating with connected components
    double componentRollout(double gamma);

//    double getReward(double oldCcNum);
    double getReward();

    double estimateRemainingReturn(int rollout,double gamma);

    // Get the size of the largest connected component
    double getMaxConnectedNodesNum();

    double getNumofConnectedComponents();
    std::vector<int> getCcSizesDescending();
    //std:: vector<std::vector<int>> getCcDescending();
    // check env status
    void printGraph();
    
    double CcNum;

    

    double norm;

    std::shared_ptr<Graph> graph;

    std::vector< std::vector<int> > state_seq;

    std::vector<int> act_seq, action_list;

    std::vector<double> reward_seq, sum_rewards;

    int numCoveredEdges;

    std::set<int> covered_set;

    std::vector<int> avail_list;

    std::vector<int > node_degrees;

    // Episode Truncation parameters
    double trunc_threshold;  // N_trunc = 50% of original graph size
    double rollout_return;
};

#endif