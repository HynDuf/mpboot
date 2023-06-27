/*
 * aco.h
 * Ant Colony Optimization
 */

#ifndef ACO_H_
#define ACO_H_
#include "iqtree.h"
#include "pllrepo/src/pll.h"
#include "tools.h"

class ACOAlgo {
  public:
    enum NodeTag {
        ROOT,
        RATCHET,
        IQP,
        RANDOM_NNI,
        NNI,
        SPR,
        TBR,
    };
    struct ACONode {
        NodeTag tag;
        int cnt;
        vector<int> adj;
        ACONode(NodeTag _tag) {
            tag = _tag;
            cnt = 0;
        }
    };
    struct ACOEdge {
        int fromNode;
        int toNode;
        double pheromone;
        double prior;
        ACOEdge(int _fromNode, int _toNode, double _prior) {
            fromNode = _fromNode;
            toNode = _toNode;
            prior = _prior;
            pheromone = PHERO_MAX;
        }
        void updateNewPhero(bool isOnPath, double EVAPORATION_RATE,
                            double c = 1.0) {
            // Smooth Max-Min Ant System (Huan, Linh-Trung, et al. 2012).
            // Default:           c = 1.0
            // Modified (TODO):   c = newScore / bestScore
            // (the better the score, the more it converges to PHERO_MAX)
            if (isOnPath) {
                pheromone = (1 - EVAPORATION_RATE) * pheromone +
                            EVAPORATION_RATE * PHERO_MAX * c;
            } else {
                pheromone = (1 - EVAPORATION_RATE) * pheromone +
                            EVAPORATION_RATE * PHERO_MIN;
            }
        }
    };

    // TODO: Try UPDATE_ITER 10% of num stop conditions
    int UPDATE_ITER;
    double EVAPORATION_RATE;
    static constexpr double PHERO_MAX = 1;
    static constexpr double PHERO_MIN = 0.001;
    int curNode;
    int curIter;
    int lastCounter;
    int curCounter;
    vector<int> par;
    vector<ACONode> nodes;
    vector<ACOEdge> edges;
    vector<pair<int, vector<int>>> savedPath;
    vector<bool> isOnPath;
    ACOAlgo();
    void setUpParamsAndGraph(Params *params);
    int moveNextNode();
    void addNode(NodeTag tag);
    void addEdge(int from, int to, double prior);
    void updateNewPheromonePath(vector<int> &edgesOnPath);
    void updateNewPheromone(int diffMP);
    void applyNewPheromone();
    void registerCounter();
    int getNumCounters();
    void reportUsage();
    void incCounter();

    NodeTag getNodeTag(int u) { return nodes[u].tag; }
    string nodeTagToString(NodeTag tag) {
        switch (tag) {
        case ROOT:
            return "ROOT";
        case RATCHET:
            return "RATCHET";
        case IQP:
            return "IQP";
        case RANDOM_NNI:
            return "RANDOM_NNI";
        case NNI:
            return "NNI";
        case SPR:
            return "SPR";
        case TBR:
            return "TBR";
        default:
            return "Unknown";
        }
    }
};

extern ACOAlgo *aco;
#endif /* ACO_H_ */