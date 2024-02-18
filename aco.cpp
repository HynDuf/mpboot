#include "aco.h"
#include <iomanip>
#include <sstream>
ACOAlgo::ACOAlgo() {}
void ACOAlgo::setUpParamsAndGraph(Params *params) {
    // UPDATE_ITER = params->aco_update_iter;
    UPDATE_ITER =
        params->aco_update_iter + (int)params->unsuccess_iteration / 100;
    EVAPORATION_RATE = params->aco_evaporation_rate;
    double NNI_PRIOR = params->aco_nni_prior;
    double SPR_PRIOR = params->aco_spr_prior;
    double TBR_PRIOR = params->aco_tbr_prior;
    cout << "ACO Params: \n";
    cout << "UPDATE_ITER = " << UPDATE_ITER << '\n';
    cout << "EVAPORATION_RATE = " << EVAPORATION_RATE << '\n';
    cout << "NNI_PRIOR = " << NNI_PRIOR << '\n';
    cout << "SPR_PRIOR = " << SPR_PRIOR << '\n';
    cout << "TBR_PRIOR = " << TBR_PRIOR << '\n';

    IS_ACO_ONCE = params->aco_once;

    addNode(ROOT);
    addNode(NNI);
    addNode(SPR);
    addNode(TBR);

    addEdge(ROOT, NNI, NNI_PRIOR);
    addEdge(ROOT, SPR, SPR_PRIOR);
    addEdge(ROOT, TBR, TBR_PRIOR);

    curIter = 0;
    curNode = ROOT;
    curCounter = 0;
    foundBetterScore = false;
    curBestRatio = 0;
    bestEdge = -1;
    reportCountIter = 10;

    isOnPath.assign(edges.size(), 0);
}
void ACOAlgo::addNode(NodeTag tag) {
    nodes.push_back(ACONode(tag));
    par.push_back(0);
}

void ACOAlgo::addEdge(int from, int to, double prior) {
    int edgeId = ACOAlgo::edges.size();
    edges.push_back(ACOEdge(from, to, prior));
    nodes[from].adj.push_back(edgeId);
}

void ACOAlgo::registerCounter() { lastCounter = curCounter; }

long long ACOAlgo::getNumCounters() { return curCounter - lastCounter; }

int ACOAlgo::moveNextNode() {
    double sum = 0;
    int u = curNode;
    for (int i = 0; i < nodes[u].adj.size(); ++i) {
        int E = nodes[u].adj[i];
        double prob = edges[E].pheromone * edges[E].prior;
        sum += prob;
    }
    double random = random_double() * sum;
    sum = 0;
    for (int i = 0; i < nodes[u].adj.size(); ++i) {
        int E = nodes[u].adj[i];
        double prob = edges[E].pheromone * edges[E].prior;
        sum += prob;
        if (random < sum || i == nodes[u].adj.size() - 1) {
            curNode = edges[E].toNode;
            par[curNode] = E;
            nodes[curNode].cnt++;
            return curNode;
        }
    }
    assert(0);
    return 0;
}

void ACOAlgo::updateNewPheromone(int oldScore, int newScore) {
    // numCounters measures how long the chosen hill-climbing procedure ran
    long long numCounters = getNumCounters();
    vector<int> edgesOnPath;
    int u = curNode;
    while (u) {
        int E = par[u];
        edgesOnPath.push_back(E);
        u = edges[E].fromNode;
    }
    if (newScore < curBestScore) {
        foundBetterScore = true;
    }
    if (newScore < curBestScore || (foundBetterScore && newScore == curBestScore)) {
        for (auto E: edgesOnPath) {
            isOnPath[E]++;
        }
        curBestScore = newScore;
        bestEdge = -2;
    } else if ((oldScore - newScore + 1) / (float)numCounters > curBestRatio) {
        curBestRatio = (oldScore - newScore + 1) / (float)numCounters;
        if (bestEdge != -2) {
            bestEdge = edgesOnPath[0];
        }
        // for (auto E: edgesOnPath) {
        //     isOnPath[E]++;
        // }
    }
    curNode = ROOT;
    curIter++;
    if (curIter == UPDATE_ITER) {
        applyNewPheromone();
        curIter = 0;
    }
    if (--reportCountIter <= 0) {
        reportCountIter = 10;
        aco->reportUsage();
    }
}

void ACOAlgo::applyNewPheromone() {
    if (bestEdge >= 0) {
        isOnPath[bestEdge]++;
    }
    for (int i = 0; i < edges.size(); ++i) {
        if (IS_ACO_ONCE) {
            edges[i].updateNewPhero(isOnPath[i] > 0, EVAPORATION_RATE);
        } else {
            if (isOnPath[i] == 0) {
                edges[i].updateNewPhero(false, EVAPORATION_RATE);
            } else {
                for (int j = 1; j <= isOnPath[i]; ++j) {
                    edges[i].updateNewPhero(true, EVAPORATION_RATE);
                }
            }
        }
        isOnPath[i] = 0;
    }
    curBestRatio = 0;
    bestEdge = -1;
    reportPheroPercentage();
}

void ACOAlgo::reportUsage() {
    for (int i = 1; i < (int)nodes.size(); ++i) {
        cout << nodeTagToString(getNodeTag(i)) << " : " << nodes[i].cnt << '\n';
    }
}

void ACOAlgo::incCounter() { curCounter++; }

void ACOAlgo::reportPheroPercentage() {
    double p_nni = edges[0].pheromone;
    double p_spr = edges[1].pheromone;
    double p_tbr = edges[2].pheromone;

    double sum = p_nni + p_spr + p_tbr;
    p_nni /= sum;
    p_spr /= sum;
    p_tbr /= sum;
    ostringstream tem;
    tem << "%Phero:\n";
    tem << fixed << setprecision(3);
    tem << "PER_NNI = " << p_nni << '\n';
    tem << "PER_SPR = " << p_spr << '\n';
    tem << "PER_TBR = " << p_tbr << '\n';
    string temStr = tem.str();
    cout << temStr;
}

int ACOAlgo::getNumStopCond(int unsuccess_iters) {
    double p_nni = edges[0].pheromone;
    double p_spr = edges[1].pheromone;
    double p_tbr = edges[2].pheromone;

    double sum = p_nni + p_spr + p_tbr;
    p_nni /= sum;
    p_spr /= sum;
    p_tbr /= sum;
    return int(p_nni * unsuccess_iters + p_spr * unsuccess_iters + p_tbr * 100);
}
