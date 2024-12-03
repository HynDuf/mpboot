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
    foundBetterScore = false;
    reportCountIter = 10;
    doReportUsage = params->aco_report_usage;
    logFileName = (string) params->out_prefix + ".aco.csv";
    ofstream outFile(logFileName, ios::trunc);
    if (!outFile.is_open()) {
        outError("Failed to open the file: ", logFileName);
    }

    for (int i = 1; i < (int)nodes.size(); ++i) {
        outFile << "#" << nodeTagToString(getNodeTag(i));
        if (i < (int)nodes.size() - 1) {
            outFile << ',';
        }
    }
    outFile << '\n';
    outFile.close();

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

int ACOAlgo::getNextNode() {
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
            return edges[E].toNode;
        }
    }
    assert(0);
    return 0;
}

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
    if (!foundBetterScore) {
        // NNI
        isOnPath[0]++;
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
    reportPheroPercentage();
}

void ACOAlgo::reportUsage() {
    if (!doReportUsage) {
        return;
    }
    std::ofstream outFile(logFileName, std::ios::app);
    if (!outFile.is_open()) {
        outError("Failed to open the file: ", logFileName);
    }
    for (int i = 1; i < (int)nodes.size(); ++i) {
        outFile << nodes[i].cnt;
        if (i < (int)nodes.size() - 1) {
            outFile << ',';
        }
    }
    outFile << '\n';
    outFile.close();
    if (verbose_mode >= VB_MED) {
        for (int i = 1; i < (int)nodes.size(); ++i) {
            cout << nodeTagToString(getNodeTag(i)) << " : " << nodes[i].cnt << '\n';
        }
    };
}

void ACOAlgo::reportPheroPercentage() {
    if (verbose_mode >= VB_MED) {
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
}

void ACOAlgo::reportDifficulty() {
    int nnis = nodes[1].cnt;
    int sprs = nodes[2].cnt;
    int tbrs = nodes[3].cnt;

    int sum = nnis + sprs + tbrs;
    double p_nni = nnis / (double) sum;
    double p_spr = sprs / (double) sum;
    double p_tbr = tbrs / (double) sum;
    double pa_nni = p_nni / edges[0].prior;
    double pa_spr = p_spr / edges[1].prior;
    double pa_tbr = p_tbr / edges[2].prior;
    double suma = pa_nni + pa_spr + pa_tbr;
    pa_spr /= suma;
    pa_tbr /= suma;
    double dif = 0.5 * pa_spr + 1.0 * pa_tbr;
    ostringstream tem;
    tem << fixed << setprecision(3);
    tem << "Dataset Difficulty Estimate (0 -> 1): " << dif << '\n';
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
