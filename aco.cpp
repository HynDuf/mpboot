#include "aco.h"
ACOAlgo::ACOAlgo() {}
void ACOAlgo::setUpParamsAndGraph(Params *params) {
    // UPDATE_ITER = params->aco_update_iter;
    UPDATE_ITER =
        params->aco_update_iter + (int)params->unsuccess_iteration * 2 / 100;
    if (params->aco_heuristic_perturb) {
        cntHeuristicPerturb = (int)params->unsuccess_iteration * 20 / 100;
        cntHeuristicPerturb = (cntHeuristicPerturb + 2) / 3 * 3;
        scoreRATCHET = scoreIQP = scoreRANDOM_NNI = 0;
        timeRATCHET = timeIQP = timeRANDOM_NNI = 0;
    }
    EVAPORATION_RATE = params->aco_evaporation_rate;
    double RATCHET_PRIOR = params->aco_ratchet_prior;
    double IQP_PRIOR = params->aco_iqp_prior;
    double RANDOM_NNI_PRIOR = params->aco_random_nni_prior;
    double NNI_PRIOR = params->aco_nni_prior;
    double SPR_PRIOR = params->aco_spr_prior;
    double TBR_PRIOR = params->aco_tbr_prior;
    cout << "ACO Params: \n";
    cout << "UPDATE_ITER = " << UPDATE_ITER << '\n';
    cout << "EVAPORATION_RATE = " << EVAPORATION_RATE << '\n';
    cout << "RATCHET_PRIOR = " << RATCHET_PRIOR << '\n';
    cout << "IQP_PRIOR = " << IQP_PRIOR << '\n';
    cout << "RANDOM_NNI_PRIOR = " << RANDOM_NNI_PRIOR << '\n';
    cout << "NNI_PRIOR = " << NNI_PRIOR << '\n';
    cout << "SPR_PRIOR = " << SPR_PRIOR << '\n';
    cout << "TBR_PRIOR = " << TBR_PRIOR << '\n';

    addNode(ROOT);
    addNode(RATCHET);
    addNode(IQP);
    addNode(RANDOM_NNI);
    addNode(NNI);
    addNode(SPR);
    addNode(TBR);

   
    if (cntHeuristicPerturb == 0) {
        addEdge(ROOT, RATCHET, RATCHET_PRIOR);
        addEdge(ROOT, IQP, IQP_PRIOR);
        addEdge(ROOT, RANDOM_NNI, RANDOM_NNI_PRIOR);
    }

    for (int i = 1; i <= 3; ++i) {
        addEdge(i, NNI, NNI_PRIOR);
        addEdge(i, SPR, SPR_PRIOR);
        addEdge(i, TBR, TBR_PRIOR);
    }
    curIter = 0;
    curNode = ROOT;
    curCounter = 0;
    foundBetterScore = false;

    isOnPath.assign(edges.size(), false);
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
    if (cntHeuristicPerturb) {
        if (curNode != ROOT) {
            return TBR;
        }
        int tmp = cntHeuristicPerturb % 3;
        if (tmp == 0) {
            return curNode = RATCHET;
        }
        if (tmp == 1) {
            return curNode = IQP;
        }
        return curNode = RANDOM_NNI;
    }
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
    if (cntHeuristicPerturb) {
        --cntHeuristicPerturb;
        if (curNode == RATCHET) {
            scoreRATCHET += newScore;
            timeRATCHET  += numCounters; 
            // cout << "scoreRATCHET = " << newScore << '\n'; 
        }
        if (curNode == IQP) {
            scoreIQP += newScore;
            timeIQP  += numCounters; 
            // cout << "scoreIQP = " << newScore << '\n'; 
        }
        if (curNode == RANDOM_NNI) {
            scoreRANDOM_NNI += newScore;
            timeRANDOM_NNI  += numCounters; 
            // cout << "scoreRANDOM_NNI = " << newScore << '\n'; 
        }
        ++nodes[curNode].cnt;
        ++nodes[TBR].cnt;
        if (cntHeuristicPerturb == 0) {
            assert(nodes[RATCHET].cnt == nodes[IQP].cnt && nodes[IQP].cnt == nodes[RANDOM_NNI].cnt);

            cout << "scoreRATCHET = " << scoreRATCHET << '\n'; 
            cout << "scoreIQP = " << scoreIQP << '\n';
            cout << "scoreRANDOM_NNI = " << scoreRANDOM_NNI << '\n';
            
            cout << "timeRATCHET = " << timeRATCHET << '\n'; 
            cout << "timeIQP = " << timeIQP << '\n';
            cout << "timeRANDOM_NNI = " << timeRANDOM_NNI << '\n';

            long long sumScore = scoreRATCHET + scoreIQP + scoreRANDOM_NNI;
            long long sumTime = timeRATCHET + timeIQP + timeRANDOM_NNI;
            double RATCHET_PRIOR = 1.0 * (sumScore - scoreRATCHET) / sumScore * (sumTime - timeRATCHET) / sumTime;
            double IQP_PRIOR = 1.0 * (sumScore - scoreIQP) / sumScore * (sumTime - timeIQP) / sumTime;
            double RANDOM_NNI_PRIOR = 1.0 * (sumScore - scoreRANDOM_NNI) / sumScore * (sumTime - timeRANDOM_NNI) / sumTime;        
            cout << "RATCHET_PRIOR * 100 = " << RATCHET_PRIOR * 100 << '\n';
            cout << "IQP_PRIOR * 100 = " << IQP_PRIOR * 100 << '\n';
            cout << "RANDOM_NNI_PRIOR * 100 = " << RANDOM_NNI_PRIOR * 100 << '\n';
            cout << '\n';
            addEdge(ROOT, RATCHET, RATCHET_PRIOR);
            addEdge(ROOT, IQP, IQP_PRIOR);
            addEdge(ROOT, RANDOM_NNI, RANDOM_NNI_PRIOR);
        }
        return;
    }
    vector<int> edgesOnPath;
    int u = curNode;
    while (u) {
        int E = par[u];
        edgesOnPath.push_back(E);
        u = edges[E].fromNode;
    }
    if (newScore < curBestScore) {
        // cout << "P0\n";
        for (int i = 0; i < edges.size(); ++i) {
            isOnPath[i] = false;
        }
        for (int E : edgesOnPath) {
            isOnPath[E] = true;
            edges[E].updateNewPhero(true, EVAPORATION_RATE);
        }
        for (int i = 0; i < edges.size(); ++i) {
            if (!isOnPath[i]) {
                edges[i].updateNewPhero(false, EVAPORATION_RATE);
            }
            isOnPath[i] = false;
        }
        savedPath.clear();
        curIter = 0;
        curBestScore = newScore;
        foundBetterScore = true;
    } else if (foundBetterScore && newScore == curBestScore) {
        // cout << "P1\n";
        for (int E : edgesOnPath) {
            isOnPath[E] = true;
            edges[E].updateNewPhero(true, EVAPORATION_RATE);
        }
        // } else if (oldScore - newScore >= newScore - curBestScore) {
        //     cout << "P2\n";
        //     for (int E : edgesOnPath) {
        //         isOnPath[E] = true;
        //     }
    } else {
        // cout << "P3\n";
        savedPath.push_back({numCounters, edgesOnPath});
    }
    curNode = ROOT;
    curIter++;
    if (curIter == UPDATE_ITER) {
        applyNewPheromone();
        curIter = 0;
    }
}

void ACOAlgo::applyNewPheromone() {
    // Get the paths that is fastest
    sort(savedPath.begin(), savedPath.end(),
         [&](const pair<long long, vector<int>> &A,
             const pair<long long, vector<int>> &B) {
             return A.first < B.first;
         });
    // If there are less than half of UPDATE_ITER paths that have diffMP > 0,
    // Update using savedPath until there are half of paths updated
    // cout << "foundBetterScore = " << foundBetterScore << '\n';
    for (int i = 0;
         i < min((int)savedPath.size(),
                 UPDATE_ITER / 2 - (UPDATE_ITER - (int)savedPath.size()));
         ++i) {
        if (foundBetterScore) {
            for (int E : savedPath[i].second) {
                isOnPath[E] = true;
                // edges[E].updateNewPhero(true, EVAPORATION_RATE);
            }
        } else {
            for (int E : savedPath[i].second) {
                isOnPath[E] = true;
                edges[E].updateNewPhero(true, EVAPORATION_RATE);
            }
        }
    }
    savedPath.clear();
    for (int i = 0; i < edges.size(); ++i) {
        edges[i].updateNewPhero(isOnPath[i], EVAPORATION_RATE);
        isOnPath[i] = false;
    }
}

void ACOAlgo::reportUsage() {
    for (int i = 1; i < (int)nodes.size(); ++i) {
        cout << nodeTagToString(getNodeTag(i)) << " : " << nodes[i].cnt << '\n';
    }
}

void ACOAlgo::incCounter() { curCounter++; }
