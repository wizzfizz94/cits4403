import networkx as nx
from constants import *
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def plotDecDiff(fig, g, j, pos):
    vac, non = [], []
    for i in g.nodes():
        if g.node[i]['decision'] == 1:
            vac.append(i)
        else:
            non.append(i)
    fig.add_subplot(2,2,j)
    nx.draw_networkx_nodes(g, pos,
                           nodelist=vac,
                           node_color='b',
                           alpha=0.5,
                           label="Pro Vaxx",
                           node_size=10)
    fig.add_subplot(2,2,j+2)
    nx.draw_networkx_nodes(g, pos,
                           nodelist=non,
                           node_color='r',
                           alpha=0.5,
                           label="Anti Vaxx",
                           node_size=10)

def makeLocalDecisions(g):
    n = g.nodes()
    random.shuffle(n)
    count = None
    # copy for social influence
    h = g.copy()
    while(count != 0):
        count = 0
        for i in n:
            dec = g.node[i]['decision']
            calcPerceivedRiskOfInfection(g, i)
            updateDecision(g, i)
            count += 1 if dec != g.node[i]['decision'] else 0
    addSocialInfluence(g, h)

def calcImpact(g):
    s, i, r = [], [], []
    R = I_rate = 0
    I = I_START
    v_tot = 0.0
    # exclude vaccinated individuals
    for j in g.nodes():
        if g.node[j]['decision'] == 1:
            v_tot += 1
    S = N - v_tot - I
    dS, dR, dI = None, None, None
    beta = R_0 * RECOVERY_RATE
    while (dS == None or np.absolute(dS) > 0.001
           or np.absolute(dI) > 0.001 or np.absolute(dR) > 0.001 ):
        s.append(S)
        i.append(I)
        r.append(R)
        lam = beta * I / N
        dS = -lam * S
        dI = lam * S - RECOVERY_RATE * I
        dR = RECOVERY_RATE * I
        S = max(S + dS, 0)
        I = max(I + dI, 0)
        R = max(R + dR, 0)
        I_rate -= dS
    return ([s, i, r], I_rate / N )

def plotSIR(sir_array):
    plt.figure()
    for i in sir_array:
        plt.plot(range(len(i)), i)

def addSocialInfluence(g, h):
    n = g.nodes()
    random.shuffle(n)
    for i in n:
        dec = g.node[i]['decision']
        l_vac = 0
        l_non = 0
        neighbours = h.neighbors(i)
        for j in neighbours:
            if h.node[j]['decision'] == 1:
                l_vac += g[i][j]['weight']
            else:
                l_non += g[i][j]['weight']

        l_diff = (l_vac - l_non)/(l_vac + l_non)
        prob = 1/(1 + math.exp(-RESPONSIVENESS*l_diff))
        sd = np.random.choice([1, -1], p=[prob, 1-prob])
        final = g.node[i]['decision'] = np.random.choice([g.node[i]['decision'], sd],
            p=[1-g.node[i]['social-influence'], g.node[i]['social-influence']])


'''changes individuals choice'''
def updateDecision(g, index):
    l = g.node[index]['percieved_risk']
    d = g.node[index]['decision']
    if COST_RATIO < l and d == -1:
        g.node[index]['decision'] = 1
        return 1
    elif COST_RATIO > l and d == 1:
        g.node[index]['decision'] = -1
        return 1
    return 0

'''Gets number of neighbours with decision to vaccinate'''
def getVacNeighbours(g, index):
    n = g.neighbors(index)
    n_vac = []
    for i in n:
        if g.node[i]['decision'] == 1:
            n_vac.append(i)
    return float(len(n_vac))

'''Gets number of neighbours with decision to not vaccinate'''
def getNonVacNeighbours(g, index):
    n = g.neighbors(index)
    n_non = []
    for i in n:
        if g.node[i]['decision'] == -1:
            n_non.append(i)
    return float(len(n_non))

'''calc func for perceived risk of disease infection'''
def calcPerceivedRiskOfInfection(g, index):
    n_vac = getVacNeighbours(g, index)
    n_non = getNonVacNeighbours(g, index)
    p_risk = g.node[index]['percieved_risk'] = PERCIEVED_INFECTION_RATE * (n_non / (n_non + n_vac))

def makeWSGraph():
    g = nx.watts_strogatz_graph(N, 6, 0.1)
    for i in g.nodes():
        dec = np.random.choice([1, -1], p=[1 - PROB_OF_ANTI_VAC, PROB_OF_ANTI_VAC])
        g.node[i]['decision'] = dec
        si = float(np.random.binomial(230, SOCIAL_INFLUENCE_FACTOR, 1)) / float(230)
        g.node[i]['social-influence'] = si
    for u, v in g.edges():
        g[u][v]['weight'] = float(np.random.binomial(230, CLOSENESS_FACTOR, 1)) / float(230)
    return g

def makeBAGraph():
    g = g = nx.barabasi_albert_graph(N, 5)
    for i in g.nodes():
        dec = np.random.choice([1, -1], p=[1 - PROB_OF_ANTI_VAC, PROB_OF_ANTI_VAC])
        g.node[i]['decision'] = dec
        si = float(np.random.binomial(230, SOCIAL_INFLUENCE_FACTOR, 1)) / float(230)
        g.node[i]['social-influence'] = si
    for u, v in g.edges():
        g[u][v]['weight'] = float(np.random.binomial(230, CLOSENESS_FACTOR, 1)) / float(230)
    return g

def makeSFGraph():
    multi_g = nx.scale_free_graph(N, alpha=0.4,
                                    beta=0.2,
                                    gamma=0.4,
                                    delta_in=0,
                                    delta_out=0
                                ).to_undirected()

    # covert to normal and add weights
    g = nx.Graph()
    for u, v, data in multi_g.edges_iter(data=True):
        w = float(np.random.binomial(230, CLOSENESS_FACTOR, 1)) / float(1000)
        if not g.has_edge(u, v):
            g.add_edge(u, v, weight=w)

    # add inital decisions and social influence to nodes
    for i in g.nodes():
        g.node[i]['decision'] = \
            np.random.choice([1, -1], p=[1 - PROB_OF_ANTI_VAC, PROB_OF_ANTI_VAC])
        g.node[i]['social-influence'] = float(np.random.binomial(1000, SOCIAL_INFLUENCE_FACTOR, 1)) / float(1000)
    return g

def calcCoverage(g):
    vac = 0
    for n in g.nodes():
        if g.node[n]['decision'] == 1:
            vac += 1
    return float(vac)/N


def plotCoverage(grain):

    x,y,dz = [],[],[]
    assert isinstance(grain, float)

    for i in range(int(grain)):
        for j in range(int(grain)):
            global SOCIAL_INFLUENCE_FACTOR
            SOCIAL_INFLUENCE_FACTOR = j/grain
            global COST_RATIO
            COST_RATIO = i/grain
            g = makeWSGraph()
            x.append(SOCIAL_INFLUENCE_FACTOR)
            y.append(COST_RATIO)
            makeLocalDecisions(g)
            dz.append(calcCoverage(g))

    fig = plt.figure()
    fig.suptitle("%d%% Pro Vaccine Initially" % (math.ceil(100 * (1 - PROB_OF_ANTI_VAC))))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("p factor")
    ax.set_ylabel("Cost ratio")
    ax.set_zlabel("Vaccine Coverage (%)")
    z = [0] * int(grain**2)
    cm = plt.get_cmap('gist_rainbow')
    cc = [cm(k/(grain**2)) for k in range(int(grain**2))]
    ax.bar3d(x, y, z, 1/grain, 1/grain, dz, color=cc)
    plt.show()

def plotAttackRates(grain):

    x,y,dz = [],[],[]
    assert isinstance(grain, float)

    for i in range(int(grain)):
        for j in range(int(grain)):
            global SOCIAL_INFLUENCE_FACTOR
            SOCIAL_INFLUENCE_FACTOR = j/grain
            global COST_RATIO
            COST_RATIO = i/grain
            g = makeWSGraph()
            x.append(SOCIAL_INFLUENCE_FACTOR)
            y.append(COST_RATIO)
            makeLocalDecisions(g)
            dz.append(calcImpact(g)[1])

    fig = plt.figure()
    fig.suptitle("%d%% Pro Vaccine Initially" % (math.ceil(100 * (1-PROB_OF_ANTI_VAC))))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("p factor")
    ax.set_ylabel("Cost ratio")
    ax.set_zlabel("Attack Rate for Infection")
    z = [0] * int(grain**2)
    cm = plt.get_cmap('gist_rainbow')
    cc = [cm(k/(grain**2)) for k in range(int(grain**2))]
    ax.bar3d(x, y, z, 1/grain, 1/grain, dz, color=cc)
    plt.show()

if __name__ == '__main__':
    # graphs = []
    # graphs.append(nx.watts_strogatz_graph(N, 4, 0.5))
    # graphs.append(nx.scale_free_graph(
    #     N, alpha=0.4, beta=0.2, gamma=0.4, delta_in=0, delta_out=0).to_undirected())
    # graphs.append(nx.barabasi_albert_graph(N, 2))
    # for i, g in enumerate(graphs):
    #     for i in g.nodes():
    #         g.node[i]['decision'] = \
    #             np.random.choice([1, -1], p=[1 - PROB_OF_ANTI_VAC, PROB_OF_ANTI_VAC])
    #         g.node[i]['social-influence'] = float(np.random.binomial(1000, SOCIAL_INFLUENCE_FACTOR, 1)) / float(1000)
    #     for u,v in g.edges():
    #         g[u][v]['weight'] = float(np.random.binomial(1000, CLOSENESS_FACTOR, 1)) / float(1000)
    # for g in graphs:
    #     plotDecDiff(g)
    #     makeLocalDecisions(g)
    #     plotDecDiff(g)
    #     addSocialInfluence(g)
    #     plotDecDiff(g)
    #     sir = calcImpact(g)
    #     plotSIR(sir)
    # plt.show()

    # show before and after social pressures
    # g = makeSFGraph()
    # plt.xticks([])
    # plt.yticks([])
    # fig = plt.figure()
    # fig.suptitle('Before vs After Social Influence')
    # makeLocalDecisions(g)
    # plotDecDiff(fig, g, 1)
    # addSocialInfluence(g)
    # plotDecDiff(fig, g, 2)
    # plt.show()

    # plotCoverage(15.0)
    # g = makeSFGraph()
    # makeLocalDecisions(g)
    # addSocialInfluence(g)
    # sir, ir = calcImpact(g)
    # plotSIR(sir)

    plotAttackRates(10.0)
    plotCoverage(10.0)
    plt.show()

