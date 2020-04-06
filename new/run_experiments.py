import argparse
import os
import numpy as np
from strategies.simulator import SimulationConfig, simulate
from strategies import random_nodes, learn_target_parents, edge_prob, var_score, information_gain, budgeted_experiment_design
from config import DATA_FOLDER
import causaldag as cd
from causaldag.classes.dag import CycleError
from multiprocessing import Pool, cpu_count
import random

np.random.seed(1729)
random.seed(1729)

NUM_STARTING_SAMPLES = 10000

parser = argparse.ArgumentParser(description='Simulate strategy for learning parent nodes in a causal DAG.')

parser.add_argument('--samples', '-n', type=int, help='number of samples')
parser.add_argument('--batches', '-b', type=int, help='number of batches allowed')
parser.add_argument('--max_interventions', '-k', type=int, help='maximum number of interventions per batch')
parser.add_argument('--intervention-strength', '-s', type=float,
                    help='number of standard deviations away from mean interventions occur at')
parser.add_argument('--boot', type=int, help='number of bootstrap samples')
parser.add_argument('--intervention-type', '-i', type=str)
parser.add_argument('--mbsize', '-m', type=int, help='Minibatch size')
parser.add_argument('--verbose', '-v', type=bool)
parser.add_argument('--target', type=int)

parser.add_argument('--folder', type=str, help='Folder containing the DAGs')
parser.add_argument('--strategy', type=str, help='Strategy to use')
parser.add_argument('--target-allowed', type=int, help='Whether or not the specified target node can be intervened on')

parser.add_argument('--starting-samples', type=int, help='Number of initial interventional samples') # A-ICP paper: To compare the effect of different observational sample sizes


args = parser.parse_args()

ndags = len(os.listdir(os.path.join(DATA_FOLDER, args.folder, 'dags')))
amats = [np.loadtxt(os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, 'adjacency.txt')) for i in range(ndags)] 
means = [np.loadtxt(os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, 'means.txt')) for i in range(ndags)] # A-ICP paper
variances = [np.loadtxt(os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, 'variances.txt')) for i in range(ndags)] # A-ICP paper
dags = [cd.GaussDAG.from_amat(amat, means=mean, variances=variance) for (amat,mean,variance) in zip(amats, means, variances)] # A-ICP paper: Allow variances/means different from 1/0, sampled at random
nnodes = len(dags[0].nodes)
target = args.target if args.target is not None else int(np.ceil(nnodes/2) - 1)
starting_samples = args.starting_samples if args.starting_samples is not None else NUM_STARTING_SAMPLES

SIM_CONFIG = SimulationConfig(
    starting_samples = starting_samples,
    n_samples=args.samples,
    n_batches=args.batches,
    max_interventions=args.max_interventions,
    strategy=args.strategy,
    intervention_strength=args.intervention_strength,
    target=target,
    intervention_type=args.intervention_type if args.intervention_type is not None else 'gauss',
    target_allowed=args.target_allowed != 0 if args.target_allowed is not None else True
)


def parent_functionals(target, nodes):
    def get_parent_functional(parent):
        def parent_functional(dag):
            return parent in dag.parents[target]
        return parent_functional

    return [get_parent_functional(node) for node in nodes if node != target]


def get_mec_functionals(dag_collection):
    def get_isdag_functional(dag):
        def isdag_functional(test_dag):
            return dag.arcs == test_dag.arcs
        return isdag_functional
    return [get_isdag_functional(dag) for dag in dag_collection]


def get_mec_functional_k(dag_collection):
    def get_dag_ix_mec(dag):
        return next(d_ix for d_ix, d in enumerate(dag_collection) if d.arcs == dag.arcs)
    return get_dag_ix_mec


def get_k_entropy_fxn(k):
    def get_k_entropy(fvals, weights):
        # find probs
        probs = np.zeros(k)
        for fval, w in zip(fvals, weights):
            probs[fval] += w

        # = find entropy
        mask = probs != 0
        plogps = np.zeros(len(probs))
        plogps[mask] = np.log2(probs[mask]) * probs[mask]
        return -plogps.sum()

    return get_k_entropy


def descendant_functionals(target, nodes):
    def get_descendant_functional(descendant):
        def descendant_functional(dag):
            return descendant in dag.downstream(target)
        return descendant_functional

    return [get_descendant_functional(node) for node in nodes if node != target]

def get_strategy(strategy, dag):
    if strategy == 'budgeted_exp_design':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        return budgeted_experiment_design.create_bed_strategy(dag_collection)
    if strategy == 'random':
        return random_nodes.random_strategy
    if strategy == 'random-smart':
        d = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        return random_nodes.create_random_smart_strategy(d.cpdag())
    if strategy == 'learn-parents':
        return learn_target_parents.create_learn_target_parents(target, args.boot)
    if strategy == 'edge-prob':
        return edge_prob.create_edge_prob_strategy(target, args.boot)
    if strategy == 'var-score':
        node_vars = np.diag(dag.covariance)
        return var_score.create_variance_strategy(target, node_vars, [2*np.sqrt(node_var) for node_var in node_vars])
    if strategy == 'entropy':
        return information_gain.create_info_gain_strategy(args.boot, parent_functionals(target, dag.nodes))
    if strategy == 'entropy-enum':
        return information_gain.create_info_gain_strategy(args.boot, parent_functionals(target, dag.nodes), enum_combos=True)
    if strategy == 'entropy-dag-collection':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        # mec_functionals = get_mec_functionals(dag_collection)
        mec_functional = get_mec_functional_k(dag_collection)
        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]
        # print([m(base_dag) for m in mec_functionals])

        gauss_iv = args.intervention_type == 'gauss'
        return information_gain.create_info_gain_strategy_dag_collection(dag_collection, [mec_functional], functional_entropies, gauss_iv, args.mbsize, verbose=args.verbose)
    if strategy == 'entropy-dag-collection-multiple-mec':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        other_dags = []
        non_reversible_arcs = list(base_dag.arcs - base_dag.reversible_arcs())
        random.shuffle(non_reversible_arcs)
        while len(other_dags) < 3:
            if len(non_reversible_arcs) == 0:
                break
            arc = non_reversible_arcs.pop()
            other_dag = base_dag.copy()
            try:
                other_dag.reverse_arc(*arc)
            except CycleError:
                pass
            if not any(other_dag.markov_equivalent(d) for d in other_dags) and len(other_dag.cpdag().all_dags()) < 25:
                other_dags.append(other_dag)
        print(other_dags)

        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        for other_dag in other_dags:
            dag_collection.extend([cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in other_dag.cpdag().all_dags()])
        print('length of dag collection:', len(dag_collection))
        mec_functional = get_mec_functional_k(dag_collection)
        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]

        gauss_iv = args.intervention_type == 'gauss'
        return information_gain.create_info_gain_strategy_dag_collection(dag_collection, [mec_functional], functional_entropies, gauss_iv, args.mbsize, verbose=args.verbose)

    if strategy == 'entropy-dag-collection-descendants':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        binary_entropy_fxn = get_k_entropy_fxn(2)
        d_functionals = descendant_functionals(target, dag.nodes)
        d_functionals_entropies = [binary_entropy_fxn] * len(d_functionals)
        gauss_iv = args.intervention_type == 'gauss'
        return information_gain.create_info_gain_strategy_dag_collection(dag_collection, d_functionals, d_functionals_entropies, gauss_iv, args.mbsize, verbose=args.verbose)

    if strategy == 'entropy-dag-collection-enum':
        base_dag = cd.DAG(nodes=set(dag.nodes), arcs=dag.arcs)
        dag_collection = [cd.DAG(nodes=set(dag.nodes), arcs=arcs) for arcs in base_dag.cpdag().all_dags()]
        # mec_functionals = get_mec_functionals(dag_collection)
        mec_functional = get_mec_functional_k(dag_collection)

        functional_entropies = [get_k_entropy_fxn(len(dag_collection))]
        # print([m(base_dag) for m in mec_functionals])
        return information_gain.create_info_gain_strategy_dag_collection_enum(dag_collection, [mec_functional], functional_entropies)


folders = [
    os.path.join(DATA_FOLDER, args.folder, 'dags', 'dag%d' % i, args.strategy + ',n=%s,b=%s,k=%s' % (args.samples, args.batches, args.max_interventions))
    for i in range(ndags)
]


def simulate_(tup):
    gdag, folder, num = tup
    dag = cd.DAG(nodes=set(gdag.nodes), arcs=gdag.arcs)
    print('SIMULATING FOR DAG: %d' % num)
    print('Folder:', folder)
    print('Size of MEC:', len(dag.cpdag().all_dags()))
    return simulate(get_strategy(args.strategy, gdag), SIM_CONFIG, gdag, folder, save_gies=True, dag_num = num) # A-ICP paper: set save_gies to True


print("Number of workers: %d" % cpu_count() - 1)
    
with Pool(cpu_count()-1) as p:
    result = p.map(simulate_, zip(dags, folders, range(len(dags))))

# A-ICP paper: Store posterior results
import pickle
import time

filename = "pp_%d" % time.time()
for (k,v) in vars(SIM_CONFIG).items():
    filename += "_%s:%s" % (k,v)
filename += ".pickle"

pickle.dump(result, open(filename, "wb"))

