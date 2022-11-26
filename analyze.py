import numpy as np
import mdtraj as md
from openmmtools.multistate import ReplicaExchangeSampler, ReplicaExchangeAnalyzer
import openmmtools

reporter = openmmtools.multistate.MultiStateReporter("output-1_checkpoint.nc")
analyzer = ReplicaExchangeAnalyzer(reporter)

(   replica_energies,
    unsampled_state_energies,
    neighborhoods,
    replica_state_indices,
) = analyzer.read_energies()


# data = np.load('coords-0.npy',allow_pickle=True).item()

r0 = analyzer.reporter.read_sampler_states(0)


# coords = []
# for step in range(replica_state_indices.shape[1]):
#     coords.append([])
#     ix = int(np.where(replica_state_indices[:,step] == 0)[0])
#     coords[-1].append(analyzer.reporter.read_sampler_states(step)[ix].positions)

# topo = md.load('alanine-dipeptide.pdb').topology
# coords = np.concatenate(coords,axis=0)
# traj = md.Trajectory(
#     coords, topo
# )
# traj.save_dcd('T-0.dcd')
