from deap import base, creator
from deap.benchmarks.tools import hypervolume as deap_hv

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def Hypervolume(df, ref_point=None):
    front = []
    for _, row in df.iterrows():
        ind = creator.Individual([row['acc'], row['ddp']])
        ind.fitness.values = (row['acc'], row['ddp'])
        front.append(ind)
        #front.append([-row['acc'], row['ddp']])

    if ref_point is None:
        ref_point = [0, 1]

    hv = deap_hv(front, ref_point)
    return hv
