import torch

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import lzma
import pickle
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import numpy as np
from .train_pdm_scorer import PDMScorerConfig, PDMScorer
# from .train_pdm_scorer_v2_dev import PDMScorerConfig, PDMScorer

# --- numpy 2.x pickle compatibility shim (safe under numpy 1.26) ---
# The metric_cache pickles were generated with numpy>=2.0, whose arrays reference
# the 'numpy._core.*' module path. This environment ships numpy 1.26, which only
# exposes the legacy 'numpy.core.*'. Alias the new paths to the legacy ones so that
# pickle.load() can resolve them. This runs in every process that imports this
# module (including the Ray PDM-scoring workers).
import sys as _sys
import importlib as _importlib
import importlib.abc as _importlib_abc
import numpy.core as _np_core  # noqa: F401  ensure legacy numpy.core is importable


class _NumpyCoreCompatFinder(_importlib_abc.MetaPathFinder, _importlib_abc.Loader):
    def find_module(self, name, path=None):
        if name == "numpy._core" or name.startswith("numpy._core."):
            return self
        return None

    def load_module(self, name):
        if name in _sys.modules:
            return _sys.modules[name]
        module = _importlib.import_module(name.replace("numpy._core", "numpy.core", 1))
        _sys.modules[name] = module
        return module


def _install_numpy_core_compat():
    if not any(isinstance(_f, _NumpyCoreCompatFinder) for _f in _sys.meta_path):
        _sys.meta_path.append(_NumpyCoreCompatFinder())


# Install at import time (covers processes that import this module by reference,
# e.g. Ray workers created via cloudpickle-by-reference).
_install_numpy_core_compat()
# --- end shim ---

# metric_cache_loader = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache"))
proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig( )
scorer = PDMScorer(proposal_sampling, config)

def get_scores(args):

    return [get_sub_score(a["token"], a["poses"]) for a in args]


def get_sub_score(metric_cache, poses, test=None):

    # Ensure the numpy 2.x pickle compatibility shim is active in this (possibly
    # freshly-forked Ray worker) process before unpickling the metric_cache.
    _install_numpy_core_compat()
    with lzma.open(metric_cache, "rb") as f:
        metric_cache = pickle.load(f)

    initial_ego_state = metric_cache.ego_state

    trajectory_states = []

    for model_trajectory in poses:
        pred_trajectory = transform_trajectory(Trajectory(model_trajectory), initial_ego_state)

        pred_states = get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling,
                                              initial_ego_state.time_point)

        trajectory_states.append(pred_states)

    trajectory_states = np.stack(trajectory_states, axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)#32,41,11

    final_scores=scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress
    )

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, :  ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]


    return np.stack(
        [
            no_at_fault_collisions,
            drivable_area_compliance,
            ego_progress,
            time_to_collision_within_bound,
            comfort,
            driving_direction_compliance,
            final_scores,
        ],
        axis=-1,
    )
