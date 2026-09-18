"""Reproduce the deterministic SciPy 1.15.3 comparison fixtures.

Run with Python, NumPy, and SciPy 1.15.3 installed:
    python generate_scipy_fixtures.py > scipy_1_15_3.json

No Python dependency is required for the Rust test suite. The private solver
predicate is used only to inspect convergence before the first generation;
SciPy's public maxiter=0 result always reports an iteration-limit termination.
"""
import scipy

if scipy.__version__ != "1.15.3":
    raise RuntimeError("Reference fixture generation requires scipy==1.15.3")

import json
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

out = {}
common = dict(strategy="best1bin", mutation=0.0, recombination=1.0,
              updating="deferred", polish=False, rng=42, tol=0.0, atol=0.0)

def record(name, result):
    out[name] = dict(x=result.x.tolist(), fun=float(result.fun),
        population=result.population.tolist(), energies=result.population_energies.tolist(),
        generations=int(result.nit), evaluations=int(result.nfev), success=bool(result.success))

bounds = [(0.0,1.0), (0.0,1.0)]
init = np.array([[0.,0.], [0.25,0.25], [0.5,0.5], [0.75,0.75], [1.,1.]])
record("best1_zero_mutation", differential_evolution(lambda x: np.sum((x-0.25)**2),
       bounds, init=init, maxiter=1, **common))
clip_init = np.array([[5.,5.], [0.25,5.], [0.75,-5.], [5.,0.75], [0.5,0.5]])
record("initialization_clipping", differential_evolution(lambda x: np.sum(x*x),
       bounds, init=clip_init, maxiter=0, **common))
fixed_init = np.array([[-1.,3.,0.], [-0.5,3.,0.5], [0.,3.,1.], [0.5,3.,1.5], [1.,3.,2.]])
record("mixed_fixed_dimensions", differential_evolution(lambda x: (x[0]-0.25)**2+(x[2]-1.25)**2,
       [(-1.,1.),(3.,3.),(0.,2.)], init=fixed_init, maxiter=1, **common))

energies = np.array([0.,0.,0.,0.,5.])
thresholds = [float(np.nextafter(2.,0.)), 2., float(np.nextafter(2.,np.inf))]
checks = []
for atol in thresholds:
    cfg = dict(common, atol=atol)
    with DifferentialEvolutionSolver(lambda x: x[0], [(0.,5.)], init=energies[:,None], maxiter=0, **cfg) as solver:
        solver.population_energies[:] = energies
        convergence = bool(solver.converged())
    public = differential_evolution(lambda x: x[0], [(0.,5.)], init=energies[:,None], maxiter=0, **cfg)
    checks.append(dict(atol=atol, mean=float(np.mean(energies)), std=float(np.std(energies)),
                      criterion=convergence, public_success=bool(public.success), public_nit=int(public.nit)))
out["convergence_boundary"] = checks
print(json.dumps(out, indent=2))
