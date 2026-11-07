from urbantwin.traffic.optimize import genetic_algorithm,particle_swarm
from urbantwin.traffic.baselines import green_phase_indices


def test_optimizers_on_deterministic_objective():
    fn=lambda pair:(pair[0]-27)**2+(pair[1]-42)**2
    for run in (genetic_algorithm,particle_swarm):
        result=run(fn,(12,55),seed=42)
        assert all(12<=x<=55 for x in result["plan"])
        assert result["objective"]>=0
        assert len(result["history"])>0


def test_green_phase_detection():
    class Phase:
        def __init__(self,state):self.state=state
    phases=[Phase("GGrr"),Phase("yyrr"),Phase("rrGG"),Phase("rryy")]
    assert green_phase_indices(phases)==[0,2]
