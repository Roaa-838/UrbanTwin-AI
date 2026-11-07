from urbantwin.traffic.baselines import green_phase_indices
from urbantwin.traffic.optimize import genetic_algorithm, particle_swarm, random_search


def test_optimizers_on_deterministic_objective():
    objective = lambda pair: (pair[0] - 27) ** 2 + (pair[1] - 42) ** 2
    for optimizer in (genetic_algorithm, particle_swarm):
        result = optimizer(objective, (12, 55), seed=42)
        assert all(12 <= value <= 55 for value in result["plan"])
        assert result["objective"] >= 0
        assert len(result["history"]) > 0


def test_equal_budget_random_search_and_heuristics_count_every_evaluation():
    budget = 8
    calls: dict[str, list[tuple[int, int]]] = {}

    def run(name, optimizer, **settings):
        calls[name] = []

        def score(plan):
            calls[name].append(tuple(plan))
            return (plan[0] - 20) ** 2 + (plan[1] - 30) ** 2

        result = optimizer(score, (12, 55), seed=9, **settings)
        assert result["objective_evaluations"] == budget
        assert len(calls[name]) == budget
        return result

    run("GA", genetic_algorithm, population=4, generations=2)
    run("PSO", particle_swarm, particles=4, iterations=2)
    first = run("random", random_search, evaluations=budget)
    second = random_search(
        lambda plan: (plan[0] - 20) ** 2 + (plan[1] - 30) ** 2,
        (12, 55),
        evaluations=budget,
        seed=9,
    )
    assert first["plan"] == second["plan"]
    assert first["history"] == second["history"]
    assert len(set(calls["random"])) == budget


def test_green_phase_detection():
    class Phase:
        def __init__(self, state):
            self.state = state

    phases = [Phase("GGrr"), Phase("yyrr"), Phase("rrGG"), Phase("rryy")]
    assert green_phase_indices(phases) == [0, 2]
