import pandas as pd

from urbantwin.traffic.benchmark import _paired_outputs


def test_paired_outputs_join_every_method_to_same_seed_baseline():
    rows = []
    for method, loss in [("fixed", 20.0), ("GA", 17.0), ("PSO", 18.0), ("random_search", 19.0)]:
        rows.append(
            {
                "scenario": "medium",
                "method": method,
                "seed": 101,
                "data_kind": "simulation_abstract",
                "mean_time_loss_s_completed": loss,
                "mean_waiting_time_s_completed": loss / 2,
                "mean_approach_queue_vehicles": 2.0,
                "completed_throughput_vehicles_per_sim_hour": 100.0,
                "completed_vehicles": 8,
                "unfinished_vehicles": 1,
                "pending_vehicles": 1,
                "completion_fraction_generated": 0.8,
                "simulated_co2_g_inserted_fleet": 100.0,
            }
        )

    deltas, summary = _paired_outputs(pd.DataFrame(rows))

    assert set(deltas["method"]) == {"GA", "PSO", "random_search"}
    ga = deltas.loc[deltas["method"] == "GA"].iloc[0]
    assert ga["delta_mean_time_loss_s_completed_method_minus_fixed"] == -3.0
    assert set(summary["runs"]) == {1}
