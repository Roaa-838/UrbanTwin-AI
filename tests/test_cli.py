from pathlib import Path

import yaml

from urbantwin import cli


def test_quick_config_uses_equal_optimizer_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = tmp_path / "experiments.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "traffic": {
                    "warmup_seconds": 60,
                    "vehicle_rates_per_approach_per_hour": {"low": 100},
                }
            }
        ),
        encoding="utf-8",
    )

    generated = cli._quick_config(source)
    traffic = yaml.safe_load(generated.read_text(encoding="utf-8"))["traffic"]

    assert traffic["ga_population"] * traffic["ga_generations"] == 4
    assert traffic["pso_particles"] * traffic["pso_iterations"] == 4
    assert traffic["optimizer_training_seeds"] == [17]
    assert traffic["evaluation_seeds"] == [101]
    assert generated == Path(tmp_path, "results", "quick_experiments.yaml")


def test_status_is_explicit_when_nothing_has_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    cli._print_status()

    output = capsys.readouterr().out
    assert "NOT RUN: real EO" in output
    assert "NOT RUN: synthetic demo data" in output
    assert "NOT RUN: real GeoAI results" in output
    assert "NOT RUN: SUMO results" in output
