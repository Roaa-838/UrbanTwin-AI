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


def test_traffic_parser_supports_resume_and_optional_mlflow(tmp_path):
    args = cli._parser().parse_args(
        [
            "traffic-run",
            "--resume",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--mlflow",
            "--mlflow-tracking-uri",
            "sqlite:///tracking.db",
        ]
    )

    assert args.resume is True
    assert args.checkpoint_dir == tmp_path / "checkpoints"
    assert args.mlflow is True
    assert args.mlflow_tracking_uri == "sqlite:///tracking.db"


def test_observed_parsers_support_current_roads_and_tracking(tmp_path):
    roads = cli._parser().parse_args(
        [
            "observed-roads",
            "--study-dir",
            str(tmp_path / "study"),
            "--output-dir",
            str(tmp_path / "roads"),
            "--pbf",
            str(tmp_path / "egypt.osm.pbf"),
            "--retry-attempts",
            "2",
        ]
    )
    study = cli._parser().parse_args(
        [
            "observed-study",
            "--road-dir",
            str(tmp_path / "roads"),
            "--resume",
            "--mlflow",
        ]
    )

    assert roads.study_dir == tmp_path / "study"
    assert roads.output_dir == tmp_path / "roads"
    assert roads.pbf == tmp_path / "egypt.osm.pbf"
    assert roads.retry_attempts == 2
    assert study.road_dir == tmp_path / "roads"
    assert study.resume is True
    assert study.mlflow is True


def test_cli_reports_interruption_and_returns_130(monkeypatch, capsys):
    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_traffic", interrupted)

    assert cli.main(["traffic-run"]) == 130
    assert "--resume" in capsys.readouterr().err
