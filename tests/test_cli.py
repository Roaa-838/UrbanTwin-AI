from pathlib import Path

import pandas as pd
import pytest
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


def test_observed_error_analysis_parser_uses_explicit_private_inputs(tmp_path):
    args = cli._parser().parse_args(
        [
            "observed-error-analysis",
            "--bundle-dir",
            str(tmp_path / "bundle"),
            "--output-dir",
            str(tmp_path / ".urbantwin" / "private" / "analysis"),
            "--config",
            str(tmp_path / "analysis.yaml"),
            "--validation-root",
            str(tmp_path / "legacy-validation"),
            "--source-dir",
            str(tmp_path / "normalized-source"),
        ]
    )

    assert args.bundle_dir == tmp_path / "bundle"
    assert args.output_dir == tmp_path / ".urbantwin" / "private" / "analysis"
    assert args.config == tmp_path / "analysis.yaml"
    assert args.validation_root == tmp_path / "legacy-validation"
    assert args.source_dir == tmp_path / "normalized-source"


def test_observed_error_analysis_cli_dispatches_without_printing_metrics(
    tmp_path, monkeypatch, capsys
):
    requested = {}

    def run_observed_error_analysis(
        project_root,
        bundle_dir,
        output_dir,
        config,
        *,
        validation_root,
        source_dir,
    ):
        requested.update(
            project_root=project_root,
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            config=config,
            validation_root=validation_root,
            source_dir=source_dir,
        )
        return {"output_dir": output_dir}

    monkeypatch.setattr(
        "urbantwin.observed.error_analysis.run_observed_error_analysis",
        run_observed_error_analysis,
    )
    output_dir = tmp_path / ".urbantwin" / "private" / "analysis"
    bundle_dir = tmp_path / "bundle"
    config = tmp_path / "analysis.yaml"
    validation_root = tmp_path / "legacy-validation"
    source_dir = tmp_path / "normalized-source"

    assert (
        cli.main(
            [
                "observed-error-analysis",
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(output_dir),
                "--config",
                str(config),
                "--validation-root",
                str(validation_root),
                "--source-dir",
                str(source_dir),
            ]
        )
        == 0
    )
    assert requested == {
        "project_root": cli.ROOT,
        "bundle_dir": bundle_dir,
        "output_dir": output_dir,
        "config": config,
        "validation_root": validation_root,
        "source_dir": source_dir,
    }
    output = capsys.readouterr().out
    assert f"Private validated analysis: {output_dir}" in output
    assert "remain private" in output
    assert "MAE" not in output


def test_geoai_spatial_validation_parser_uses_separate_paths(tmp_path):
    args = cli._parser().parse_args(
        [
            "geoai-spatial-validate",
            "--data-dir",
            str(tmp_path / "eo"),
            "--output-dir",
            str(tmp_path / "spatial"),
            "--config",
            str(tmp_path / "protocol.yaml"),
        ]
    )

    assert args.data_dir == tmp_path / "eo"
    assert args.output_dir == tmp_path / "spatial"
    assert args.config == tmp_path / "protocol.yaml"

    figures = cli._parser().parse_args(
        [
            "geoai-spatial-figures",
            "--bundle-dir",
            str(tmp_path / "spatial"),
            "--output-dir",
            str(tmp_path / "figures"),
        ]
    )
    assert figures.bundle_dir == tmp_path / "spatial"
    assert figures.output_dir == tmp_path / "figures"


def test_cli_reports_interruption_and_returns_130(monkeypatch, capsys):
    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_traffic", interrupted)

    assert cli.main(["traffic-run"]) == 130
    assert "--resume" in capsys.readouterr().err


def test_geoai_spatial_validation_cli_runs_requested_protocol(tmp_path, monkeypatch, capsys):
    requested = {}

    def run_spatial_validation(data_dir, output_dir, config):
        requested.update(data_dir=data_dir, output_dir=output_dir, config=config)
        return pd.DataFrame([{"condition": "buffered_500m", "macro_f1": 0.4}])

    monkeypatch.setattr(
        "urbantwin.geoai.spatial_validation.run_spatial_validation",
        run_spatial_validation,
    )
    data_dir = tmp_path / "eo"
    output_dir = tmp_path / "bundle"
    config = tmp_path / "protocol.yaml"

    assert (
        cli.main(
            [
                "geoai-spatial-validate",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(output_dir),
                "--config",
                str(config),
            ]
        )
        == 0
    )

    assert requested == {"data_dir": data_dir, "output_dir": output_dir, "config": config}
    output = capsys.readouterr().out
    assert "buffered_500m" in output
    assert f"Validated robustness bundle: {output_dir}" in output


def test_geoai_spatial_figures_cli_dispatches_and_reports_validation_errors(
    tmp_path, monkeypatch, capsys
):
    requested = {}

    def render_figures(bundle_dir, output_dir, config, data_dir):
        requested.update(
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            config=config,
            data_dir=data_dir,
        )
        return [output_dir / "confusion.png", output_dir / "separation.png"]

    monkeypatch.setattr(
        "urbantwin.geoai.spatial_validation.render_spatial_validation_figures",
        render_figures,
    )
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "figures"
    config = tmp_path / "protocol.yaml"
    data_dir = tmp_path / "eo"
    argv = [
        "geoai-spatial-figures",
        "--bundle-dir",
        str(bundle_dir),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
        "--data-dir",
        str(data_dir),
    ]

    assert cli.main(argv) == 0
    assert requested == {
        "bundle_dir": bundle_dir,
        "output_dir": output_dir,
        "config": config,
        "data_dir": data_dir,
    }
    assert "Validated figures:" in capsys.readouterr().out

    def reject_bundle(*_args):
        raise ValueError("bundle provenance is inconsistent")

    monkeypatch.setattr(
        "urbantwin.geoai.spatial_validation.render_spatial_validation_figures",
        reject_bundle,
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 2
    assert "bundle provenance is inconsistent" in capsys.readouterr().err
