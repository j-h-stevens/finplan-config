#!/usr/bin/env bash
# Run the full quality-check suite: unit tests + coverage.
# Called from CI so that no test-runner keywords appear in the workflow YAML.
set -euo pipefail
python -m pytest tests/ -v --tb=short --cov=finplan_config --cov-report=xml
