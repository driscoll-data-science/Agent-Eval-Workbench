"""Agent Eval Bench: an MLflow-based evaluation kit for AI agents."""

import os as _os

# MLflow prints an agent hint on import; silence it for CLI users unless they opt back in.
_os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "0.1.0"
