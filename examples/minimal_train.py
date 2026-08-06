from mirage.data.sources.synthetic import ClosedLoopSCMGenerator, SyntheticSCMConfig
from mirage.experiments.runner import train_experiment


if __name__ == "__main__":
    ClosedLoopSCMGenerator(SyntheticSCMConfig(n_steps=960, n_variables=8, n_regimes=2, max_lag=2)).prepare(
        "data/processed/synthetic/smoke"
    )
    print(train_experiment("configs/experiment/smoke.yaml").to_dict())

