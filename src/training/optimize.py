"""Small reproducible Optuna search; no test split is opened here."""
import copy
import optuna
from src.common import ROOT, read_json, write_json, experiment
from src.training.train import train

def search(prompt, task, n_trials=3):
    folder = experiment(f"optuna_p{prompt}_{task}")
    base = read_json(ROOT / "configs/training.json")
    study = optuna.create_study(direction="maximize", study_name=f"p{prompt}_{task}",
              storage="sqlite:///" + (folder / "study.db").as_posix(),
              sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=2),
              pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1))
    def objective(trial):
        cfg = copy.deepcopy(base)
        cfg.update(learning_rate=trial.suggest_float("learning_rate", 1e-5, 6e-5, log=True),
                   batch_size=trial.suggest_categorical("batch_size", [8,16]),
                   epochs=trial.suggest_int("epochs", 3, 5),
                   weight_decay=trial.suggest_float("weight_decay", .001, .1, log=True),
                   dropout=trial.suggest_float("dropout", .1, .3),
                   max_length=trial.suggest_categorical("max_length", [192,256]))
        result = train(prompt, task, cfg, stage=f"optuna{trial.number}", trial=trial)
        trial.set_user_attr("run", result["path"])
        return result["validation_qwk"]
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
    best = {"prompt":prompt, "task":task, "validation_qwk":study.best_value, "parameters":study.best_params,
            "path":study.best_trial.user_attrs["run"], "trials":len(study.trials),
            "note":"Compute-bounded exploratory search, not evidence of a global optimum."}
    write_json(folder / "best.json", best)
    return best
