# RTS-PnO
Risk-aware Fund Allocation based on Time-Series Forcasting

### Run PnO Experiment
```
  python src/run_pno.py --config PNO_config
```

### Run PtO Experiment
First run the prediction stage
```
  python src/run_conformal.py --config Predict_config
```

Then run the optimization stage
```
  python src/run_pto.py --config PtO_config
```

### Run Heuristic Experiment

First run the prediction stage
```
  python src/run_conformal.py --config Predict_config
```

Then run the heuristic allocation stage
```
  python src/run_heuristic.py --config Heuristic_config
```


*We will open-source the config files after acceptance.*