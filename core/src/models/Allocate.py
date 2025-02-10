import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

class AllocateModel(optGrbModel):
    def __init__(self, uncertainty, uncertainty_quantile=0.5):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = uncertainty_quantile
        self.uncertainty_bar = np.quantile(self.uncertainty, self.uncertainty_quantile)
        # print(self.uncertainty_bar)
        self.pred_len = len(self.uncertainty[0])
        super().__init__()

    def update_uncertainty(self, uncertainty):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_quantile = np.quantile(self.uncertainty, self.uncertainty_quantile)
        # print(self.uncertainty_bar)

    def _getModel(self): 
        # create a model
        optmodel = gp.Model()
        # variables
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS)
        # model sense
        optmodel.ModelSense = GRB.MINIMIZE
        # constraints
        optmodel.addConstr(gp.quicksum(self.uncertainty[0,i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)
        return optmodel, action

class AllocateModelOld(optGrbModel):
    def __init__(self, uncertainty, uncertainty_bar):
        self.uncertainty = np.array(uncertainty)
        self.uncertainty_bar = uncertainty_bar
        self.pred_len = len(self.uncertainty[0])
        super().__init__()

    def _getModel(self): 
        # create a model
        optmodel = gp.Model()
        # variables
        action = optmodel.addVars(self.pred_len, name="action", vtype=GRB.CONTINUOUS)
        # model sense
        optmodel.ModelSense = GRB.MINIMIZE
        # constraints
        optmodel.addConstr(gp.quicksum(self.uncertainty[0,i] * action[i] for i in range(self.pred_len)) <= self.uncertainty_bar)
        optmodel.addConstr(gp.quicksum(action[i] for i in range(self.pred_len)) == 1)
        return optmodel, action
