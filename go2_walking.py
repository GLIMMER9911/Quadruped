import os
import signal
import sys
import time
import crocoddyl
import pinocchio
import numpy as np

from pinocchio.robot_wrapper import RobotWrapper

# load go2 robot
modelPath = "/home/dong/humanrobot/quadruped/robot/"
URDF_FILENAME = "go2_description.urdf"

Robot_Go2 = RobotWrapper.BuildFromURDF(modelPath + URDF_FILENAME, modelPath,
                pinocchio.JointModelFreeFlyer()) # Load URDF file
model = Robot_Go2.model

lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"

display = crocoddyl.MeshcatDisplay(
    Robot_Go2, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])

q0 = pinocchio.utils.zero(model.nq)
q0[2] = 0.325 # z
q0[6] = 1  # q.w 7+12
q0[7:] = [0., np.pi/4, -np.pi/2, 0., np.pi/4, -np.pi/2, 0.,np.pi/4,-np.pi/2, 0.,np.pi/4,-np.pi/2]

v0 = pinocchio.utils.zero(Robot_Go2.model.nv)
x0 = np.concatenate([q0, v0])
display.display([x0])
time.sleep(0.05)

from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution
Robot_Go2.model.referenceConfigurations["standing"] = q0
gait = SimpleQuadrupedalGaitProblem(Robot_Go2.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Setting up all tasks
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.25,
            "stepHeight": 0.15,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    },
    {
        "trotting": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    },
    {
        "pacing": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 5,
        }
    },
    {
        "bounding": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 5,
        }
    },
    {
        "jumping": {
            "jumpHeight": 0.15,
            "jumpLength": [0.0, 0.3, 0.0],
            "timeStep": 1e-2,
            "groundKnots": 10,
            "flyingKnots": 20,
        }
    },
]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "trotting":
            # Creating a trotting problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createTrottingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "pacing":
            # Creating a pacing problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createPacingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "bounding":
            # Creating a bounding problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createBoundingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "jumping":
            # Creating a jumping problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
            
    # Added the callback functions
    print("*** SOLVE " + key + " ***")
    if 0 :
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 100, False)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]

    # while True 这里容易出问题
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        time.sleep(1.0)