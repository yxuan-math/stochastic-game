# stochastic-game

1) To run the code for training, you could simply run "CUDA_VISIBLE_DEVICES=0 python3 main.py"  if you are using GPU 0.  The tuple of (step, player, validation loss of player) will be shown every "logging_frequency" steps in running.

2) All the hyperparameters are in configs/CovidMulti3.json. Remarks:
a) "Infected" is not used since it is initialized in code. We just list it here in case we need it. b)
To enable decay learning rate, include the changing learning rate and changing stages in configs/CovidMulti3.json and use the minimizer commented in solver.py. c)"lockdowncost" is calculated by 172.6/a.

3) The trained model would be stored in data/debug.

4) After training, if you want to re-evaluate the trained model, you could modify " mode='train' " to " mode='restore' " in main.py and then kick off "CUDA_VISIBLE_DEVICES=0 python3 main.py".
