# OpenAI_problems
Tackling OpenAI RF problems like cartpole, atari games and more

The main idea behind this repo lies in tackling reinforcement learning problems through separation of agent, environment and learning algorithm 

# How to use it
Mostly of the relevant code lies in rl_agent and rl_learner scripts. When facing a new environment to solve (either from OpenAI gym or from a custom class that has reset() and step() methods) all what is currently left is to import abstract RL_Agent class, specify tensorflow graph from input to probabilities in the __init__ mathod of inherited class and feed that agent together with the environment to one of the learners 

# Currently supported learning rules
Policy gradient (Williams, 1992)
TRPO (Schulman et al., 2015)
