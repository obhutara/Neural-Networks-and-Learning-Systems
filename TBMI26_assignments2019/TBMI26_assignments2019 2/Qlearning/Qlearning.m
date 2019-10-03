%% cleaning
clearvars;

%% initialise the qtable to zeros (actions are equal and no randomness)
Qtable = zeros (10 , 15, 4);
%leaving the gridworld cannot be the best action
Qtable (1 ,: ,2) = -inf ;
Qtable (10 ,: ,1) = -inf ;
Qtable (: ,1 ,4) = -inf ;
Qtable (: ,15 ,3) = -inf ;

%% training parameters
world = 5;
n_episodes = 300;
possibleAction = [1 2 3 4];
%actions are equally likely
probRandomActions = [0.25 0.25 0.25 0.25];
%The discount factor gamma (0 focus on short term rewards and 1 focus on long
%term rewards)
gamma = 0.9;
%epsilon -( exploration factor 1 on exploration and 0 on exploitation)
epsilon = 0.3;
% explore to exploit ( epsilon is reduced )
epsilonDecay = 0.998;
% Learning rate (0 focus on learned experience and 1 will overwrite previous experience with new info)   
eta = 0.5;

%% learning process
for i=1: n_episodes
% update the random action probability
epsilon = epsilon * epsilonDecay ;
% initialise the world
gwinit(world)
% drawing the world
gwdraw
% get state of the robot
state = gwstate();
while state.isterminal ~= 1
% saving old position
oldPosition = state.pos;
% choose an action based on the Qtable ( optimal ) or random
% adjust how often random actions are taken using the epsilon parameter
action = chooseaction(Qtable,oldPosition(1),oldPosition(2),possibleAction,epsilon,probRandomActions,[1-epsilon,epsilon]);
% take an action and get the resulting state of the robot
state = gwaction(action);
% update the Qtable only if the action was valid 
% (should not leave the gridworld)
if (state.isvalid)
Qtable(oldPosition(1),oldPosition(2),action) = (1 - eta)*Qtable(oldPosition(1), oldPosition(2), action) + eta*(state.feedback + gamma*max(Qtable(state.pos(1),state.pos(2), :)));
gwplotarrow(oldPosition, action);
end
end
end

%% plot the V function (the best actions depending on the state)
% draw the world without any arrows
gwdraw
% extract the optimal directions (optimal policy)
[~, I] = max (Qtable, [], 3);
% draw the arrows for each grid point
for xx = 1:10
for yy = 1:15
gwplotarrow ([xx; yy], I(xx, yy));
end
end