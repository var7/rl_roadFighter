function [ state_values ] = eval_policy( MDP, pi )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
state_values =  zeros(MDP.GridSize(1), MDP.GridSize(2));
old_state_values = zeros(MDP.GridSize(1), MDP.GridSize(2));
stop = false;
theta = 0.001;
c = 0;
while (stop == false)
%     delta = 0;
    for i = 1:MDP.GridSize(1)
        for j = 1:MDP.GridSize(2)
            agentLocation = [i j];
            old_state_values(i, j) = state_values(i, j);  % old v
            actionTaken = pi(i, j);
            [ possibleTransitions, probabilityForEachTransition ] = ...
                MDP.getTransitions(agentLocation, actionTaken);  % action
            num_transitions = size(possibleTransitions, 1);
%             rewards = zeros([num_transitions 1]);
            value = 0;
            for k = 1:num_transitions
                transitionLocation = possibleTransitions(k, :);
%                 agentLocation
                reward = MDP.getReward( ...
                    agentLocation, transitionLocation, actionTaken );
                value = ...
                    value + ...
                    probabilityForEachTransition(k) * ...
                    (reward + state_values(transitionLocation(1), ...
                    transitionLocation(2)));
            end
            state_values(i, j) = value;
        end
    end
    if (abs(state_values-old_state_values) == 0)
        stop = true;
    end
%     stop = true;
end
end

