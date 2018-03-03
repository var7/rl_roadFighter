function [ new_pi ] = improve_policy( MDP, state_values, pi )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
new_pi = zeros(size(pi));

for i = 1:MDP.GridSize(1)
    for j = 1:MDP.GridSize(2)
%         b = pi(i, j)
        agentLocation = [i j];
        value = zeros([3 1]);
        for l = 1:3
            actionTaken = l;
            [ possibleTransitions, probabilityForEachTransition ] = ...
            MDP.getTransitions(agentLocation, actionTaken);  % action
            num_transitions = size(possibleTransitions, 1);
            for k = 1:num_transitions
                transitionLocation = possibleTransitions(k, :);
%                 agentLocation
                reward = MDP.getReward( ...
                    agentLocation, transitionLocation, actionTaken );
                value(l) = ...
                    value(l) + ...
                    probabilityForEachTransition(k) * ...
                    (reward + state_values(transitionLocation(1), ...
                    transitionLocation(2)));
            end
        end
        [~, argmax_a] = max(value);
        new_pi(i, j) = argmax_a;
    end
end
end

