clc; clear all;

%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;


%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis 
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance. 
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

%discountFactor_gamma = 1 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved 
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).

noCarOnRowProbability = 0.8 ; % the probability that there is no car 
% spawned for each row

seed = 1234;
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
    noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
    rewards );


%% Initialising the state observation (state features) and setting up the 
% exercise approximate Q-function:
stateFeatures = ones( 4, 5 );
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100;% obviously this is not a correctly computed Q-function; it does imply a policy however: Always go Up! (though on a clear road it will default to the first indexed action: go left)


%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by 
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.
ALGORITHM = 0; % MC
total_episodes = 1000;
discount = 1;
alpha = 0.01;
decay = .1;
weights = rand(4, 5, 3);

if ALGORITHM == 0
    disp('MC Policy evaluation')
    for episode = 1:total_episodes


        %%
        currentTimeStep = 0 ;
        MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
            blockSize, noCarOnRowProbability, ...
            probabilityOfUniformlyRandomDirectionTaken, rewards );
        currentMap = MDP ;
        agentLocation = currentMap.Start ;
        startingLocation = agentLocation ; % Keeping record of initial location.

        % If you need to keep track of agent movement history:
        agentMovementHistory = zeros(episodeLength+1, 2) ;
        agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;

        realAgentLocation = agentLocation ; % The location on the full test map.
        Return = 0;

        states = zeros(episodeLength, 4, 5);
        rewards = zeros(episodeLength, 1);
        actions = zeros(episodeLength, 1);
        step = 0;
        % COMMENT: For MC go through the whole episode till it ends
        % Record the action taken, the state visited, and the reward
        % obtained
        % THEN: Do a rollout of the actions, states and find the Return for a
        % given state, action pair
        % Do the MC estimate and update the function
        for i = episode:episodeLength
            step = step + 1;
            % Use the $getStateFeatures$ function as below, in order to get the
            % feature description of a state:
            stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns

            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
            end % for each possible action
    %         if i == 1
    %             actionTaken = randi(3);
    %         else
            [~, actionTaken] = max(action_values);
    %         end


            % The $GridMap$ functions $getTransitions$ and $getReward$ act as the
            % problems transition and reward function respectively.
            %
            % Your agent might not know these functions, but your simulator
            % does! (How wlse would we get our data?)
            %
            % $actionMoveAgent$ can be used to simulate agent (the car) behaviour.

            %     [ possibleTransitions, probabilityForEachTransition ] = ...
            %         MDP.getTransitions( realAgentLocation, actionTaken );
            %     [ numberOfPossibleNextStates, ~ ] = size(possibleTransitions);
            previousAgentLocation = realAgentLocation;

            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;

            %     MDP.getReward( ...
            %             previousAgentLocation, realAgentLocation, actionTaken )

            % COMMENT: Storing values needed
            rewards(step) = agentRewardSignal;
            actions(step) = actionTaken;
            states(step, :, :) = stateFeatures;
            Return = Return + agentRewardSignal;

            % If you want to view the agents behaviour sequentially, and with a
            % moving view window, try using $pause(n)$ to pause the screen for $n$
            % seconds between each draw:
    %         
    %         [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
    %             MDP, realAgentLocation, blockSize );
    %         % $agentLocation$ is the location on the viewable grid map for the
    %         % simulation. It is used by $refreshScreen$.
    %         
    %         currentMap = viewableGridMap ; %#ok<NASGU>
    %         % $currentMap$ is keeping track of which part of the full test map
    %         % should be printed by $refreshScreen$ or $printAgentTrajectory$.
    %         
    %         refreshScreen
    %         
    %         pause(0.15)

        end
    %     step
        prevWeight = weights;

        % COMMENT: rollout the episode
        for i=1:step
            stateFeatures = states(i); % state visited
            actionTaken = actions(i); % action taken in the state
            % find the experienced return of that state
            Return = 0; 
            for k = i:step
                Return = Return + discount^(k-i) * rewards(k);
            end
            % find our prev estimate of the value of the state, action pair
            q_value = sum(sum(weights(:,:,actionTaken) .* stateFeatures));
            % update our weights to correct for the difference in Return and
            % q_value using gradient descent update
            weights(:, :, actionTaken) = weights(:, :, actionTaken) + ...
                    alpha .* ((Return - q_value) .* stateFeatures);
        end
    %     disp("Weights after update")
    %     disp(weights)
        if abs(prevWeight - weights) < 1e-05
            disp("Weights are not changing anymore")
            episode
            break
        end
        % use an appriopriately decaying learning rate
        alpha = alpha * (1/(1 + decay * episode));
        currentMap = MDP ;
        agentLocation = realAgentLocation ;

    %     printAgentTrajectory
    %     pause(1)

    end % for each episode
else % TD 
    disp('TD Policy Evaluation')
    
    for episode = 1:total_episodes
        prevWeight = weights;

        %%
        currentTimeStep = 0 ;
        MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
            blockSize, noCarOnRowProbability, ...
            probabilityOfUniformlyRandomDirectionTaken, rewards );
        currentMap = MDP ;
        agentLocation = currentMap.Start ;
        startingLocation = agentLocation ; % Keeping record of initial location.

        % If you need to keep track of agent movement history:
        agentMovementHistory = zeros(episodeLength+1, 2) ;
        agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;

        realAgentLocation = agentLocation ; % The location on the full test map.
        Return = 0;
 
        % COMMENT: For TD go through the whole episode till it ends
        % Record the action taken, the state visited, and the reward
        % obtained
        % THEN: Do a rollout of the actions, states and find the Return for a
        % given state, action pair
        % Do the MC estimate and update the function
        for i = episode:episodeLength

            % Use the $getStateFeatures$ function as below, in order to get the
            % feature description of a state:
            stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4rows x 5columns

            for action = 1:3
                action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
            end % for each possible action

            [~, actionTaken] = max(action_values);
 


            % The $GridMap$ functions $getTransitions$ and $getReward$ act as the
            % problems transition and reward function respectively.
            %
            % Your agent might not know these functions, but your simulator
            % does! (How wlse would we get our data?)
            %
            % $actionMoveAgent$ can be used to simulate agent (the car) behaviour.

            %     [ possibleTransitions, probabilityForEachTransition ] = ...
            %         MDP.getTransitions( realAgentLocation, actionTaken );
            %     [ numberOfPossibleNextStates, ~ ] = size(possibleTransitions);
            previousAgentLocation = realAgentLocation;

            [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
                agentMovementHistory ] = ...
                actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
                currentTimeStep, agentMovementHistory, ...
                probabilityOfUniformlyRandomDirectionTaken ) ;

            %     MDP.getReward( ...
            %             previousAgentLocation, realAgentLocation, actionTaken )

            % COMMENT: Storing values needed
            
            nextStateFeatures = MDP.getStateFeatures(realAgentLocation);
            
            for action = 1:3
                next_action_values(action) = ...
                    sum ( sum( Q_test1(:,:,action) .* nextStateFeatures ) );
            end % for each possible action

            [q_value_next_state, nextActionTaken] = max(next_action_values);
            
            q_value = sum(sum(weights(:,:,actionTaken) .* stateFeatures));
            
            td_target = agentRewardSignal + discount * q_value_next_state;
%             actionTaken
%             weights(:, :, actionTaken)
            weights(:, :, actionTaken) = weights(:, :, actionTaken) + ...
                    alpha .* ((td_target - q_value) .* stateFeatures);
            % If you want to view the agents behaviour sequentially, and with a
            % moving view window, try using $pause(n)$ to pause the screen for $n$
            % seconds between each draw:
    %         
    %         [ viewableGridMap, agentLocation ] = setCurrentViewableGridMap( ...
    %             MDP, realAgentLocation, blockSize );
    %         % $agentLocation$ is the location on the viewable grid map for the
    %         % simulation. It is used by $refreshScreen$.
    %         
    %         currentMap = viewableGridMap ; %#ok<NASGU>
    %         % $currentMap$ is keeping track of which part of the full test map
    %         % should be printed by $refreshScreen$ or $printAgentTrajectory$.
    %         
    %         refreshScreen
    %         
    %         pause(0.15)

        end
    %     step
        
    %     disp("Weights after update")
    %     disp(weights)
        if abs(prevWeight - weights) < 1e-05
            disp("Weights are not changing anymore")
            episode
            break
        end
        % use an appriopriately decaying learning rate
        alpha = alpha * (1/(1 + decay * episode));
        currentMap = MDP ;
        agentLocation = realAgentLocation ;

    %     printAgentTrajectory
    %     pause(1)

    end % for each episode
end
weights