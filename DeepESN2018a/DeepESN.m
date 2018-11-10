classdef DeepESN < handle
    %DeepESN: Deep Echo State Network learning model
    %
    %Reference article:
    %C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
    %Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99
    %
    %
    %This file is part of the DeepESN18a Toolbox, November 2018
    %Claudio Gallicchio
    %gallicch@di.unipi.it - https://sites.google.com/site/cgallicch/
    %
    %Department of Computer Science - University of Pisa (Italy)
    %Computational Intelligence & Machine Learning (CIML) Group
    %http://www.di.unipi.it/groups/ciml/
    properties
        Nr %Reservoir dimension (i.e., number of recurrent units) in each layer [default = 10]
        Nu %Input dimension (i.e., number of input units) [default = 1]
        Ny %Output dimension (i.e., number of readout units) [default = 1]
        Nl %Number of reservoir layers [default = 10]
        
        spectral_radius %Spectral radius of the recurrent reservoir matrix in each layer [default = 0.9]
        input_scaling %Scaling of input connections [default = 1]
        inter_scaling %Scaling of inter reservoir conenctions for each layer > 1 [default = 1]
        leaking_rate %Leaking rate of each reservoir layer (this should be in (0,1]) [default = 1]
        input_scaling_mode %A string that describes the way in which input and inter-reservoir connections are scaled
        % e.g., 'bynorm' (scales the 2-norm of the matrices), 
        %       or 'byrange' (scales the interval from which to choose the weight values)
        % [default = 'bynorm']
        f %Activation function of the reservoir units (e.g., @tanh) [default = @tanh]
        bias %Input bias for reservoir and readout [default = 1]
        washout %Transient period for states washout [default = 1000]
        
        readout_regularization %Lambda for ridge regression training of the readout [default = 0]
        
        Win %Input-to-reservoir weight matrix
        Wil %Cell structure containing the inter-layer reservoir weight matrices. For each l > 1, Wil{l} contains
        % the weight values for the connections from layer (l-1) to layer l.
        W %Cell structure containing the recurrent reservoir weight matrices. For each l>1, W{l} contains
        % the recurrent weight values for layer l
        Wout %Reservoir-to-readout weight matrix
        
        run_states %Nr x Nt reservoir activation matrix corresponding to the last reservoir run (Nt is the number of time steps)
        initial_state %Initial state for the reservoir in each layer. This is a vector of size Nr x 1, used for all the Nl layers.
        state %Cell structure where the l-th element contains the present state of the reservoir in layer l
        
        
    end
    methods (Access = public)
        
        function self = default(self)
        %Set default values for the DeepESN properties
        self.Nr = 10;
        self.Nu = 1;
        self.Ny = 1;
        self.Nl = 10;
        
        self.spectral_radius = 0.9;
        self.input_scaling = 1;
        self.inter_scaling = 1;
        self.leaking_rate = 1;
        self.input_scaling_mode = 'bynorm';
        self.f = @tanh;
        self.bias = 1;
        self.washout = 1000;       
        
        self.readout_regularization = 0;
        
        self.Win = [];
        self.Wil = cell(self.Nl,1);
        self.W = cell(self.Nl,1);
        self.Wout = [];
        
        self.run_states = cell(self.Nl,1);
        self.initial_state = [];
        self.state = cell(self.Nl,1);
        end
        
        function self = DeepESN()
        %Class constructor. Set all the DeepESN properties to the default values
        self.default();
        end
        
        
        function init_state(self)
        %(Re-)initialize the state in each layer of the deep reservoir to initial_state
        for layer = 1:self.Nl
            self.state{layer} = self.initial_state;
            self.run_states{layer} = [];
        end
        end
        
        function initialize(self)
        %Initialize the deep reservoir of the DeepESN. This involves (i) setting up all the weight
        %matrices, and (ii) setting up the state of the deep reservoir
        
        %initialize the input-reservoir weight matrix
        self.Win = 2*rand(self.Nr,self.Nu+1)-1;
        %scale the input-to-reservoir weight matrix
        switch self.input_scaling_mode
            case 'bynorm'
                self.Win = self.input_scaling * self.Win / norm(self.Win);
            case 'byrange'
                self.Win = self.Win * self.input_scaling;
        end
        
        %initialize the inter-reservoir weight matrices
        for i = 2:self.Nl
            %initialization of the i-th inter-layer weight matrix
            self.Wil{i} = 2*rand(self.Nr,self.Nr+1)-1;
            %scaling of the i-th inter-layer weight matrix
            switch self.input_scaling_mode
                case 'bynorm'
                    self.Wil{i} = self.input_scaling * self.Wil{i} / norm(self.Wil{i});
                case 'byrange'
                    self.Wil{i} = self.Wil{i} * self.input_scaling;
            end
        end
        
        %initialize the recurrent reservoir weight matrices
        for i = 1:self.Nl
            %initialization of the i-th recurrent weight matrix
            self.W{i} = 2*rand(self.Nr,self.Nr)-1;
            %scaling of the i-th recurrent weight matrix
            I = eye(self.Nr);
            Wt = (1-self.leaking_rate) * I + self.leaking_rate * self.W{i};
            Wt = self.spectral_radius * Wt / max(abs(eig(Wt)));
            self.W{i} = (Wt - (1-self.leaking_rate) * I)/self.leaking_rate;
        end
        
        %initialize the state of the reservoir in each layer to a zero state
        self.initial_state = zeros(self.Nr,1);
        self.init_state();
        %note: if you need to initialize the reservoir states to an initial condition different
        %from the zero state, set the initial_state property to the desired initial conditions
        %and then call the init_state() method
        end
        
        
        
        function states = run(self,input)
        %Run the deep reservoir on the given input, returning the computed states.
        %Note(s):
        % - the execution of this function stores the computed states into the run_states property of
        % the DeepESN object
        % - this method does not reset the state of the reservoir to the initial conditions, call
        % method init_state (before this method) if you need to reset the state to initial
        % conditions before computing the reservoir states
        %
        % Parameter(s):
        % - input: An Nu x Nt matrix representing the input time-series.
        %          input(:,t) is input at time step t
        % Returned value(s):
        % - states: An Nlx1 cell containing the states computed by each layer of the deep reservoir.
        %           For each layer l, states{l} is an Nr x Nt matrix. 
        %           In particular, states{l}(:,t) is the state of the l-th reservoir layer at time step t
        
        Nt = size(input,2); %number of time steps in the input time-series
        % prepare the self.run_states variable
        for layer = 1:self.Nl
            self.run_states{layer} = zeros(self.Nr,Nt);
        end
        
        %run the deep reservoir on the given input
        old_state = self.state; %this is the first state
        for t = 1:Nt
            % t - time step under consideration
            for layer = 1:self.Nl
                % layer - layer under consideration
                x = old_state{layer}; %this plays the role of previous state
                u = []; %input for this specific layer
                %now focus on the specific input for the layer and compute the input_part in the
                %state update equation
                % (also the input bias is concatenated to the proper input)
                input_part = [];
                if layer == 1
                    u = input(:,t); %only the first layer receives the external input
                    input_part = self.Win * [u;self.bias];
                else
                    u = self.run_states{layer-1}(:,t); %successive layers receive in input
                    %the output of the previous layer
                    input_part = self.Wil{layer} * [u;self.bias];
                end
                self.state{layer} = (1-self.leaking_rate) * x + self.f(input_part + self.W{layer} * x);
                self.run_states{layer}(:,t) = self.state{layer};
                old_state{layer} = self.state{layer};
            end
        end
        states = self.run_states;
        end
                
        function self = train_readout(self,target)
        %Train the readout of DeepESN. Training is performed using the states corresponding to the
        %last deep reservoir run, i.e. those in self.run_states.
        %The readout is trained by ridge-regression with regularization parameter 
        %self.readout_regularization, to solve the least mean squares problem
        %self.Wout * X = target
        %where X is a proper concatenation of the states in each layer for all the
        %time steps under consideration)
        %Note(s):
        % - the first self.washout time steps are discarded as initial transient
        % 
        %Parameter(s):
        % - target: An Ny x Nt matrix representing the target (i.e., desired output) time-series.
        %           target(:,t) is the target at time step t.
        
        % organize a matrix containing a layer-wise concatenation of the network states
        X = self.shallow_states(self.run_states);
        % remove the washout transient
        X = X(:,self.washout+1:end);
        target = target(:,self.washout+1:end);
        
        % add the input bias
        X = [X;self.bias * ones(1,size(X,2))];
        
        % if the regularization term is zero then use pseudo-inverse training
        % otherwise use ridge-regression
        if self.readout_regularization == 0
            self.Wout = target * pinv(X);
        else
            self.Wout = target * X' / (X*X'+self.readout_regularization *eye(size(X,1)));
        end
        
        self.Ny = size(self.Wout,1); %also adjust the self.Ny value to the correct one given the target
        
        end
        
        function states = train(self, input, target)
        %Run the deep reservoir and train the readout of DeepESN on given input and target
        %time-series.
        %The deep reservoir is run on the input time-series (without state re-initialization),
        %computing the states in all the layers. After that, the readout weights are trained by
        %ridge-regression (with regularization parameter self.readout_regularization).
        %Note(s):
        % - the execution of this function stores the computed states into the run_states property of
        % the DeepESN object
        % - this method does not reset the state of the reservoir to the initial conditions, call
        % method init_state (before this method) if you need to reset the state to initial
        % conditions before computing the reservoir states.
        % - the first self.washout time steps are discarded as initial transient
        % 
        %Parameter(s):
        % - input: An Nu x Nt matrix representing the input time-series. 
        %          input(:,t) is input at time step t
        % - target: An Ny x Nt matrix representing the target (i.e., desired output) time-series.
        %           target(:,t) is the target at time step t
        %
        %Returned value(s):
        % - states: An Nl x 1 cell containing the states computed by each layer of the deep reservoir.
        %           For each layer l, states{l} is an Nr x Nt matrix. 
        %           In particular, states{l}(:,t) is the state of the l-th reservoir layer at time step t
        
        states = self.run(input);
        self.train_readout(target);
        end
        
        function output = compute_output(self,states,remove_washout)
        %Compute the output of the network for given states.
        %The output is computed by applying the linear readout layer to the patterns in the state
        %space.
        %
        %Parameters(s):
        % - states: A Nl x 1 cell structure, where each element is a Nr x Nt matrix with the states of
        % the corresponding layer in the network. In particular, for each layer l, states{l}(:,t) 
        % is the state of the l-th reservoir layer at time step t
        % - remove_washout: A binary flag which should be set to 1 if in the output computation
        % it is required to account for the initial washout of the network, discarding the first
        % self.washout time steps (otherwise set it to 0). In practice, set this value to 1 when
        % computing the output for a training set, to 0 in the case of validation or test set
        %
        %Returned value(s):
        % - output: An Ny x Nt matrix containing the output of the network for each time step, where
        % Nt is the length of the time-series under consideration (after washout removal).
        % In particular output(:,t) is the output of the network at time step t.
        
        states = self.shallow_states(states);
        if remove_washout
            states = states(:,self.washout+1:end);
        end
        output = self.Wout * [states;self.bias * ones(1,size(states,2))];
        end  
        
        
        
        function [outputTR,outputTS] = train_test(self,input,target,training,test)
        %Train the readout of DeepESN on given input and target time-series, specifying training and
        %test samples. The output of the DeepESN for training and test data is then returned in
        %output.
        %The deep reservoir is run on the input time-series (with state re-initialization),
        %computing the states in all the layers. Then, the readout weights are trained by
        %pseudo-inversion or ridge-regression (with regularization parameter self.readout_regularization).
        %After that, the output of the network is computed for the training samples.
        %Finally, states and output of the trained network are computed also on the test samples
        % 
        %Parameter(s):
        % - input: An Nu x Nt matrix representing the input time-series. 
        %          input(:,t) is input at time step t
        % - target: An Ny x Nt matrix representing the target (i.e., desired output) time-series.
        %           target(:,t) is the target at time step t
        %
        %Returned value(s):
        % - outputTR: An Ny x Ntr matrix containing the output of the network for each time step in
        % the training time-series (of length Ntr). In particular, outputTR(:,t) is the output of the network at time step t.
        % - outputTS: An Ny x Nts matrix containing the output of the network for each time step in
        % the test time-series (of length Nts). In particular, outputTS(:,t) is the output of the network at time step t.
        
        %initialize the state
        self.init_state(); 
        
        %train the network
        training_input = input(:,training);
        training_target = target(:,training);
        training_states = self.train(training_input,training_target);
        
        %compute the output of the model on the training set
        outputTR = self.compute_output(training_states,1);
        
        %compute the output of the model on the assessment data
        %first compute the states
        test_input = input(:,test);
        test_states = self.run(test_input); %here do not re-initialize the state
        outputTS = self.compute_output(test_states,0);
        
        end  
        
    end
    
   
    
        
    methods (Static)
        function perf = MSE(target, output)
        %Compute the Mean Squared Error given target and output data.
        perf = mean((target-output).^2);
        end
        
        function [perf,d] = MCscore(target, output)
        %Compute the score for the Memory Capacity task, given target and output data
        delays = size(target,1);
        for delay = 1:delays
            c = corrcoef(target(delay,:),output(delay,:));
            d(delay) = c(1,2)^2;
        end
        perf = sum(d);
        end
        
    end
    
    
    methods (Access = private)
        function X = shallow_states(self,states)
        %Converts the state representation of the deep reservoir from a layer-wise organization to a
        %global representation, in which at each time step the states of all layers are concatenated
        %in a row-wise fashion
        
        Nt = size(states{1},2); %number of time steps
        X = zeros(self.Nl * self.Nr,Nt); %this matrix will contain the input for the readout
        %i.e., for each time step (column): the states computed at each layer of the reservoir
        %concatenated along the rows dimension.
        for t = 1:Nt
            for layer = 1:self.Nl
                X(1+(layer-1)*self.Nr:self.Nr+(layer-1)*self.Nr,t) = states{layer}(:,t);
            end
        end
        end
    end
    
end
        