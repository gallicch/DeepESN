classdef Task < handle
    %Task: Class for representing learning tasks in Deep Echo State Networks applications
    %
    %
    %This file is part of the DeepESN18a Toolbox, November 2018
    %Claudio Gallicchio
    %gallicch@di.unipi.it - https://sites.google.com/site/cgallicch/
    %
    %Department of Computer Science - University of Pisa (Italy)
    %Computational Intelligence & Machine Learning (CIML) Group
    %http://www.di.unipi.it/groups/ciml/
    %
    %Reference article:
    %C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A
    %Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99
    properties
        name %A string name for the task
        input %Nu x Nt matrix where each column represents an input pattern. In particular,
              %input(:,t) is the input pattern at time step t.
        target %Ny x Nt matrix where each column represents a target pattern. In particular
               %target(:,t) is the target pattern at time step t.
        folds %A Nf x 1 cell structure where each element describes the splitting of the available data
              %for cross-fold validation purposes. In particular, double cross-fold validation is
              %supported, where for each fold f:
              % folds{f}.design contains the indices of the design samples 
              % folds{f}.test contains the indices of the test samples
              % Then, for each nested fold n:
              % folds{f}.training{n} contains the indices of the training samples
              % folds{f}.validation{n{ contains the indices of the validation samples
              %Note: for each fold f, the union of folds{f}.training{n} and folds{f}.validation{n}
              %gives the design set folds{f}.design.        
    end
    methods (Access = public)
        function self = default(self)
        %Set all the Task properties to the default values
            self.input = [];
            self.target = [];
            self.folds = cell(0,0);
            self.name = 'Default Task';
        end
        
        function self = Task()
        %Class constructor. Just set all the Task properties to the default values
            self.default();
        end
        
        %The following methods are used to fill the dataset with the needed information
        function self = set_name(self,name)
        %Set the name of the task
            self.name = name;
        end
        function self = set_data(self,input,target)
            %Set the input and target data to specific values.
            %Note that both input and target represent time-series, where
            %the number of rows is the size of the corresponding (input or
            %output) space, and the number of columns is the length of the
            %time-series. 
            %In this version only sequence transductions are supported where input and target time series 
            %are of the same length. As such, the number of columns in input and target should be
            %the same.
            
            if (size(input,2)~=size(target,2))
                warning('Input and target time-series should be of the same length.');
            else
                self.input = input;
                self.target = target;
            end
        end
        
        function self = set_holdout_folds(self,training,validation,design,test)
        %Organize the self.folds variable to contain the indices for 
        %holdout cross-validation scheme.
        %Parameters are used to represent the indices of training,
        %validation, design and test samples.
        %This function assumes that there is only one sequence, and splits
        %it on a time steps resolution.
        self.folds = [];
        self.folds{1}.design = design;
        self.folds{1}.test = test;
        self.folds{1}.training{1} = training;
        self.folds{1}.validation{1} = validation;
        end
 
    end
end