DeepESN2018a - Deep Echo State Network Toolbox v1.0 (November 2018)

** GENERAL INFORMATION **
Deep Echo State Networks (DeepESN) extends the Reservoir Computing paradigm towards the Deep Learning framework. 
Essentially, a DeepESN is a deep Recurrent Neural Network composed of a stacked composition of multiple recurrent reservoir layers, and of a linear readout layer that computes the output of the model. The deep reservoir part is left untrained after initialization, and the readout is the only part of the architecture that undergoes a training process.
All details on the DeepESN model can be found in the reference paper reported below in the  CITATION REQUEST section.
Also note that DeepESNs with a single layer reduce to standard Echo State Networks (ESNs), thereby the code provided in this toolbox can also be used for standard (i.e., shallow) ESN applications.

The toolbox contains the files listed below.
- DeepESN.m: The file contains the definition of the class DeepESN (the main class in the toolbox).
- Task.m: The file contains the definition of the auxiliary class Task.
- example_DeepESN_1.m: The file contains an example of the usage of the classes in the DeepESN toolbox for the short-term Memory Capacity (MC) task.
- MC100.mat: The file contains an object of class Task, representing the information for the MC task (up to 100 reservoir units), used in the provided example code.

All the files come with full documentation, accessible through the individual reference pages, or through the help function. E.g., for info on the DeepESN class, type 'help DeepESN' in the Matlab command window.

** CITATION REQUEST **
The DeepESN model has been proposed in the following journal paper, which represents a citation request for the usage of this toolbox:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis", Neurocomputing, 2017, vol. 268, pp. 87-99

** FURTHER READING **
An up-to-date overview of the research developments on DeepESN can be found in:
C. Gallicchio, A. Micheli, "Deep Echo State Network (DeepESN): A brief survey", arXiv preprint arXiv:171204323, 2018

** AUTHOR INFORMATION **
Claudio Gallicchio
gallicch@di.unipi.it - https://sites.google.com/site/cgallicch/

Department of Computer Science - University of Pisa (Italy)
Computational Intelligence & Machine Learning (CIML) Group
http://www.di.unipi.it/groups/ciml/

