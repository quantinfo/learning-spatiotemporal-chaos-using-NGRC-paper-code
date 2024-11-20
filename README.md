# learning-spatiotemporal-chaos-using-NGRC-paper-code
This project contains the codes to reproduce the results presented in our paper: 

"Wendson A. S. Barbosa and Daniel J. Gauthier, _Learning Spatiotemporal Chaos using Next-Generation Reservoir Computing_, Chaos 32, 093137 (2022), https://doi.org/10.1063/5.0098707"


## Important code parameters

**L96system**: You can choose between three different set of paramters for the Lorenz 96 system: "hard", "medium" and "easy". They are related to types, number and timescales of the variables that compose the chaotic system.  

**integrate**: When using this code for the first time, you need to set "integrate = True" to integrate the Lorenz96 system equations and create the data set. It may take a while and the data file might be very large depending on your total integration time "t_total". The default is "t_total=11000" and the size of output file is 3.2 GB for "L96system = hard", 800 MB for "L96system = medium" and 3.5 GB for "L96system = easy". 

The rest of the code is carefully commented to guide you to reproduce the paper figures or create your own! 
