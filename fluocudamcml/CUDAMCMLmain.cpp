/////////////////////////////////////////////////////////////
//		CUDA-based Monte Carlo simulation of photon migration in layered media (CUDAMCML).
//	
//			Some documentation is avialable for CUDAMCML and should have been distrbuted along 
//			with this source code. If that is not the case: Documentation, source code and executables
//			for CUDAMCML are available for download on our webpage:
//			http://www.atomic.physics.lu.se/Biophotonics
//			or, directly
//			http://www.atomic.physics.lu.se/fileadmin/atomfysik/Biophotonics/Software/CUDAMCML.zip
//
//			We encourage the use, and modifcation of this code, and hope it will help 
//			users/programmers to utilize the power of GPGPU for their simulation needs. While we
//			don't have a scientifc publication describing this code, we would very much appreciate
//			if you cite our original GPGPU Monte Carlo letter (on which CUDAMCML is based) if you 
//			use this code or derivations thereof for your own scientifc work:
//			E. Alerstam, T. Svensson and S. Andersson-Engels, "Parallel computing with graphics processing
//			units for high-speed Monte Carlo simulations of photon migration", Journal of Biomedical Optics
//			Letters, 13(6) 060504 (2008).
//
//			To compile and run this code, please visit www.nvidia.com and download the necessary 
//			CUDA Toolkit and SKD. We also highly recommend the Visual Studio wizard 
//			(available at:http://forums.nvidia.com/index.php?showtopic=69183) 
//			if you use Visual Studio 2005 
//			(The express edition is available for free at: http://www.microsoft.com/express/2005/). 
//
//			This code is distributed under the terms of the GNU General Public Licence (see
//			below). 
//
///////////////////////////////////////////////////////////////

/*	This file is part of CUDAMCML.

    CUDAMCML is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.*/

#include "CUDAMCML.h"/*
#include "cudamcml.hcu"
#include "cutil.h"*/
using namespace std;

extern string InputFileName,OutputFileName,FullLaunchingPath,FolderPath,SafeprimesPath;
extern size_t found;

void	UpdateSimulation(vector<SimulationStruct>* allSimulations, MemStruct* HostMem){
	for (int i=0;i<*HostMem->num_emitted_fluo_photons;i++) {
		(*allSimulations)[HostMem->fluoWavelengthOut[i]].fluoR.push_back(HostMem->fluoROut[i]);
		(*allSimulations)[HostMem->fluoWavelengthOut[i]].fluoZ.push_back(HostMem->fluoZOut[i]);
		(*allSimulations)[HostMem->fluoWavelengthOut[i]].fluoWeight.push_back(HostMem->fluoWeightOut[i]);
	}
}



int main(int argc,char* argv[])
{ 
	int i;
	vector<SimulationStruct> simulations;
	unsigned long long seed = (unsigned long long) time(NULL);// Default, use time(NULL) as seed
	int ignoreAdetection = 0;
	if(argc<2)
	{
		printf("Main : Not enough input arguments!\n");
		return 1;
	}
	else
	{
		InputFileName=argv[1];
	}
	double Zc = -1;
	if (argc>=4)
		Zc = atof(argv[3]);
	if(interpret_arg(argc,argv,&seed,&ignoreAdetection)){return 1;}    
	// Creating path
	FullLaunchingPath=argv[0];
	found=FullLaunchingPath.find_last_of("/\\");
	FolderPath=FullLaunchingPath.substr(0,found);
	cout<<  FolderPath;
	InputFileName=argv[1];
	cout << FolderPath<< ' ' << InputFileName;
	read_simulation_data(InputFileName,&simulations, ignoreAdetection, Zc);
	
	if(simulations.size() == 0)
	{    
		printf("Main : Something wrong with read_simulation_data!\nCheck if the number of simulations\ndoesn't exceed 200 !\n");
		return 1;
	}
	else 
	{
		printf("Main : Let's go for  %d simulations !\n",simulations.size());
	}
	// Allocate memory for RNG's
	unsigned long long x[NUM_THREADS];
	unsigned int a[NUM_THREADS];
	if(init_RNG(argv[0],x, a, NUM_THREADS, seed)) return 1;

	for(i=0;i<(int)simulations.size();i++)
	{	
		printf("\n\n\n------Simulations [%d] ------\n",i);
		DoOneSimulation(&(simulations[i]),x,a,FolderPath,OutputFileName, &simulations);
	}
	return 0;  
	//break;
}



//
//__global__ void addition(float* TDevice)
//{
//	//SimulationStruct sim;
//	for ( int i=0 ; i<25 ; i++)
//	{
//		// 3 * i + j  avec j=0->X        j=1->Z      j=2->Poids
//		TDevice[3*i+0]+=i;
//	}
//}






