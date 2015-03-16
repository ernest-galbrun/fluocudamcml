#pragma once
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <errno.h>

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <ctime>



#ifdef __linux__ //uses 25 registers per thread (64-bit)
	#define NUM_THREADS_PER_BLOCK 320 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS 17920
#endif

#ifdef _WIN32 //uses 26 registers per thread
	#define NUM_THREADS_PER_BLOCK 288 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS 16128
#endif




#define NUMSTEPS_GPU 1000
#define PI 3.141592654f
#define RPI 0.318309886f
#define MAX_LAYERS 12
#define STR_LEN 200

//#define WEIGHT 0.0001f
#define WEIGHTI 429497u //0xFFFFFFFFu*WEIGHT
#define CHANCE 0.1f

#define N_WAVELENGTH 800
#define MAX_FLUO_EMITTED 1000000



// TYPEDEFS

struct FluoStructAll
{
	float quantumYield;
	float absorptionSpectrum[N_WAVELENGTH];
	float emissionSpectrum[N_WAVELENGTH];
};

struct FluoStructDevice
{
	float mua;
	float quantumYield;
	float emissionDistributionFunction[200];// pour stocker le Randement quantique et tout
};

struct LayerStructDevice
{
	float z_min;		// Layer z_min [cm]
	float z_max;		// Layer z_max [cm]
	float mutr;			// Reciprocal mu_total [cm]
	float mus;
	float mua;			// Absorption coefficient [1/cm]
	float g;			// Anisotropy factor [-]
	float n;			// Refractive index [-]

	FluoStructDevice fluo[3];
	short int fluoNb;
	float thickness;
};


struct LayerStructAll
{
	float z_min;		// Layer z_min [cm]
	float z_max;		// Layer z_max [cm]
	float mutr[N_WAVELENGTH];			// Reciprocal mu_total [cm]
	float mus[N_WAVELENGTH];
	float mua[N_WAVELENGTH];			// Absorption coefficient [1/cm]
	float g[N_WAVELENGTH];			// Anisotropy factor [-]
	float n[N_WAVELENGTH];			// Refractive index [-]

	FluoStructAll fluo[3];
	short int fluoNb;
	float thickness;
};



struct /*__align__(16)*/ PhotonStruct
{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float dx;		// (Global, normalized) x-direction
	float dy;		// (Global, normalized) y-direction
	float dz;		// (Global, normalized) z-direction
	unsigned int weight;			// Photon weight
	int layer;				// Current layer

};

struct MemStruct
{
	PhotonStruct* p;					// Pointer to structure array containing all the photon data
	unsigned long long* x;				// Pointer to the array containing all the WMC x's
	unsigned int* a;					// Pointer to the array containing all the WMC a's
	unsigned int* thread_active;		// Pointer to the array containing the thread active status
	unsigned int* num_terminated_photons;	//Pointer to a scalar keeping track of the number of terminated photons

	unsigned long long* Rd_ra;
	unsigned long long* A_rz;			// Pointer to the 2D detection matrix!
	unsigned long long* Tt_ra;
	
	float* fluoRIn;
	float* fluoZIn;
	float* fluoWeightIn;	
	
	int*   num_emitted_fluo_photons;
	float* fluoROut;
	float* fluoZOut;
	float* fluoWeightOut;
	int*   fluoWavelengthOut;	
};



struct DetStruct
{
	float dr;		// Detection grid resolution, r-direction [cm]
	float dz;		// Detection grid resolution, z-direction [cm]

	float Zc;		// Critical depth
	
	int na;			// Number of grid elements in angular-direction [-]
	int nr;			// Number of grid elements in r-direction
	int nz;			// Number of grid elements in z-direction
};



struct SimulationStruct
{
	int lambda;
	float relativeLightIntensity;
	unsigned long number_of_photons;
	unsigned int n_layers;
	unsigned int start_weight;
	char outp_filename[STR_LEN];
	char inp_filename[STR_LEN];
	DetStruct det;
	LayerStructDevice* layers;
	
	// Fluorescence data : initialised to empty array
	// Populated during each simulation for the subsequent ones.
	// unsigned long fluo_photons;
	std::vector<float> fluoR;
	std::vector<float> fluoZ;
	std::vector<float> fluoWeight;
};




struct AllData
{
	int layersNb;
	double photonsNb;
	int lambdaMin;
	int lambdaMax;
	int lambdaStep;
	float dr,dz,da;
	LayerStructAll layers[12];
	float inputLightIntensity[N_WAVELENGTH];
};


int InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim);
int InitDCMem(SimulationStruct* sim);
int CopyDeviceToHostMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim);
void WriteHeader(std::string FolderPath,std::string OutputFileName, clock_t simulation_time);
int Write_Simulation_Results(MemStruct* HostMem, SimulationStruct* sim,std::string FolderPath,std::string OutputFileName);
void FreeMemStructs(MemStruct* HostMem, MemStruct* DeviceMem);
int interpret_arg(int argc, char* argv[], unsigned long long* seed, int* ignoreAdetection);
int read_simulation_data(std::string InputFileName, std::vector<SimulationStruct>* simulations, int ignoreAdetection, double Zc);
int init_RNG(char* exepath, unsigned long long *x, unsigned int *a, const unsigned int n_rng, unsigned long long xinit);
void DoOneSimulation(SimulationStruct* simulation, unsigned long long* x,unsigned int* a,
					 std::string FolderPath,std::string OutputFileName, std::vector<SimulationStruct>* allSimulations);
void	UpdateSimulation(std::vector<SimulationStruct>* allSimulations, MemStruct* HostMem);