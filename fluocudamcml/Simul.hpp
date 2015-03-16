#include <vector>
#include <string>
#include <iostream>

#include "cuda_runtime.h"

//#include <boost/filesystem.hpp>

#define WEIGHTI 429497u //0xFFFFFFFFu*WEIGHT
#define CHANCE 0.1f

#define PI 3.141592654f
#define RPI 0.318309886f
#define NUMSTEPS_GPU 1000
#define DR 0.01f //dr in cm
#define NR 500U
#define NZ 1000U



const size_t N_WAVELENGTH =800;//800;
const int START_WAVELENGTH =200;
const size_t MAX_FIBERS =20;
const unsigned int MAX_LAYERS =10;
const size_t MAX_FLUO =3;

const size_t NUM_THREADS = 200;
const size_t NUM_BLOCKS = 100;

struct  PhotonStruct
{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float dx;		// (Global, normalized) x-direction
	float dy;		// (Global, normalized) y-direction
	float dz;		// (Global, normalized) z-direction
	unsigned int weight;			// Photon weight
	int layer;				// Current layer (from -1 (above layer) to nLayer (below)
    unsigned int lambda;
};

struct FluoDesc{
 float quantumYield;
 std::vector<float> mua;
 std::vector<float> emissionSpectrum;
};

struct Layer{
 std::vector<float> n;
 std::vector<float> mua;
 std::vector<float> mus;
 std::vector<float> g;
 std::vector<FluoDesc> fluo;
 float thickness;
};

struct Fiber{
 float position;
 float radius;
};


class Simulation{
	//cuda data :
 float4 *data;
 float2 *fluo_data;
 unsigned int *photonsToLaunch;
 unsigned int *photonsLaunched;
 float2* fibersDesc;
 float *zInterface;
 float *cu_nFiber;
 unsigned long long *result_fiber;
 unsigned long long *result_abs;
 unsigned long long *result_dif;	 
 unsigned long long *result_tr; // end_cuda data

 unsigned long long nPhotons;
 std::vector<Layer> layers;
 std::vector<float> nAbove;
 std::vector<float> nBelow;
 std::vector<float> nFiber;
 std::vector<float> excitationSpectrum;
 std::vector<Fiber> fibers;
 unsigned long long seed;


 void AddHybridDiffuseReflectance(float *source, float *result);
 void PrepareMemory();

public:
 ~Simulation();
 Simulation(std::string p);
 void LaunchSimulation(std::ostream &out, std::ostream &out_fibers);
 void SetSeed(unsigned long long _seed) {seed=_seed;};
};