*	This file is part of CUDAMCML.

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


#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <errno.h>
#include "CUDAMCML.h"
#include "cutil.h"
//#include "cudamcml.hcu"

#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;


#define NUM_BLOCKS 56 //Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)
//The register usage varies with platform. 64-bit Linux and 32.bit Windows XP have been tested.
// Global Variables :

__device__ __constant__ unsigned int n_layers_dc[1];	
__device__ __constant__ unsigned int start_weight_dc[1];	
__device__ __constant__ LayerStructDevice layers_dc[MAX_LAYERS];	
__device__ __constant__ DetStruct det_dc[1];
__device__ __constant__ unsigned int photonToLaunch[1];
__device__ __constant__ unsigned int photonToLaunchFluo[1];



// forward declaration of the device code
__global__ void MCd(MemStruct DeviceMem);

__device__ float rand_MWC_oc(unsigned long long*,unsigned int*);
__device__ float rand_MWC_co(unsigned long long*,unsigned int*);

__device__ void LaunchPhoton_Default(PhotonStruct*,unsigned long long*, unsigned int*);
__device__ void LaunchPhoton_Fluo(PhotonStruct* p, int photonNumber, unsigned long long* x, unsigned int* a ,MemStruct DeviceMem);
__global__ void LaunchPhoton_Global(MemStruct);

__device__ void Spin(PhotonStruct*, float,unsigned long long*,unsigned int*);
__device__ void Fluorescence(PhotonStruct* p, LayerStructDevice layer, float stepLength, unsigned long long* x, unsigned int* a, MemStruct* DeviceMem);
__device__ unsigned int Reflect(PhotonStruct*, int, unsigned long long*, unsigned int*);
__device__ unsigned int PhotonSurvive(PhotonStruct*, unsigned long long*, unsigned int*);
__device__ void AtomicAddULL(unsigned long long* address, unsigned int add);





// wrapper for device code
void DoOneSimulation(SimulationStruct* simulation, unsigned long long* x,unsigned int* a,
					 string FolderPath,string OutputFileName, vector<SimulationStruct>* allSimulations)
{
	
	MemStruct HostMem;
	MemStruct DeviceMem;
	
	unsigned int threads_active_total=1;
	unsigned int i=0,ii=0;
		
    cudaError_t cudastat;
    clock_t time1,time2;

	// Start the clock
    time1=clock();

	// x and a are already initialised in memory
	HostMem.x=x;
	HostMem.a=a;

	InitMemStructs(&HostMem,&DeviceMem, simulation);
	InitDCMem(simulation);	

    dim3 dimBlock(NUM_THREADS_PER_BLOCK);
    dim3 dimGrid(NUM_BLOCKS);
	
//----------
	LaunchPhoton_Global<<<dimGrid,dimBlock>>>(DeviceMem);
//----------	
	CUDA_SAFE_CALL( cudaThreadSynchronize() ); // Wait for all threads to finish
	//cudaThreadSynchronize();
	cudastat=cudaGetLastError(); // Check if there was an error
	if(cudastat)printf("Error code=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));

	while(threads_active_total>0)
	{
		i++;		
		MCd<<<dimGrid,dimBlock>>>(DeviceMem);
		CUDA_SAFE_CALL (cudaThreadSynchronize());
		cudastat=cudaGetLastError(); // Check if there was an error
		if(cudastat)
			printf("Error avec cudastat -> code=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));

		// Copy thread_active from device to host
		CUDA_SAFE_CALL (cudaMemcpy(HostMem.thread_active,DeviceMem.thread_active,NUM_THREADS*sizeof(unsigned int),cudaMemcpyDeviceToHost));
		//cudaMemcpy(HostMem.thread_active,DeviceMem.thread_active,NUM_THREADS*sizeof(unsigned int),cudaMemcpyDeviceToHost);
		threads_active_total = 0;
		for(ii=0;ii<NUM_THREADS;ii++) threads_active_total+=HostMem.thread_active[ii];
		cudaMemcpy(HostMem.num_terminated_photons,DeviceMem.num_terminated_photons,sizeof(unsigned int),cudaMemcpyDeviceToHost);
		printf("Run %u, Number of photons launched %u, Threads active %u\n",i,*HostMem.num_terminated_photons,threads_active_total);
	}
	printf("Simulation done!\n");
	CopyDeviceToHostMem(&HostMem, &DeviceMem, simulation);
	UpdateSimulation(allSimulations, &HostMem);
    time2=clock();
	printf("Simulation time: %.2f sec\n",(double)(time2-time1)/CLOCKS_PER_SEC);
	WriteHeader(FolderPath,simulation->outp_filename,(time2-time1));
	Write_Simulation_Results(&HostMem, simulation,FolderPath,simulation->outp_filename);


	FreeMemStructs(&HostMem,&DeviceMem);
}
//-------------------------------------------------------------------------------------

__global__ void MCd(MemStruct DeviceMem)
{
    //Block index
    int bx=blockIdx.x;

    //Thread index
    int tx=threadIdx.x;	


    //First element processed by the block
    int begin=NUM_THREADS_PER_BLOCK*bx;
	int i;
	

    
	unsigned long long int x=DeviceMem.x[begin+tx];//coherent
	unsigned int a=DeviceMem.a[begin+tx];//coherent
	float s;	//lambdaStep length
	unsigned int index;
	unsigned int w_temp;
	int new_layer;
	
	PhotonStruct p = DeviceMem.p[begin+tx];	

//First, make sure the thread (photon) is active
	unsigned int ii = 0;
	if(!DeviceMem.thread_active[begin+tx]) 
		ii = NUMSTEPS_GPU;
    for(;ii<NUMSTEPS_GPU;ii++) //this is the main for loop
	{
		//-----------------------------------------------------------------
		if(!PhotonSurvive(&p,&x,&a)) // Check if photons survives or not
		{	
			int	old = atomicAdd(DeviceMem.num_terminated_photons,1u);
			if (old < *photonToLaunchFluo){
				if(old < *photonToLaunch)
				{
					LaunchPhoton_Default(&p,&x,&a);	// initalisation		
				}
				else
				{
					LaunchPhoton_Fluo(&p,old - *photonToLaunch,&x,&a,DeviceMem);
				}				
			}		
			else {
				DeviceMem.thread_active[begin+tx] = 0u;// Set thread to inactive
				atomicSub(DeviceMem.num_terminated_photons,1u); // undo the atomicAdd since the photon was not launched
				break;
			}
		}		
		if(layers_dc[p.layer].mutr!=FLT_MAX)
			s = -__logf(rand_MWC_oc(&x,&a))*layers_dc[p.layer].mutr;//sample lambdaStep length [cm] //HERE AN OPEN_OPEN FUNCTION WOULD BE APPRECIATED
		else
			s = 100.0f;//temporary, say the lambdaStep in glass is 100 cm.
		
		//Check for layer transitions and in case, calculate s
		new_layer = p.layer;
		if(p.z+s*p.dz<layers_dc[p.layer].z_min){
			new_layer--; 
			s = __fdividef(layers_dc[p.layer].z_min-p.z,p.dz);
		} //Check for upwards reflection/transmission & calculate new s
		if(p.z+s*p.dz>layers_dc[p.layer].z_max){
			new_layer++; 
			s = __fdividef(layers_dc[p.layer].z_max-p.z,p.dz);
		} //Check for downward reflection/transmission

		p.x += p.dx*s;
		p.y += p.dy*s;
		p.z += p.dz*s;

		if(p.z>layers_dc[p.layer].z_max) p.z=layers_dc[p.layer].z_max;//needed?
		if(p.z<layers_dc[p.layer].z_min) p.z=layers_dc[p.layer].z_min;//needed?

		if(new_layer!=p.layer)
		{
			// set the remaining lambdaStep length to 0
			s = 0.0f;  
 
			if(Reflect(&p,new_layer,&x,&a)==0u)//Check for reflection
			{ // Photon is transmitted
				if(new_layer == 0)
				{ //Diffuse reflectance
					index = __float2int_rz(acosf(-p.dz)*2.0f*RPI*det_dc[0].na)*det_dc[0].nr+min(__float2int_rz(__fdividef(sqrtf(p.x*p.x+p.y*p.y),det_dc[0].dr)),(int)det_dc[0].nr-1);
					AtomicAddULL(&(DeviceMem.Rd_ra[index]), p.weight);
					p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
				}
				if(new_layer > *n_layers_dc)
				{	//Transmitted
					index = __float2int_rz(acosf(p.dz)*2.0f*RPI*det_dc[0].na)*det_dc[0].nr+min(__float2int_rz(__fdividef(sqrtf(p.x*p.x+p.y*p.y),det_dc[0].dr)),(int)det_dc[0].nr-1);
					AtomicAddULL(&(DeviceMem.Tt_ra[index]), p.weight);
					p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
				}
            }		
		} //w=0;

		if(s > 0.0f)
		{
			// Drop weight (apparently only when the photon is scattered)
			w_temp = __float2uint_rn(layers_dc[p.layer].mua*layers_dc[p.layer].mutr*__uint2float_rn(p.weight));
			p.weight -= w_temp;
			// Process Fluorescence
			////sn=0;
			//if (layers_dc[p.layer].fluoNb)
			//	Fluorescence (&p, (layers_dc[p.layer]), s, &x, &a, &DeviceMem);//layers_dc[p.layer].fluoNb);
			for (i=0;i<layers_dc[p.layer].fluoNb;++i) {
				if (rand_MWC_co(&x,&a) < s * layers_dc[p.layer].fluo[i].mua) { // photon absorbed
					if (rand_MWC_co(&x,&a) < layers_dc[p.layer].fluo[i].quantumYield) { // new photon emitted
						int j = 0;
						float rand = rand_MWC_co(&x,&a);
						while (layers_dc[p.layer].fluo[i].emissionDistributionFunction[j] < rand) {
							j++;
						}
						int old = atomicAdd(DeviceMem.num_emitted_fluo_photons,1u);
						if (old<MAX_FLUO_EMITTED) {
							DeviceMem.fluoROut[old] = sqrtf(p.x*p.x+p.y*p.y);
							DeviceMem.fluoZOut[old] = (p.z);
							DeviceMem.fluoWeightOut[old] = p.weight;
							DeviceMem.fluoWavelengthOut[old] = j;
						}		
					}
				p.weight = 0; //kill the photon
				break; // dont test other fluorophores
				}
			}
			Spin(&p,layers_dc[p.layer].g,&x,&a);
			float musp = layers_dc[p.layer].mus * (1-layers_dc[p.layer].g);
			if (p.z>det_dc->Zc && p.dz>0) { // make one last step in random direction
				s = 1/musp;
				p.x += p.dx*s;
				p.y += p.dy*s;
				p.z += p.dz*s;
				p.weight*=musp/(musp+layers_dc[p.layer].mua);
				index = (min(__float2int_rz(__fdividef(p.z,det_dc[0].dz)),(int)det_dc[0].nz-1)*det_dc[0].nr+min(__float2int_rz(__fdividef(sqrtf(p.x*p.x+p.y*p.y),det_dc[0].dr)),(int)det_dc[0].nr-1) );
				AtomicAddULL(&DeviceMem.A_rz[index], p.weight);
				p.weight=0;
			}
		}	
	} // main for loop
	
	__syncthreads();//necessary?

	//save the state of the MC simulation in global memory before exiting
	DeviceMem.p[begin+tx] = p;	//This one is incoherent!!!
	DeviceMem.x[begin+tx] = x; //this one also seems to be coherent
}


//end MCd
//-------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------
inline __device__ void LaunchPhoton_Fluo(PhotonStruct* p, int photonNumber, unsigned long long* x, unsigned int* a ,MemStruct DeviceMem)	
{
	// We are currently not using the RNG but might do later
	//float input_fibre_radius = 0.03;//[cm]
	//p->x=input_fibre_radius*sqrtf(rand_MWC_co(x,a));

	p->x = (unsigned int)(DeviceMem.fluoRIn[photonNumber]);
	p->y  = 0.0f;
	p->z  = (unsigned int)(DeviceMem.fluoZIn[photonNumber]);
	p->dx = 0.0f;
	p->dy = 0.0f;
	p->dz = 1.0f;
	p->layer = 1;
	while (p->z > layers_dc[p->layer].z_max && p->layer < *n_layers_dc) {
		(p->layer)++;
	}
	p->weight = (unsigned int)(DeviceMem.fluoWeightIn[photonNumber]);
	Spin(p,0,x,a);
}
// Fin LaunchPhoton_FLUO ------------------------------------------------------------------------------------- 


//------------------------------------------------------------------------------------------------------------
__device__ void LaunchPhoton_Default(PhotonStruct* p, unsigned long long* x, unsigned int* a)	// Rajouter à l'appel , le poids du photon select
{
	// We are currently not using the RNG but might do later
	//float input_fibre_radius = 0.03;//[cm]
	//p->x=input_fibre_radius*sqrtf(rand_MWC_co(x,a));

	p->x  = 0.0f;
	p->y  = 0.0f;
	p->z  = 0.0f;
	p->dx = 0.0f;
	p->dy = 0.0f;
	p->dz = 1.0f;
	p->layer = 1;
	p->weight = (*start_weight_dc);//specular reflection!
	
}
// Fin LaunchPhoton Default -------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
// Initialise tous les photons en créant les structures pour chq photons
__global__ void LaunchPhoton_Global(MemStruct DeviceMem)//PhotonStruct* pd, unsigned long long* x, unsigned int* a)
{
	int bx=blockIdx.x;
    int tx=threadIdx.x;	

    //First element processed by the block
    int begin=NUM_THREADS_PER_BLOCK*bx;

	PhotonStruct p;
//	SimulationStruct simu;
	unsigned long long int x=(DeviceMem.x)[begin+tx];//coherent
	unsigned int a=(DeviceMem.a)[begin+tx];//coherent
		
	atomicAdd(DeviceMem.num_terminated_photons,1u);
	LaunchPhoton_Default(&p,&x,&a);

	//__syncthreads();//necessary?
	DeviceMem.p[begin+tx]=p;//incoherent!?
}
// LaunchPhoton_Global -------------------------------------------------------------------------------------


__device__ void Fluorescence(PhotonStruct* p, LayerStructDevice layer, float stepLength, unsigned long long* x,
							 unsigned int* a, MemStruct* DeviceMem){
	int i;
	for (i=0;i<layer.fluoNb;++i) {
		if (rand_MWC_co(x,a) < stepLength * layer.fluo[i].mua) { // photon absorbed
			if (rand_MWC_co(x,a) < layer.fluo[i].quantumYield) { // new photon emitted
				int j = 0;
				float rand = rand_MWC_co(x,a);
				while (layer.fluo[i].emissionDistributionFunction[j] < rand) {
					j++;
				}
				int old = atomicAdd(DeviceMem->num_emitted_fluo_photons,1u);
				if (old<MAX_FLUO_EMITTED) {
					DeviceMem->fluoROut[old] = sqrtf(p->x*p->x+p->y*p->y);
					DeviceMem->fluoZOut[old] = (p->z);
					DeviceMem->fluoWeightOut[old] = p->weight;
					DeviceMem->fluoWavelengthOut[old] = j;
				}		
			}
		p->weight = 0; //kill the photon
		break; // dont test other fluorophores
		}
	}
}

//-------------------------------------------------------------------------------------
__device__ void Spin(PhotonStruct* p, float g, unsigned long long* x, unsigned int* a)
{
	float cost, sint;	// cosine and sine of the 
						// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
						// azimuthal angle psi. 
	float temp;

	float tempdir=p->dx;

	//This is more efficient for g!=0 but of course less efficient for g==0
	temp = __fdividef((1.0f-(g)*(g)),(1.0f-(g)+2.0f*(g)*rand_MWC_co(x,a)));//Should be close close????!!!!!
	cost = __fdividef((1.0f+(g)*(g) - temp*temp),(2.0f*(g)));
	if(g==0.0f)
		cost = 2.0f*rand_MWC_co(x,a) -1.0f;//Should be close close??!!!!!

	sint = sqrtf(1.0f - cost*cost);

	__sincosf(2.0f*PI*rand_MWC_co(x,a),&sinp,&cosp);// spin psi [0-2*PI)
	
	temp = sqrtf(1.0f - p->dz*p->dz);

	if(temp==0.0f) //normal incident.
	{
		p->dx = sint*cosp;
		p->dy = sint*sinp;
		p->dz = copysignf(cost,p->dz*cost);
	}
	else // regular incident.
	{
		p->dx = __fdividef(sint*(p->dx*p->dz*cosp - p->dy*sinp),temp) + p->dx*cost;
		p->dy = __fdividef(sint*(p->dy*p->dz*cosp + tempdir*sinp),temp) + p->dy*cost;
		p->dz = -sint*cosp*temp + p->dz*cost;
	}

	//normalisation seems to be required as we are using floats! Otherwise the small numerical error will accumulate
	temp=rsqrtf(p->dx*p->dx+p->dy*p->dy+p->dz*p->dz);
	p->dx = p->dx*temp;
	p->dy = p->dy*temp;
	p->dz = p->dz*temp;
}
// Spin  -------------------------------------------------------------------------------------

			
//-------------------------------------------------------------------------------------
__device__ unsigned int Reflect(PhotonStruct* p, int new_layer, unsigned long long* x, unsigned int* a)
{
	//Calculates whether the photon is reflected (returns 1) or not (returns 0)
	// Reflect() will also update the current photon layer (after transmission) and photon direction (both transmission and reflection)


	float n1 = layers_dc[p->layer].n;
	float n2 = layers_dc[new_layer].n;
	float r;
	float cos_angle_i = fabsf(p->dz);

	if(n1==n2)//refraction index matching automatic transmission and no direction change
	{	
		p->layer = new_layer;
		return 0u;
	}

	if(n1>n2 && n2*n2<n1*n1*(1-cos_angle_i*cos_angle_i))//total internal reflection, no layer change but z-direction mirroring
	{
		p->dz *= -1.0f;
		return 1u; 
	}

	if(cos_angle_i==1.0f)//normal incident
	{		
		r = __fdividef((n1-n2),(n1+n2));
		if(rand_MWC_co(x,a)<=r*r)
		{
			//reflection, no layer change but z-direction mirroring
			p->dz *= -1.0f;
			return 1u;
		}
		else
		{	//transmission, no direction change but layer change
			p->layer = new_layer;
			return 0u;
		}
	}
	
	//gives almost exactly the same results as the old MCML way of doing the calculation but does it slightly faster
	// save a few multiplications, calculate cos_angle_i^2;
	float e = __fdividef(n1*n1,n2*n2)*(1.0f-cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
	r=2*sqrtf((1.0f-cos_angle_i*cos_angle_i)*(1.0f-e)*e*cos_angle_i*cos_angle_i);//use r as a temporary variable
	e=e+(cos_angle_i*cos_angle_i)*(1.0f-2.0f*e);//Update the value of e
	r = e*__fdividef((1.0f-e-r),((1.0f-e+r)*(e+r)));//Calculate r	

	if(rand_MWC_co(x,a)<=r)
	{ 
		// Reflection, mirror z-direction!
		p->dz *= -1.0f;
		return 1u;
	}
	else
	{	
		// Transmission, update layer and direction
		r = __fdividef(n1,n2);
		e = r*r*(1.0f-cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
		p->dx *= r;
		p->dy *= r;
		p->dz = copysignf(sqrtf(1-e) ,p->dz);
		p->layer = new_layer;
		return 0u;
	}

}
//  Reflect -------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
//Roulette Russe !
__device__ unsigned int PhotonSurvive(PhotonStruct* p, unsigned long long* x, unsigned int* a)
{	//Calculate wether the photon survives (returns 1) or dies (returns 0)

	if(p->weight>WEIGHTI) return 1u; // No roulette needed
	if(p->weight==0u) return 0u;	// Photon has exited slab, i.e. kill the photon

	if(rand_MWC_co(x,a)<CHANCE)
	{
		p->weight = __float2uint_rn(__fdividef((float)p->weight,CHANCE));
		return 1u;
	}

	//else
	return 0u;
}
//-------------------------------------------------------------------------------------




//-------------------------------------------------------------------------------------
//Device function to add an unsigned integer to an unsigned long long using CUDA Compute Capability 1.1
__device__ void AtomicAddULL(unsigned long long* address, unsigned int add)
{
	if(atomicAdd((unsigned int*)address,add)+add<add)
		atomicAdd(((unsigned int*)address)+1,1u);
}


//RANDOM 

__device__ float rand_MWC_co(unsigned long long* x,unsigned int* a)
{
		//Generate a random number [0,1)
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return __fdividef(__uint2float_rz((unsigned int)(*x)),(float)0x100000000);// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)

}//end __device__ rand_MWC_co

__device__ float rand_MWC_oc(unsigned long long* x,unsigned int* a)
{
		//Generate a random number (0,1]
		return 1.0f-rand_MWC_co(x,a);
}//end __device__ rand_MWC_oc


// MEM.CU

int Taille=800;	// Taille du Tableau TabTout

size_t size;


//float* TabPoidsPond;		// Tableau avec les poids pond qui sera alloué en golbal



int CopyDeviceToHostMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{ //Copy data from Device to Host memory

	int rz_size = sim->det.nr*sim->det.nz;
	int ra_size = sim->det.nr*sim->det.na;

	// Copy A_rz, Rd_ra and Tt_ra
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->A_rz,DeviceMem->A_rz,rz_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->Rd_ra,DeviceMem->Rd_ra,ra_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->Tt_ra,DeviceMem->Tt_ra,ra_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	//Also copy the state of the RNG's
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->x,DeviceMem->x,NUM_THREADS*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	//Copy the fluorescence photons that have been emitted
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->num_emitted_fluo_photons, DeviceMem->num_emitted_fluo_photons, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->fluoROut, DeviceMem->fluoROut, MAX_FLUO_EMITTED * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->fluoZOut, DeviceMem->fluoZOut, MAX_FLUO_EMITTED * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->fluoWeightOut, DeviceMem->fluoWeightOut, MAX_FLUO_EMITTED * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL( cudaMemcpy(HostMem->fluoWavelengthOut, DeviceMem->fluoWavelengthOut, MAX_FLUO_EMITTED * sizeof(int), cudaMemcpyDeviceToHost));

	return 0;
}


int InitDCMem(SimulationStruct* sim)
{
	// Copy det-data to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(det_dc,&(sim->det),sizeof(DetStruct)) );
	
	// Copy n_layers_dc to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(n_layers_dc,&(sim->n_layers),sizeof(unsigned int)));

	// Copy start_weight_dc to constant device memory
	//( sim->start_weight ) = (sim->start_weight)* 
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(start_weight_dc,&(sim->start_weight),sizeof(unsigned int)));

	// Copy layer data to constant device memory
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(layers_dc,sim->layers,(sim->n_layers+2)*sizeof(LayerStructDevice)) );

	// Copy num_photons_dc to constant device memory
//	CUDA_SAFE_CALL( cudaMemcpyToSymbol(num_photons_dc,&(sim->number_of_photons),sizeof(unsigned int)));


	// INIT photonToLaunch and photonToLaunchFluo
	int N_fluo = sim->number_of_photons + sim->fluoR.size();
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(photonToLaunch,&(sim->number_of_photons),sizeof(unsigned int)));
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(photonToLaunchFluo,&(N_fluo),sizeof(unsigned int)));
    
	cudaError_t cudastat=cudaGetLastError(); 
	return 0;
	
}

int InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{
	int rz_size,ra_size;
	size = Taille * sizeof(float);

	rz_size = sim->det.nr*sim->det.nz;
	ra_size = sim->det.nr*sim->det.na;

	// Allocate p on the device!!
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->p,NUM_THREADS*sizeof(PhotonStruct)) );

		
	// Allocate A_rz on host and device
	HostMem->A_rz = (unsigned long long*) malloc(rz_size*sizeof(unsigned long long));
	if(HostMem->A_rz==NULL){printf("Error allocating HostMem->A_rz"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->A_rz,rz_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->A_rz,0,rz_size*sizeof(unsigned long long)) );

	// Allocate Rd_ra on host and device
	HostMem->Rd_ra = (unsigned long long*) malloc(ra_size*sizeof(unsigned long long));
	if(HostMem->Rd_ra==NULL){printf("Error allocating HostMem->Rd_ra"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->Rd_ra,ra_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->Rd_ra,0,ra_size*sizeof(unsigned long long)) );

	//------------------------------
	// 
	HostMem->Tt_ra = (unsigned long long*) malloc(ra_size*sizeof(unsigned long long));
	if(HostMem->Tt_ra==NULL){printf("Error allocating HostMem->Tt_ra");exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->Tt_ra,ra_size*sizeof(unsigned long long)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->Tt_ra,0,ra_size*sizeof(unsigned long long)) );


	// Allocate x and a on the device (For MWC RNG)
    CUDA_SAFE_CALL(cudaMalloc((void**)&DeviceMem->x,NUM_THREADS*sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->x,HostMem->x,NUM_THREADS*sizeof(unsigned long long),cudaMemcpyHostToDevice));
	
    CUDA_SAFE_CALL(cudaMalloc((void**)&DeviceMem->a,NUM_THREADS*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->a,HostMem->a,NUM_THREADS*sizeof(unsigned int),cudaMemcpyHostToDevice));


	// Allocate thread_active on the device and host
	HostMem->thread_active = (unsigned int*) malloc(NUM_THREADS*sizeof(unsigned int));
	if(HostMem->thread_active==NULL){printf("Error allocating HostMem->thread_active"); exit (1);}
	for(int i=0;i<NUM_THREADS;i++)HostMem->thread_active[i]=1u;

	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->thread_active,NUM_THREADS*sizeof(unsigned int)) );
	CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->thread_active,HostMem->thread_active,NUM_THREADS*sizeof(unsigned int),cudaMemcpyHostToDevice));

	//Allocate num_terminated_photons on the device and host
	HostMem->num_terminated_photons = (unsigned int*) malloc(sizeof(unsigned int));
	if(HostMem->num_terminated_photons==NULL){printf("Error allocating HostMem->num_terminated_photons"); exit (1);}
	*HostMem->num_terminated_photons=0;

	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->num_terminated_photons,sizeof(unsigned int)) );
	CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->num_terminated_photons,HostMem->num_terminated_photons,sizeof(unsigned int),cudaMemcpyHostToDevice));
		
	// Allocate fluo outputs array on the device and host
	HostMem->num_emitted_fluo_photons = (int*) malloc(sizeof(int));
	if(HostMem->num_emitted_fluo_photons==NULL){printf("Error allocating HostMem->num_emitted_photons"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->num_emitted_fluo_photons,sizeof(int)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->num_emitted_fluo_photons,0,sizeof(int)) );

	HostMem->fluoROut = (float*) malloc(MAX_FLUO_EMITTED * sizeof(float));
	if(HostMem->fluoROut==NULL){printf("Error allocating HostMem->fluoROut"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoROut,MAX_FLUO_EMITTED * sizeof(float)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->fluoROut,0,MAX_FLUO_EMITTED * sizeof(float)) );

	HostMem->fluoZOut = (float*) malloc(MAX_FLUO_EMITTED * sizeof(float));
	if(HostMem->fluoZOut==NULL){printf("Error allocating HostMem->fluoZOut"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoZOut,MAX_FLUO_EMITTED * sizeof(float)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->fluoZOut,0,MAX_FLUO_EMITTED * sizeof(float)) );

	HostMem->fluoWeightOut = (float*) malloc(MAX_FLUO_EMITTED * sizeof(float));
	if(HostMem->fluoWeightOut==NULL){printf("Error allocating HostMem->fluoWeightOut"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoWeightOut,MAX_FLUO_EMITTED * sizeof(float)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->fluoWeightOut,0,MAX_FLUO_EMITTED * sizeof(float)) );

	HostMem->fluoWavelengthOut = (int*) malloc(MAX_FLUO_EMITTED * sizeof(int));
	if(HostMem->fluoWavelengthOut==NULL){printf("Error allocating HostMem->fluoWavelengthOut"); exit (1);}
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoWavelengthOut,MAX_FLUO_EMITTED * sizeof(int)) );
	CUDA_SAFE_CALL( cudaMemset(DeviceMem->fluoWavelengthOut,0,MAX_FLUO_EMITTED * sizeof(int)) );
	
	// Allocate fluo input arrays on the device and host
	HostMem->fluoWeightIn = (float*) malloc((*sim).fluoWeight.size() * sizeof(float));
	if(HostMem->fluoWeightIn==NULL){printf("Error allocatinf HostMem->fluoWeightIn");exit (1);}
	for (int i=0;i<(int)(*sim).fluoWeight.size();i++)
		HostMem->fluoWeightIn[i] = (*sim).fluoWeight[i];
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoWeightIn, (*sim).fluoWeight.size() * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->fluoWeightIn,HostMem->fluoWeightIn,
		(*sim).fluoWeight.size()*sizeof(float),cudaMemcpyHostToDevice));
	
	HostMem->fluoRIn = (float*) malloc((*sim).fluoR.size() * sizeof(float));
	if(HostMem->fluoRIn==NULL){printf("Error allocatinf HostMem->fluoRIn");exit (1);}
	for (int i=0;i<(int)(*sim).fluoR.size();i++)
		HostMem->fluoRIn[i] = (*sim).fluoR[i];
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoRIn, (*sim).fluoR.size() * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->fluoRIn,HostMem->fluoRIn, 
		(*sim).fluoR.size()*sizeof(float),cudaMemcpyHostToDevice));
	
	HostMem->fluoZIn = (float*) malloc((*sim).fluoZ.size() * sizeof(float));
	if(HostMem->fluoZIn==NULL){printf("Error allocatinf HostMem->fluoZIn");exit (1);}
	for (int i=0;i<(int)(*sim).fluoZ.size();i++)
		HostMem->fluoZIn[i] = (*sim).fluoZ[i];
	CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->fluoZIn, (*sim).fluoZ.size() * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(DeviceMem->fluoZIn,HostMem->fluoZIn, 
		(*sim).fluoZ.size()*sizeof(float),cudaMemcpyHostToDevice));

	return 1;
}

void FreeMemStructs(MemStruct* HostMem, MemStruct* DeviceMem)
{
	free(HostMem->A_rz);
	free(HostMem->Rd_ra);
	free(HostMem->Tt_ra);
	free(HostMem->thread_active);
	free(HostMem->num_terminated_photons);
	
	cudaFree(DeviceMem->A_rz);
	cudaFree(DeviceMem->Rd_ra);
	cudaFree(DeviceMem->Tt_ra);
    cudaFree(DeviceMem->x);
    cudaFree(DeviceMem->a);
	cudaFree(DeviceMem->thread_active);
	cudaFree(DeviceMem->num_terminated_photons);


}
