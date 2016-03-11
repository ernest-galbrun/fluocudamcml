#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <device_launch_parameters.h>
#include "Simul.hpp"
#include <algorithm>
#include <iostream>

using namespace std;

texture<float4, cudaTextureType2D> tex;


 Simulation::~Simulation(){
 cudaFree(data);
 cudaFree(fluo_data);
 cudaFree(photonsToLaunch);
 cudaFree(photonsLaunched);
 cudaFree(fibersDesc);
 cudaFree(zInterface);
 cudaFree(cu_nFiber);
 cudaFree(result_fiber);
 cudaFree(result_abs);
 cudaFree(result_dif);	 
 cudaFree(result_tr);
 

 }

__device__ unsigned int isFiber(float r, curandState_t *state, unsigned int NFiber, float2* fibersDesc, float* portion) {
 //returns 0 if not a fiber, 2 + index of the fiber otherwise
	for (unsigned int i=0;i<NFiber;++i){
		float rf = fibersDesc[i].y;
		float d = fibersDesc[i].x;
        float p_cumulative = 0;
		if (r+d<=rf) {
			return 2+i;
			//*portion = 1;
		}
		if (r>d-rf && r<d+rf) {
			float x = (d*d + r*r - rf*rf) / (2*d);
			float y = sqrt(r*r - x*x);
			float p = atan(y/x)/PI / (1-p_cumulative);
            p_cumulative+=p;
			*portion = 1;//p;
			if (curand_uniform(state)<=p){
				return 2+i;
			}
		}
	}
	return 0;
}
 			
//-------------------------------------------------------------------------------------
__device__ unsigned int Reflect(PhotonStruct* p, int new_layer, curandState_t *state, unsigned int nLayers, unsigned int NFiber, float2* fibersDesc, float *cu_nFiber)
{
	// Calculates whether the photon is reflected (returns 1) or not (returns 0) or transmitted to a fiber (returns 2 + index of the fiber)
	// Reflect() will also update the current photon layer (after transmission) and photon direction (both transmission and reflection)
	unsigned int _isFiber = 0;
	float fiber_portion=1;
	if (new_layer == -1){
		float r = sqrt(p->x*p->x + p->y*p->y);
		_isFiber = isFiber(r,state,NFiber,fibersDesc, &fiber_portion);
	}
    float4 from = tex2D(tex,p->lambda,p->layer + 2);
	float4 to = tex2D(tex,p->lambda,new_layer==nLayers?1:(new_layer+2));
	float n1 = from.x;
	float n2 = _isFiber?cu_nFiber[p->lambda]:to.x;
	float r;
	float cos_angle_i = fabsf(p->dz);

	if(n1==n2)//refraction index matching automatic transmission and no direction change
	{	
		p->layer = new_layer;
		p->weight*=fiber_portion;
		return _isFiber;
	}

	if(n1>n2 && n2*n2<n1*n1*(1-cos_angle_i*cos_angle_i))//total internal reflection, no layer change but z-direction mirroring
	{
		p->dz *= -1.0f;
		return 1u; 
	}

	if(cos_angle_i==1.0f)//normal incident
	{		
		r = __fdividef((n1-n2),(n1+n2));
		if(curand_uniform(state)<=r*r)
		{
			//reflection, no layer change but z-direction mirroring
			p->dz *= -1.0f;
			return 1u;
		}
		else
		{	//transmission, no direction change but layer change
			p->layer = new_layer;
			p->weight*=fiber_portion;
            return _isFiber;
		}
	}
	
	// gives almost exactly the same results as the old MCML way of doing the calculation but does it slightly faster
	// save a few multiplications, calculate cos_angle_i^2;
	float e = __fdividef(n1*n1,n2*n2)*(1.0f-cos_angle_i*cos_angle_i); //e is the sin square of the transmission angle
	r=2*sqrtf((1.0f-cos_angle_i*cos_angle_i)*(1.0f-e)*e*cos_angle_i*cos_angle_i);//use r as a temporary variable
	e += (cos_angle_i*cos_angle_i)*(1.0f-2.0f*e);//Update the value of e
	r = e*__fdividef((1.0f-e-r),((1.0f-e+r)*(e+r)));//Calculate r	

	if(curand_uniform(state)<=r)
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
        float NA = 0.22;
        bool go_in_fiber = e < NA*NA/(n1*n1); // check that the incidence angle is inferior to the numerical aperture
		p->dx *= r;
		p->dy *= r;
		p->dz = copysignf(sqrtf(1-e) ,p->dz);
		p->layer = new_layer;
		p->weight*=fiber_portion;
		return (_isFiber&&go_in_fiber)?_isFiber:0u;
	}

};
//  Reflect -------------------------------------------------------------------------------------


 __device__ unsigned int PhotonSurvive(PhotonStruct* p, curandState_t *state)
{	//Calculate wether the photon survives (returns 1) or dies (returns 0)

	if(p->weight>WEIGHTI) return 1u; // No roulette needed
	if(p->weight==0u) return 0u;	// Photon has exited slab, i.e. kill the photon

	if(curand_uniform(state)<CHANCE)
	{
		p->weight = __float2uint_rn(__fdividef((float)p->weight,CHANCE));
		return 1u;
	}

	//else
	return 0u;
};



__device__ void LaunchPhoton(PhotonStruct* p, unsigned int wavelength)	// Rajouter à l'appel , le poids du photon select
{
	// We are currently not using the RNG but might do later
	// float input_fibre_radius = 0.03;//[cm]
	// p->x=input_fibre_radius*sqrtf(rand_MWC_co(x,a));
    float4 data_above = tex2D(tex,wavelength,0);
    float4 data_first = tex2D(tex,wavelength,2);
    float n1=data_above.x;
    float n2=data_first.x;
	float r = (n1-n2)/(n1+n2);
	//printf("\n\nn1=%f\tn2=%f\n\n",n1,n2);
	r = r*r;
	float startWeight = (unsigned int)((float)0xffffffff*(1-r));
	p->x  = 0.0f;
	p->y  = 0.0f;
	p->z  = 0.0f;
	p->dx = 0.0f;
	p->dy = 0.0f;
	p->dz = 1.0f;
	p->layer = 0;
	p->weight = startWeight;//specular reflection!
    p->lambda = wavelength;
	
};

__device__ void Spin(PhotonStruct* p, float g, curandState_t* state)
{
	float cost, sint;	// cosine and sine of the 
						// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
						// azimuthal angle psi. 
	float temp;

	float tempdir=p->dx;

	//This is more efficient for g!=0 but of course less efficient for g==0
	temp = __fdividef((1.0f-(g)*(g)),(1.0f-(g)+2.0f*(g)*curand_uniform(state)));//Should be close close????!!!!!
	cost = __fdividef((1.0f+(g)*(g) - temp*temp),(2.0f*(g)));
	if(g==0.0f)
		cost = 2.0f*curand_uniform(state) -1.0f;//Should be close close??!!!!!

	sint = sqrtf(1.0f - cost*cost);

	__sincosf(2.0f*PI*curand_uniform(state),&sinp,&cosp);// spin psi [0-2*PI)
	
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
};
// Spin  -------------------------------------------------------------------------------------


__global__ void Propagation(unsigned long long seed, int* fluo_count, float2 *fluo_data, float *quantum_yields, unsigned int *photonsToLaunch, unsigned int *photonsLaunched,
                            unsigned int NFiber, float2* fibersDesc, float *cu_nFiber, unsigned long long *result, unsigned int nLayers, float *zInterface,
                            unsigned long long *result_fiber, unsigned long long* hist_out) {
    //float2 s_fibers[fibersSize];
    __shared__ float z_s[MAX_LAYERS+1];
    __shared__ unsigned long long count[8];
    __shared__ unsigned long long hist[8*512];
 
    for (int i=0;i*blockDim.x+threadIdx.x<8 * 512;++i)
        hist[i*blockDim.x+threadIdx.x] = 0;
    for (int i=0;i*blockDim.x+threadIdx.x<10;++i)
        count[i*blockDim.x+threadIdx.x] = 0;
    if (threadIdx.x<(min(nLayers,MAX_LAYERS)+1)){
        z_s[threadIdx.x] = zInterface[threadIdx.x];
    }
    __syncthreads();
    curandState_t state;
    curand_init(seed, blockDim.x * blockIdx.x + threadIdx.x,0,&state);

    PhotonStruct p;
    unsigned stepsCalculated = NUMSTEPS_GPU;
    for (size_t i=0;i<N_WAVELENGTH;++i){
    __syncthreads();
    for (unsigned int pl=photonsLaunched[i];pl<photonsToLaunch[i];pl = atomicInc(&(photonsLaunched[i]),0xffffffff)){
        // printf("old for block %u thread %u : %u\n",blockIdx.x,threadIdx.x,old);
        // Launch a new photon and follow it until it dies
        LaunchPhoton(&p,i);
        unsigned int numSteps = 0;
        // continue;
        while (numSteps < 10000 && PhotonSurvive(&p, &state)){
                ++numSteps;
                // mutr = 1/(mua + mus)
                float4 d = tex2D(tex, p.lambda, p.layer + 2);
                float s; //Step length
                float mutr = 1.f / (d.y + d.z);
                if (d.z != 0.f){
                    s = -__logf(curand_uniform(&state)) * mutr;
                }
                else {// we are in glass => set step length to 100 cm
                    s = 100.f;
                }
                //Check for layer transitions and in case, calculate s
                int new_layer = p.layer;
                if (p.z + s*p.dz < z_s[p.layer]){
                    --new_layer;
                    s = __fdividef(z_s[p.layer] - p.z, p.dz);
                } //Check for upwards reflection/transmission & calculate new s
                if (p.z + s*p.dz > z_s[p.layer + 1]){
                    ++new_layer;
                    s = __fdividef(z_s[p.layer + 1] - p.z, p.dz);
                } //Check for downward reflection/transmission

                p.x += p.dx*s;
                p.y += p.dy*s;
                p.z += p.dz*s;

                if (new_layer != p.layer)
                {
                    // set the remaining lambdaStep length to 0
                    s = 0.0f;
                    unsigned int r = Reflect(&p, new_layer, &state, nLayers, NFiber, fibersDesc, cu_nFiber);
                    if (r == 0u)//Check for reflection
                    { // Photon is transmitted
                        if (new_layer == -1)
                        { //Diffuse reflectance

                            size_t index = min(__float2int_rz(__fdividef(sqrtf(p.x*p.x + p.y*p.y), DR)), NR - 1);
                            atomicAdd(&(result[index]), p.weight);
                            p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
                        }
                        if (new_layer == nLayers)
                        {	//Transmitted
                            size_t index = min(__float2int_rz(__fdividef(sqrtf(p.x*p.x + p.y*p.y), DR/*dr*/)), NR - 1/*nr-1*/);
                            //atomicAdd(&(DeviceMem.Tt_ra[index]), p.weight);
                            p.weight = 0; // Set the remaining weight to 0, effectively killing the photon
                        }
                    }
                    else if (r >= 2u)
                    { //photon went into a fiber             
                        size_t index = min(__float2int_rz(__fdividef(sqrtf(p.x*p.x + p.y*p.y), DR)), NR - 1);
                        atomicAdd(&(result[index*NR + p.lambda]), p.weight);
                        atomicAdd(&(result_fiber[(p.lambda)*NFiber + r - 2]), p.weight/*/(4 * fibersDesc[r-2].x / fibersDesc[r-2].y)*/);
                        unsigned long long index2 = p.weight / (0xffffffff / 512);
                        unsigned long long index_hist = (r - 2) * 512 + index2;
                        ++hist[index_hist];
                        ++count[r - 2];
                        p.weight = 0;
                    }
                } //w=0;
                if (s > 0.0f)
                {
                    // Drop weight (apparently only when the photon is scattered)
                    float4 data = tex2D(tex, p.lambda, p.layer + 2);
                    float mutr = 1 / (data.y + data.z);
                    unsigned int w_temp = __float2uint_rn(data.y* mutr*__uint2float_rn(p.weight));
                    p.weight -= w_temp;
                    // Process Fluorescence
                    //sn=0;
                   // if (fluo_count[p.layer])
                   // 	Fluorescence (&p, (layers_dc[p.layer]), s, &x, &a, &DeviceMem);//layers_dc[p.layer].fluoNb);
                    for (int fluo_id = 0; fluo_id<fluo_count[p.layer]; ++fluo_id) {
                        float2 mua_em = fluo_data[p.lambda + N_WAVELENGTH*(MAX_FLUO*p.layer + fluo_id)];
                        if (curand_uniform(&state) < s * mua_em.x) { // photon absorbed
                            if (curand_uniform(&state) < quantum_yields[MAX_FLUO*p.layer + fluo_id]) { // new photon emitted
                                int j = 0;
                                float rand = curand_uniform(&state);
                                mua_em = fluo_data[j + N_WAVELENGTH*(MAX_FLUO*p.layer + fluo_id)];
                                while (mua_em.y < rand) {
                                    j++;
                                    mua_em = fluo_data[j + N_WAVELENGTH*(MAX_FLUO*p.layer + fluo_id)];
                                }
                                p.lambda = j;
                                data = tex2D(tex, p.lambda, p.layer + 2);
                            }
                            else {
                                p.weight = 0; //kill the photon
                            }
                    	break; // dont test other fluorophores
                    	}
                    }
                    Spin(&p, data.w, &state);
                    //float musp = layers_dc[p.layer].mus * (1-layers_dc[p.layer].g);
                    //if (p.z>det_dc->Zc && p.dz>0) { // make one last step in random direction
                    //	s = 1/musp;
                    //	p.x += p.dx*s;
                    //	p.y += p.dy*s;
                    //	p.z += p.dz*s;
                    //	p.weight*=musp/(musp+layers_dc[p.layer].mua);
                    //	index = (min(__float2int_rz(__fdividef(p.z,det_dc[0].dz)),(int)det_dc[0].nz-1)*det_dc[0].nr+min(__float2int_rz(__fdividef(sqrtf(p.x*p.x+p.y*p.y),det_dc[0].dr)),(int)det_dc[0].nr-1) );
                    //	AtomicAddULL(&DeviceMem.A_rz[index], p.weight);
                    //	p.weight=0;
                    //}
                }
            }
        // printf("threadIdx#%u, numSteps : %u\n",threadIdx.x,numSteps);
        }
    }
 
    __syncthreads();
    if (threadIdx.x == 0){
	    for (int i=0;i<8;++i){
		    for(int j=0;j<512;++j) {
			    atomicAdd(&(hist_out[j+i*512]), hist[j+i*512]);
		    }
	    }
    }
};

void Simulation::LaunchSimulation(ostream &out, ostream &out_fibers){

    PrepareMemory();
    unsigned long long* host_result;
    unsigned long long* host_result_fiber;
    unsigned long long* hist_dev,*hist_host;
    int nf = fibers.size();
    int n_hist_bins = 512;
    cudaMalloc(&hist_dev, sizeof(*hist_dev) * N_WAVELENGTH * n_hist_bins * nf);
    cudaMemset(hist_dev, 0, sizeof(*hist_dev) * N_WAVELENGTH * n_hist_bins * nf);
    hist_host = (unsigned long long*)malloc(sizeof(*hist_host) * N_WAVELENGTH * n_hist_bins * nf);
    host_result = (unsigned long long*) malloc(NR*N_WAVELENGTH*sizeof(unsigned long long));
    host_result_fiber =(unsigned long long*) malloc(fibers.size()*N_WAVELENGTH*sizeof(unsigned long long));
 
    cudaError_t errot;
    errot = cudaMemcpy(host_result_fiber,result_fiber,fibers.size()*N_WAVELENGTH*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    Propagation << <NUM_BLOCKS, NUM_THREADS >> > (seed, fluo_count, fluo_data, quantum_yields, photonsToLaunch, photonsLaunched, fibers.size(), fibersDesc, cu_nFiber,
										    result_dif,layers.size(), zInterface, result_fiber, hist_dev); 
    errot = cudaMemcpy(hist_host, hist_dev, sizeof(*hist_host) * N_WAVELENGTH * n_hist_bins * nf, cudaMemcpyDeviceToHost);
    errot = cudaMemcpy(host_result,result_dif,NR*N_WAVELENGTH*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    errot = cudaMemcpy(host_result_fiber,result_fiber,fibers.size()*N_WAVELENGTH*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    vector<double> mean(nf);
    vector<double> stddev(nf);
    vector<double> var(nf);
    vector<unsigned long long> N(nf);
    double scale1 =  (double)0xFFFFFFFF * nPhotons; //(double)0xFFFFFFFF
    //assume poisson distribution inside each bin
    vector<double> copy_hist(n_hist_bins);
    for (int j = 0; j<n_hist_bins; ++j){
	    for(int i=0;i<nf;++i){
            unsigned long long n = hist_host[i*n_hist_bins + j];
		    N[i]+=n;
		    double x = double(j)-0.5;///512.*double(0xffffffff);
		    mean[i]+=n*x;
		    var[i] += double(n) * x * x;
            copy_hist[j] = n;
           // if (i==0)
                //cout<<n<<'\t';
	    }
	    //cout<<'\n';
    }
    for (int i=0;i<nf;++i){
        cout<<sqrt(var[i]) / double(n_hist_bins) / nPhotons <<'\t';
    }
	cout << endl;
    for (int i = 0; i < N_WAVELENGTH; ++i){
        for (size_t j = 0; j < nf; ++j){
            out_fibers << host_result_fiber[j + i* nf] / scale1 << '\t';
        }
        out_fibers << endl;
    }
    out<<"l\\r";
    for (size_t r=0;r<NR;++r){
        out<<'\t'<<r*DR;
    }
    out<<'\n';
    for (size_t l=0;l<N_WAVELENGTH;++l){
        if (excitationSpectrum[l]!=0){
        out<<l+START_WAVELENGTH;
        for(size_t r=0;r<NR;++r){
            double scale2=scale1*PI*(2*r+1)*DR*DR; //surface of the ring bw r*dr and (r+1)*dr
            out <<'\t'<<(double)host_result[NR*l+r]/scale2;
        }
        out<<endl;
        }
    }
}

void Simulation::PrepareMemory(){

    //compute and store zInterface
    float* z;
    z = (float*)malloc((layers.size()+1)*sizeof(float));
    z[0] = 0.f;
    for (size_t i=0;i<layers.size();++i){
        z[i+1] = z[i] + layers[i].thickness;
    }
    cudaMalloc(&zInterface,(layers.size()+1)*sizeof(float));
    cudaMemcpy(zInterface,z,(layers.size()+1)*sizeof(float),cudaMemcpyHostToDevice);
    //compute the number of photons to emit for each wavelength
    unsigned int NP[N_WAVELENGTH];
    //unsigned int *cuNP;
    float sum = 0;
    for (size_t i=0;i<excitationSpectrum.size();++i){
        sum+=excitationSpectrum[i];
    }
    for (size_t i=0;i<excitationSpectrum.size();++i){
        NP[i] = (nPhotons * excitationSpectrum[i]) / sum;
    }
    cudaMalloc(&photonsToLaunch,sizeof(NP));
    cudaMemcpy(photonsToLaunch,NP,sizeof(NP),cudaMemcpyHostToDevice);
    cudaMalloc(&photonsLaunched,sizeof(NP));
    cudaMemset(photonsLaunched,0,sizeof(NP));
    //create flat data structures for the simulation data
    // and copy it to the device
    float4 *host_data;//, *dev_data;
    size_t count = N_WAVELENGTH*sizeof(float4)*(layers.size()+2);
    host_data = (float4*) malloc(count);
    cudaMalloc(&data,count);  
    cudaMalloc(&fluo_data, N_WAVELENGTH*sizeof(float2) * layers.size()*MAX_FLUO);
    cudaMalloc(&quantum_yields, sizeof(quantum_yields) * layers.size() * MAX_FLUO);
    vector<float> qy(layers.size() * MAX_FLUO);
    float2 *host_fluo;//, *dev_fluo;
    host_fluo = (float2*)malloc(N_WAVELENGTH*sizeof(float2) * layers.size() * MAX_FLUO);
    for (auto i = 0; i < layers.size(); ++i){
        for (auto k = 0; k < min(MAX_FLUO, layers[i].fluo.size()); ++k){
            qy[MAX_FLUO*i + k] = layers[i].fluo[k].quantumYield;
        }
    }
    cudaMemcpy(quantum_yields, &qy[0], layers.size() * MAX_FLUO, cudaMemcpyHostToDevice);
    for (size_t j = 0; j < N_WAVELENGTH; ++j){
        host_data[j] = make_float4(nAbove[j], 0.f, 0.f, 0.f);
        host_data[j + N_WAVELENGTH] = make_float4(nBelow[j], 0.f, 0.f, 0.f);
        for (size_t i = 0; i < layers.size(); ++i){
            host_data[j + N_WAVELENGTH*(i + 2)] = make_float4(layers[i].n[j], layers[i].mua[j], layers[i].mus[j], layers[i].g[j]);           
            for (size_t k = 0; k < min(MAX_FLUO, layers[i].fluo.size()); ++k){
                host_fluo[j + N_WAVELENGTH*(MAX_FLUO*i + k)] = make_float2(layers[i].fluo[k].mua[j], layers[i].fluo[k].emissionSpectrum[j]);
            }
        }
    }
    cudaMemcpy(data,host_data, count, cudaMemcpyHostToDevice);
    cudaMemcpy(fluo_data,host_fluo,N_WAVELENGTH*sizeof(float2) * layers.size() * MAX_FLUO,cudaMemcpyHostToDevice);
    //copy array with number of fluo in each layer
    int* Nfluo = (int*)malloc(sizeof(int)*layers.size());
    for (int i = 0; i < layers.size(); ++i)
        Nfluo[i] = layers[i].fluo.size();
    cudaMalloc(&fluo_count, sizeof(fluo_count)*layers.size());
    cudaMemcpy(fluo_count, &Nfluo[0], layers.size(), cudaMemcpyHostToDevice);

    size_t height = layers.size()+2;
    size_t width = N_WAVELENGTH;
    tex.normalized = false;
    tex.filterMode = cudaFilterModePoint;
    cudaChannelFormatDesc cd = cudaCreateChannelDesc<float4>();
    cudaBindTexture2D(NULL,&tex,data,&cd,width,height,width*sizeof(float4));

    //copy fibers data
    //float2 *host_fiber;
    //host_fiber = (float2*) malloc(fibers.size()*
  
    cudaError_t errot;
    if (fibers.size()) {
        errot = cudaMalloc(&fibersDesc,fibers.size()*sizeof(float2));
        errot = cudaMemcpy(fibersDesc,&(fibers[0]),fibers.size()*sizeof(float2),cudaMemcpyHostToDevice);
    }
    errot = cudaMalloc(&cu_nFiber,N_WAVELENGTH*sizeof(float));
    errot = cudaMemcpy(cu_nFiber,&(nFiber[0]),nFiber.size()*sizeof(float),cudaMemcpyHostToDevice);

    // Prepare the result array (1cm wide with a resolution of 10 µm => size = 1000)
  
    errot = cudaMalloc(&result_dif,NR * N_WAVELENGTH * sizeof(*result_dif));
    errot = cudaMemset(result_dif,0,NR * N_WAVELENGTH * sizeof(*result_dif));
 
    errot = cudaMalloc(&result_fiber, fibers.size() * N_WAVELENGTH* sizeof(*result_fiber));
    errot = cudaMemset(result_fiber, 0, fibers.size() * N_WAVELENGTH* sizeof(*result_fiber));
}
