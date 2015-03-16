#include "Simul.hpp"

#include <boost\filesystem.hpp>
#include <boost\filesystem\fstream.hpp>

using namespace std;
using namespace boost;


 Simulation::Simulation(string s){
  //int N = 3;
  //float r = 0.01; // diametre : 200µm
  //fibers = vector<Fiber>(N);
  //fibers[0].position = (0.0262+0.0278)/2;
  //fibers[0].radius = r;
  //fibers[1].position = (0.0515+0.0541)/2;
  //fibers[1].radius = r;
  //fibers[2].position = (0.0758+0.0778)/2;
  //fibers[2].radius = r;/*
  //fibers[3].position = 0.0541;
  //fibers[3].radius = r;
  //fibers[4].position = 0.0758;
  //fibers[4].radius = r;
  //fibers[5].position = 0.0778;
  //fibers[5].radius = r;*/
  //nFiber = vector<float>(N_WAVELENGTH,1.45);
  filesystem::path p(s);
  filesystem::ifstream in(p);
  char buf[1000];
  in.getline(buf,1000);
  in >> nPhotons;
  size_t nLayers;
  in.getline(buf,1000);
  in.getline(buf,1000);
  in>> nLayers;
  in.getline(buf,1000);
  in.getline(buf,1000);
  excitationSpectrum.reserve(N_WAVELENGTH);
  nAbove.reserve(N_WAVELENGTH);
  nBelow.reserve(N_WAVELENGTH);
  nFiber.reserve(N_WAVELENGTH);
  for (size_t i = 0; i<nLayers; ++i){
   Layer l;
   in>>l.thickness;
   size_t nFluo;
   in >> nFluo;
   l.g.reserve(N_WAVELENGTH);
   l.mua.reserve(N_WAVELENGTH);
   l.mus.reserve(N_WAVELENGTH);
   l.n.reserve(N_WAVELENGTH);
   for (size_t j=0;j<nFluo;++j){
    FluoDesc f;
    in >> f.quantumYield;
    f.emissionSpectrum.reserve(N_WAVELENGTH);
    f.mua.reserve(N_WAVELENGTH);
    l.fluo.push_back(f);
   }
   layers.push_back(l);
  }
  in.getline(buf,1000);
  in.getline(buf,1000);
  size_t Nf;
  in>>Nf;
  for (size_t i=0;i<Nf;++i){
   float p,r;
   in>>p>>r;
   Fiber f;
   f.position=p;
   f.radius = r;
   fibers.push_back(f);
  }
  in.getline(buf,1000);
  in.getline(buf,1000);
  for (size_t i=0;i<N_WAVELENGTH;++i){
   int dummy;
   float e,na,nb,nf;
   in >> dummy >> e >> na >> nb >> nf;
   excitationSpectrum.push_back(i==0?e:0);//WARNING CHANGE THIS TEST !!!
   nAbove.push_back(na);
   nBelow.push_back(nb);
   nFiber.push_back(nf);  // THIS TOO §§§
   for (size_t j=0;j<layers.size();++j){
    float n,mua,mus,g;
    in >> n >> mua >> mus >> g;
    layers[j].n.push_back(n);
    layers[j].mua.push_back(mua);
    layers[j].mus.push_back(mus);
    layers[j].g.push_back(g);
    for (size_t k=0;k<layers[j].fluo.size();++k){
     float muaf,emf;
     in>>muaf>>emf;
     layers[j].fluo[k].mua.push_back(muaf);
     layers[j].fluo[k].emissionSpectrum.push_back(emf);
    }
   }
  }
 }