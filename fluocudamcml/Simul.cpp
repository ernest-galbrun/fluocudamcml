#include "Simul.hpp"

#include <boost\filesystem.hpp>
#include <boost\filesystem\fstream.hpp>

using namespace std;
using namespace boost;


 Simulation::Simulation(string s){
    filesystem::path p(s);
    filesystem::ifstream in(p);
    char buf[1000];
    in.getline(buf,1000);
    in >> nPhotons;
    size_t nLayers;
    in.getline(buf,1000);
    in.getline(buf,1000);
    in >> nLayers;
    size_t nWavelengths;
    in.getline(buf, 1000);
    in.getline(buf, 1000);
    in >> nWavelengths;
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
        if (i >= nWavelengths) {
            //create unused data
            excitationSpectrum.push_back(0);
            nAbove.push_back(0);
            nBelow.push_back(0);
            nFiber.push_back(0); 
            for (size_t j = 0; j<layers.size(); ++j){
                layers[j].n.push_back(0);
                layers[j].mua.push_back(0);
                layers[j].mus.push_back(0);
                layers[j].g.push_back(0);
                for (size_t k = 0; k<layers[j].fluo.size(); ++k){
                    layers[j].fluo[k].mua.push_back(0);
                    layers[j].fluo[k].emissionSpectrum.push_back(0);
                }
            }
            continue;
        }
        int dummy;
        float e,na,nb,nf;
        in >> dummy >> e >> na >> nb >> nf;
        excitationSpectrum.push_back(e);
        nAbove.push_back(na);
        nBelow.push_back(nb);
        nFiber.push_back(nf);
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