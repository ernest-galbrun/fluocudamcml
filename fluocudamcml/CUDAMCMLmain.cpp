//#include "CUDAMCML.h"
#include <boost/program_options.hpp>
#include <boost\filesystem.hpp>
#include "Simul.hpp"


#include <ctime>
#include <iostream>


using namespace std;
using namespace boost;
//
//extern string InputFileName,OutputFileName,FullLaunchingPath,FolderPath,SafeprimesPath;
//extern size_t found;
//
//void	UpdateSimulation(vector<SimulationStruct>* allSimulations, MemStruct* HostMem){
//	for (int i=0;i<*HostMem->num_emitted_fluo_photons;i++) {
//		(*allSimulations)[HostMem->fluoWavelengthOut[i]].fluoR.push_back(HostMem->fluoROut[i]);
//		(*allSimulations)[HostMem->fluoWavelengthOut[i]].fluoZ.push_back(HostMem->fluoZOut[i]);
//		(*allSimulations)[HostMem->fluoWavelengthOut[i]].fluoWeight.push_back(HostMem->fluoWeightOut[i]);
//	}
//}



int main(int argc,char* argv[])
{ 
 string inputFile;
 string output_file;
 string output_file_fibers;
 float Zc;
 unsigned long long seed;
 bool output_volume_data;
 program_options::options_description desc("Allowed options");
 desc.add_options()
		("help", "produce help message")
		("input-file", program_options::value<string>(&inputFile)->default_value(""),"name or path to the simulation description file")
		("output-file", program_options::value<string>(&output_file)->default_value("out"),"name or path of the reflective diffusion data")
		("output-fibers-file", program_options::value<string>(&output_file_fibers)->default_value("out_fibers"),"name or path of the fibers data")
		("Zc", program_options::value<float>(&Zc)->default_value(-1),"Thickness of the critical layer (set to 0 to have the program compute it automatically)")
        ("seed", program_options::value<unsigned long long>(&seed)->default_value(time(NULL)),"Seed for the random number generator (set to time(NULL by default))")
        ("output-volume-data", program_options::value<bool>(&output_volume_data)->default_value(false),"Output volume data in a separate file, one file per wavelength")
		;
 program_options::variables_map vm;
 program_options::store(program_options::command_line_parser(argc, argv).options(desc).run(),vm);
 program_options::notify(vm);
	if (vm.count("help")) {
		cout << desc << '\n';
		return 0;
	}
 //Building the path of the input file
	filesystem::path p(inputFile);
	if (!filesystem::is_regular_file(p)) {
		p = filesystem::current_path()/p;
	}
    if (!filesystem::is_regular_file(p)){
     cout<< "wrong input file\n";
     return -1;
    }
 Simulation s(p.string());
 s.SetSeed(seed);
 ofstream out(output_file);
 ofstream out_fibers(output_file_fibers);
 s.LaunchSimulation(out, out_fibers);
 /*cout << "Press ENTER to continue...";
 cin.ignore( numeric_limits<streamsize>::max(), '\n' );
 */
 return 0;  
}

