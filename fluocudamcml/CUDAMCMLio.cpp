/*	MIEN This file is part of CUDAMCML.

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

#define NFLOATS 5
#define NINTS 5

#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <errno.h>
#include <direct.h>
#include "CUDAMCML.h"


#include <boost\filesystem.hpp>
#include <boost\filesystem\fstream.hpp>


//#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
using namespace std;
using namespace boost;

AllData allData;
vector<string> TabString;
size_t debut=0;
string mot;
int longueur;
ostringstream nIndexFile;
string InputFileName,OutputFileName,FullLaunchingPath,FolderPath,SafeprimesPath;
size_t found;
bool pathNameAbsolute;


// Takes the fluorescence information in *in, computes the emission
// distribution function and puts it in *out

 int GetFluo(FluoStructDevice* out, FluoStructAll*in,int index, int indexCurrent, int n_simulations, int lambdaMin, int lambdaStep){
	out->mua = (in->absorptionSpectrum)[index];
	out->quantumYield = in->quantumYield;
	//compute the integral of the emission spectrum
	float buf[N_WAVELENGTH];
	buf[0] = in->emissionSpectrum[0];
	for (int i=1;i<N_WAVELENGTH;i++){
		buf[i] = buf[i-1]+in->emissionSpectrum[i];
	}
	// Fill the array with the useful data.
	// Make sure that the emitted photon has a wavelength < the current wavelength
	for (int i=0;i<n_simulations -1;i++) {
		if (i > indexCurrent){
			out->emissionDistributionFunction[i] = buf[lambdaMin + i * lambdaStep - 200] / buf[N_WAVELENGTH -1];
		}
		else {
			out->emissionDistributionFunction[i] = 0;
		}
	}
	out->emissionDistributionFunction[n_simulations-1] = 1;
	return 0;
}


//Puts all relevant data from AllData structure to the simulation array structure
 //for the simulation.
 //Returns n_simulations.

int PopulateSimulation(vector<SimulationStruct> *sim, AllData* data,string InputFileName, double Zc) 
{
	int n_simulations = (data->lambdaMax - data->lambdaMin) / data->lambdaStep + 1;
	(*sim).resize(n_simulations);
	for (int i=0;i<(int)(*sim).size();i++)
	{
		int lambda = data->lambdaMin + i * data->lambdaStep;
		// lambda starts at 200 nm.
		int index = lambda - 200;
		(*sim)[i].lambda = lambda;
		(*sim)[i].relativeLightIntensity = data->inputLightIntensity[index];
		(*sim)[i].number_of_photons = data->photonsNb;
		string input_filename=InputFileName;
		nIndexFile << i;
		found=InputFileName.find_last_of(".");
		OutputFileName=input_filename.substr(0,found);
		OutputFileName.insert(OutputFileName.size(),nIndexFile.str());
		OutputFileName.insert(OutputFileName.size(),".mco");	

		string output_filename=OutputFileName;
		strcpy((*sim)[i].inp_filename,input_filename.c_str());
		strcpy((*sim)[i].outp_filename,output_filename.c_str());
		(*sim)[i].det.dr = data->dr;
		(*sim)[i].det.dz = data->dz;
		(*sim)[i].det.na = int( (PI / 2) / (data->da) );
		// the output grid is always 1 cm wide (the z is not used)
		(*sim)[i].det.nr = int( 1./data->dr);
		(*sim)[i].det.nz = int( (data->layers)[data->layersNb].z_max / data->dz );
		(*sim)[i].n_layers = data->layersNb;
		(*sim)[i].layers = (LayerStructDevice*) malloc((data->layersNb+2) * sizeof(LayerStructDevice));
		for (int j=0;j<data->layersNb +2;j++) 
		{
			(*sim)[i].layers[j].z_min = (data->layers)[j].z_min;
			(*sim)[i].layers[j].z_max = (data->layers)[j].z_max;
			(*sim)[i].layers[j].mutr = (data->layers)[j].mutr[index];
			(*sim)[i].layers[j].mus = (data->layers)[j].mus[index];
			(*sim)[i].layers[j].mua = (data->layers)[j].mua[index];
			(*sim)[i].layers[j].g = (data->layers)[j].g[index];
			(*sim)[i].layers[j].n = (data->layers)[j].n[index];
			(*sim)[i].layers[j].fluoNb = (data->layers)[j].fluoNb;
			(*sim)[i].layers[j].thickness = (data->layers)[j].thickness;
			for (int k=0;k<3;k++) {
				int dummy = GetFluo(&((*sim)[i].layers[j].fluo[k]), &((data->layers)[j].fluo[k]), 
					index, i, n_simulations, data->lambdaMin, data->lambdaStep);
			}
		}		
		//calculate start_weight
		double n1=(*sim)[i].layers[0].n;
		double n2=(*sim)[i].layers[1].n;
		double r = (n1-n2)/(n1+n2);
		//printf("\n\nn1=%f\tn2=%f\n\n",n1,n2);
		r = r*r;
		(*sim)[i].start_weight = (unsigned int)((double)0xffffffff*(1-r));
		(*sim)[i].start_weight *= (*sim)[i]. relativeLightIntensity;
		printf("Start weight[%d]=%u\n",i,(*sim)[i].start_weight);
		nIndexFile.str("");
		OutputFileName.clear();
		found=0;
		(*sim)[i].det.Zc = Zc;
		if (Zc==0) {
			(*sim)[i].det.Zc=0;
			(*sim)[i].det.Zc += (*sim)[i].layers[(*sim)[i].n_layers - 1].z_max;
			double mus = (*sim)[i].layers[(*sim)[i].n_layers].mus;
			double g = (*sim)[i].layers[(*sim)[i].n_layers].g;
			(*sim)[i].det.Zc += (1 / (mus * (1-g)));
		}
		else if (Zc<0) {
			(*sim)[i].det.Zc=0;
			(*sim)[i].det.Zc += (*sim)[i].layers[(*sim)[i].n_layers].z_max;
		}
	}
	return n_simulations;
}


int interpret_arg(int argc, char* argv[], unsigned long long* seed, int* ignoreAdetection)
{

	int unknown_argument;
	for(int i=2;i<argc;i++)
	{
		unknown_argument=1;
		if(!strcmp(argv[i],"-A"))
		{
			unknown_argument=0;
			*ignoreAdetection=1;
			printf("Ignoring A-detection!\n");
		}
		if(!strncmp(argv[i],"-S",2) && sscanf(argv[i],"%*2c %llu",seed))
		{
		unknown_argument=0;
		printf("Seed=%llu\n",*seed);
		}
		if(unknown_argument)
		{
			printf("Unknown argument %s!\n",argv[i]);
			//return 1;
		}
	}
	return 0;
}

int Write_Simulation_Results(MemStruct* HostMem, SimulationStruct* sim,string FolderPath,string OutputFileName)
{

	
	// Copy stuff from sim->det to make things more readable:
	double dr=(double)sim->det.dr;		// Detection grid resolution, r-direction [cm]
	double dz=(double)sim->det.dz;		// Detection grid resolution, z-direction [cm]
//	double da=PI/(2*sim->det.na);		// Angular resolution [rad]?
	
	int na=sim->det.na;			// Number of grid elements in angular-direction [-]
	int nr=sim->det.nr;			// Number of grid elements in r-direction
	int nz=sim->det.nz;			// Number of grid elements in z-direction

	int rz_size = nr*nz;
	int ra_size = nr*na;
	int r,a,z;

//	unsigned int l;
	int i=0;
	unsigned long long temp=0;
	
	double scale1 =  sim->start_weight *(double)sim->number_of_photons; //(double)0xFFFFFFFF
	double scale2;

	string tempsecoule;

	filesystem::ofstream FichierSortie;
	filesystem::path p1(OutputFileName);
	if (!filesystem::exists(p1)){
	p1 = filesystem::current_path()/p1.filename();
	}
	FichierSortie.open(p1, ios::out | ios::app);

	if(FichierSortie)  
    {

		// Calculate and write RAT
		unsigned long long Rs=0;	// Specular reflectance [-]
		unsigned long long Rd=0;	// Diffuse reflectance [-]
		unsigned long long A=0;		// Absorbed fraction [-]
		unsigned long long T=0;		// Transmittance [-]

		Rs = (unsigned long long)(0xFFFFFFFFu-sim->start_weight)*(unsigned long long)sim->number_of_photons;
		for(i=0;i<rz_size;i++)
			A+= HostMem->A_rz[i];

		for(i=0;i<ra_size;i++)
		{
			T += HostMem->Tt_ra[i];
			Rd += HostMem->Rd_ra[i];
		}
		FichierSortie <<"# RAT Reflectance, Absorption Transmission" << endl;
		FichierSortie <<"# Specular reflectance : " << (double)Rs/scale1 << endl;
		FichierSortie <<"# Diffuse reflectance : "<<(double)Rd/scale1<< endl;
		FichierSortie <<"# Absorbed fraction : "<<(double)A/scale1<< endl;
		FichierSortie <<"# Transmittance : "<< (double)T/scale1<< endl <<endl ;

		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie <<"# Fluorescence results. "<<endl ;
		FichierSortie <<"# n_emitted_fluo"<<endl;
		FichierSortie <<"# R	Z	weight	wavelength" << endl;
		FichierSortie << *(HostMem->num_emitted_fluo_photons) <<endl;
		for (int k =0;k<*(HostMem->num_emitted_fluo_photons);k++) 
		{
			FichierSortie << HostMem->fluoROut[k] << '\t' << HostMem->fluoZOut[k]<<'\t';
			FichierSortie << HostMem->fluoWeightOut[k] <<'\t' << HostMem->fluoWavelengthOut[k] << endl;
		} 
		//------------------------------------------------------------------
		// Calculate and write A_l
		/*cout <<"Writing A_l"<< endl;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie <<"A_l #Absorption as a function of layer. "<<endl ;
		i=1;
		z=0;
		for(l=1;l<=sim->n_layers;l++)
		{									*/						
			/*temp=0;
			while(((double)z+0.5)*dz<=sim->layers[l].z_max)
			{
				for(r=0;r<nr;r++) temp+=HostMem->A_rz[z*nr+r];
				z++;
				if(z==nz)break;
			}
			FichierSortie << scientific  <<(double)temp/scale1<<'\t';
			if(i == 4)
			{
				FichierSortie << endl;
				i=0;
			}
			i++;			*/
	/*	}
		cout <<"Writing A_l...OK"<< endl;
		FichierSortie << endl;*/

		//------------------------------------------------------------------
		// Calculate and write A_z
		/*cout <<"Writing A_z"<< endl;
		i=1;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		scale2=scale1*dz;
		FichierSortie <<"A_z #A[0], [1],..A[nz-1]. [1/cm]" << endl;
		for(z=0;z<nz;z++)
		{
			temp=0;
			for(r=0;r<nr;r++) temp+=HostMem->A_rz[z*nr+r]; 
			FichierSortie <<(double)temp/scale2<<'\t';
			if(i == 4)
			{
				FichierSortie << endl;
				i=0;
			}
			i++;
		}
		cout <<"Writing A_z...OK"<< endl;
		FichierSortie << endl;*/

		//------------------------------------------------------------------
		// Calculate and write Rd_r
		cout <<"Writing Rd_r"<< endl;
		i=1;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie << "Rd_r #Rd[0], [1],..Rd[nr-1]. [1/cm2]"<< endl;
		for(r=0;r<nr;r++)
		{
			temp=0;
			for(a=0;a<na;a++) temp+=HostMem->Rd_ra[a*nr+r]; 
			scale2=scale1*2*PI*(r+0.5)*dr*dr;
			FichierSortie <<(double)temp/scale2<<endl;
			i++;
		}
		cout <<"Writing Rd_r...OK"<< endl;
		FichierSortie << endl;


		//------------------------------------------------------------------
		// Calculate and write Rd_a 
	/*	cout <<"Writing Rd_a"<< endl;
		i=1;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie << "Rd_a #Rd[0], [1],..Rd[na-1]. [sr-1]" << endl;
		for(a=0;a<na;a++)
		{
			temp=0;
			for(r=0;r<nr;r++) temp+=HostMem->Rd_ra[a*nr+r]; 
			scale2=scale1*4*PI*sin((a+0.5)*da)*sin(da/2);
			FichierSortie <<(double)temp/scale2<<'\t';
			if(i == 4)
			{
				FichierSortie << endl;
				i=0;
			}
			i++;
		}
		cout <<"Writing Rd_a...OK"<< endl;
		FichierSortie << endl;*/

		//------------------------------------------------------------------
		// Calculate and write Tt_r
	/*	cout <<"Writing Tt_r"<< endl;
		i=1;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie << "Tt_r #Tt[0], [1],..Tt[nr-1]. [1/cm2]"<<endl;
		for(r=0;r<nr;r++)
		{
			temp=0;
			for(a=0;a<na;a++) temp+=HostMem->Tt_ra[a*nr+r];
			scale2=scale1*2*PI*(r+0.5)*dr*dr;
			FichierSortie << (double)temp/scale2 <<'\t' ;
			if(i == 4)
			{
				FichierSortie << endl;
				i=0;
			}
			i++;
		}
		cout <<"Writing Tt_r...OK"<< endl;
		FichierSortie << endl;*/

		//------------------------------------------------------------------
		// Calculate and write Tt_a
	/*	cout <<"Writing Tt_a"<< endl;
		i=1;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie <<"Tt_a #Tt[0], [1],..Tt[na-1]. [sr-1]"<<endl;
		for(a=0;a<na;a++)
		{
			temp=0;
			for(r=0;r<nr;r++) temp+=HostMem->Tt_ra[a*nr+r]; 
			scale2=scale1*4*PI*sin((a+0.5)*da)*sin(da/2);
			FichierSortie <<(double)temp/scale2<<'\t';
			if(i == 4)
			{
				FichierSortie << endl;
				i=0;
			}
			i++;
		}
		cout <<"Writing Tt_a...OK"<< endl;
		FichierSortie << endl;
		*/
		//------------------------------------------------------------------
		// Scale and write A_rz
		cout <<"Writing A_rz"<< endl;
		i=1;
		FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		FichierSortie <<"# A[r][z]. [1/cm3]"<<endl<<"# A[nr-1][0], [nr-1][1],..[nr-1][nz-1]"<<endl;
		for(r=0;r<nr;r++)
		{
			for(z=0;z<nz;z++)
			{
				scale2=scale1*2*PI*(r+0.5)*dr*dr*dz;
				FichierSortie << setprecision(9) << (double)HostMem->A_rz[z*nr+r]/0xffffffff<<'\n';				
			}
			FichierSortie << endl;
		}
		cout <<"Writing A_rz...OK"<< endl;
		FichierSortie << endl;

		//------------------------------------------------------------------
		// Scale and write Rd_ra 
		//cout <<"Writing Rd_ra"<< endl;
		//i=1;
		//FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		//FichierSortie <<"# Rd[r][angle]. [1/(cm2sr)]."<<endl<<" Rd[nr-1][0], [nr-1][1],..[nr-1][na-1]"<<endl;
		//for(r=0;r<nr;r++)
		//{
		//	for(a=0;a<na;a++)
		//	{
		//		scale2=scale1*2*PI*(r+0.5)*dr*dr*cos((a+0.5)*da)*4*PI*sin((a+0.5)*da)*sin(da/2);
		//		FichierSortie <<(double)HostMem->Rd_ra[a*nr+r]/scale2<<'\t';
		//		if(i == 4)
		//		{
		//			FichierSortie << endl;
		//			i=0;
		//		}
		//		i++;
		//	}
		//}
		//cout <<"Writing Rd_ra...OK"<< endl;
		//FichierSortie << endl;
		////------------------------------------------------------------------
		//// Scale and write Tt_ra
		//cout <<"Writing Tt_ra"<< endl;
		//i=1;
		//FichierSortie <<"#--------------------------------------------------------------------"<<endl;
		//FichierSortie <<"# Tt[r][angle]. [1/(cm2sr)]."<<endl<<"# Tt[nr-1][0], [nr-1][1],..[nr-1][na-1]"<< endl;
		//for(r=0;r<nr;r++)
		//{
		//	for(a=0;a<na;a++)
		//	{
		//		scale2=scale1*2*PI*(r+0.5)*dr*dr*cos((a+0.5)*da)*4*PI*sin((a+0.5)*da)*sin(da/2);
		//		FichierSortie <<(double)HostMem->Tt_ra[a*nr+r]/scale2<<'\t';
		//		if(i == 4)
		//		{
		//			FichierSortie << endl;
		//			i=0;
		//		}
		//		i++;
		//	}
		//}
 	//	cout <<"Writing Tt_ra...OK"<< endl;
		//FichierSortie << endl;
		FichierSortie.close();  
    }
    else  
		cerr << "Erreur à l'ouverture du Fichier de sortie !" << endl;
	return 0;

}

int read_simulation_data(string InputFileName, vector<SimulationStruct>* simulations, int ignoreAdetection, double Zc)
{
	int i=0;
	int ii=1;
	int jj=0;
	int k=0;
	int n_simulations;
	float dtot=0;

	string ligne;																	
	filesystem::ifstream fichier;	
	fichier.exceptions ( ifstream::eofbit | ifstream::failbit | ifstream::badbit );
	filesystem::path Path(InputFileName);
	//Building the path of the input file
	//Path/=InputFileName;
	if (!filesystem::exists(Path)) {
		Path = filesystem::current_path()/Path;
	}
	fichier.open(Path,ifstream::in);
	printf("IO : Ouverture Input File OK\n");
	try 
    {
		while (getline(fichier,ligne))									// lis Une ligne de fichier
		{
			TabString.push_back(ligne);									// rajoute chaque ligne a la fin du fichier
		}
	}
	catch(ios_base::failure e)
    {
		//cout<< "ios base failure\n";
	}
	
	//Nettoyage
	for(i=0;i<(int)TabString.size();++i) 
	{
		if( TabString[i].empty() || TabString[i][0] == '#')
		{
			TabString.erase(TabString.begin()+i);							// Efface la case.
			i--;
		}   
	}
	for(i=0;i<(int)TabString.size();++i) 
	{
		// Remove trailing comments
		TabString[i] = TabString[i].substr(0,TabString[i].find("#"));		
		// Remove trailing blank spaces
		size_t endpos = TabString[i].find_last_not_of(" \t");
		if( string::npos != endpos )
			TabString[i] = TabString[i].substr( 0, endpos+1 );
	}
	cout << endl << endl;
	fichier.close();

	stringstream ss;
	for (i=0;i<(int)TabString.size();i++)
		ss << TabString[i]<<'\n';

	
		//---------------------------------------------------------
		// Read the number of photons
		ss>>allData.layersNb;		
		ss >> allData.photonsNb;
		if(allData.photonsNb>=4200000000)
		{
			printf("Error Number of photons is too big !\n");
			printf("Truncating Number_of_photons => 4 000 000 000\n\n");
			allData.photonsNb=4000000000;
		}
		printf("---------------- Donnees Generales ---------------------------\n\n");
		printf("Number of photons: %l\n",allData.photonsNb);		
		float n_above, n_below;
		ss >> n_above;
		for (int i3=0;i3<N_WAVELENGTH;i3++) {
			allData.layers[0].n[i3]=n_above;
		}
		ss >> n_below;
		for (int i3=0;i3<N_WAVELENGTH;i3++) {
			allData.layers[allData.layersNb + 1].n[i3]=n_below;
		}
		ss >> allData.lambdaMin;
		ss >> allData.lambdaMax;
		ss >> allData.lambdaStep;
		ss >> allData.dz;
		ss >> allData.dr;
		ss >> allData.da;

		dtot=0;
		for(ii=1;ii<=allData.layersNb;ii++)
		{
			debut=0;
			ss >> allData.layers[ii].thickness;
			allData.layers[ii].z_min = dtot;
			dtot += allData.layers[ii].thickness;
			allData.layers[ii].z_max = dtot;

			ss >> allData.layers[ii].fluoNb;

			for (jj=0 ; jj<allData.layers[ii].fluoNb ; jj++)
			{	
				string dummy;
				float wavelength;


				//ss >> dummy; //discard name
				ss >> dummy; // discard concentration
				ss >> allData.layers[ii].fluo[jj].quantumYield;

				for (int kk=0;kk<N_WAVELENGTH;kk++)
				{
					ss >> wavelength;				
					ss >> allData.layers[ii].fluo[jj].absorptionSpectrum[kk];
					ss >> allData.layers[ii].fluo[jj].emissionSpectrum[kk];

					//allData.layers[ii].fluo[jj].absorptionSpectrum[1] = 0;
					//allData.layers[ii].fluo[jj].emissionSpectrum[2] = 0;
				}
				
		//		allData.layers[ii].fluo[jj].absorptionSpectrum[1] = 1;
		//		allData.layers[ii].fluo[jj].emissionSpectrum[2] = 1;
				

			}
			
			for(k=0; k<N_WAVELENGTH;k++)
			{ 
				ss >> allData.layers[ii].n[k] >> allData.layers[ii].mua[k];
				ss >> allData.layers[ii].mus[k] >> allData.layers[ii].g[k];
				if(allData.layers[ii].mus[k]==0.0f)
				{
					allData.layers[ii].mutr[k]=FLT_MAX; 
				}
				else
				{   // mutr = 1/(mua + mus)
					allData.layers[ii].mutr[k]=1.0f/(allData.layers[ii].mua[k] + allData.layers[ii].mus[k]);
				}
			}
		} 
		for ( k=0 ; k<N_WAVELENGTH ; k++ )
		{
			ss >> allData.inputLightIntensity[k];
		}
	n_simulations = PopulateSimulation(simulations, &allData,InputFileName, Zc);
	
	//Test if the number of simulations to performs doesn't exceed 200.(IOT the program doesn't crashes due to memory problem ( Allocating > Device Memory 64 Ko ))
	if (allData.lambdaStep < ((allData.lambdaMax - allData.lambdaMin)/200))
	{
		printf("IO : Error => Too many simulations! \n\a");
		printf("IO : n_simulations=0\n");
		n_simulations = 0;
	}

	return n_simulations; // return total of Simulations

}

void WriteHeader(string FolderPath,string OutputFileName, clock_t simulation_time)
{
	//stringstream Path;
	filesystem::ofstream FichierSortie;
	filesystem::path p1(OutputFileName);
	if (!filesystem::exists(p1)){
    p1= filesystem::current_path()/p1.filename();
	}
	FichierSortie.open(p1, ios::out | ios::trunc);
	if(FichierSortie)  
    {
		FichierSortie << "############################" << endl;
		FichierSortie << "#   Output File : "<< OutputFileName << endl;
		FichierSortie << "############################" << endl << endl;
		FichierSortie << "# Simulation time :"<<(double)simulation_time/CLOCKS_PER_SEC <<" sec #"<<endl<<endl;
		FichierSortie <<"A1 	# Version number of the file format."<<endl<<endl;
		FichierSortie <<"###########################################################"<< endl;
		FichierSortie <<"# Data categories include :  "<< endl;
		FichierSortie <<"# InParm -> Input Parameters "<< endl;
		FichierSortie <<"# A_l -> Absorption as a function of layer.   "<<endl;
		FichierSortie <<"# A_z -> Somme des A_rz dans la direction r   "<<endl;
		FichierSortie <<"# Rd_r -> ???                                 "<<endl;
		FichierSortie <<"# Rd_a -> Somme des Rd_rz dans la direction z "<<endl;
		FichierSortie <<"# Tt_r -> Somme des Tt_rz dans la direction a "<<endl;
		FichierSortie <<"# Tt_a -> Somme des Tt_rz dans la direction r "<<endl;
		FichierSortie <<"#                                             "<<endl;
		FichierSortie <<"# A_rz -> Absorption coordonnées cylindrique  "<<endl;
		FichierSortie <<"# Rd_ra -> Reflectance diffuse 2D en coordonnées cylindrique "<<endl;
		FichierSortie <<"# Tt_ra -> Transmittance 2D en coordonnées cylindrique "<<endl;
		FichierSortie <<"############################################################"<<endl<<endl;

	}
	else
		printf(" IO.cu : ERROR -> Cannot open Output file ! \n");
	
	FichierSortie.close();

}


int init_RNG(char* exePath, unsigned long long *x, unsigned int *a, const unsigned int n_rng, unsigned long long xinit)
{
    FILE *fp;
    unsigned int begin=0u;
	unsigned int fora,tmp1,tmp2;
    string safeprimes_file;
    filesystem::path sp(filesystem::current_path());
    sp/="safeprimes_base32.txt";
    if (filesystem::exists(sp)) {
     safeprimes_file = (string)sp.string();
    } else { //try in the same folder as exe
     sp = filesystem::path(exePath);
     sp = sp.parent_path();
     sp/="safeprimes_base32.txt";     
     safeprimes_file = (string)sp.string();
    }
    fp = fopen(safeprimes_file.c_str(), "r");

	if(fp == NULL)
	{
		printf("Could not find the file of safeprimes (%s)! Terminating!\n", safeprimes_file);
		return 1;
	}

	fscanf(fp,"%u %u %u",&begin,&tmp1,&tmp2);

	// Here we set up a loop, using the first multiplier in the file to generate x's and c's
	// There are some restictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	//Make sure xinit is a valid seed (using the above mentioned restrictions)
	if((xinit == 0ull) | (((unsigned int)(xinit>>32))>=(begin-1)) | (((unsigned int)xinit)>=0xfffffffful))
	{
		//xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
		printf("%llu not a valid seed! Terminating!\n",xinit);
		return 1;
	}

	for (unsigned int i=0;i < n_rng;i++)
    {
		fscanf(fp,"%u %u %u",&fora,&tmp1,&tmp2);
		a[i]=fora;
		x[i]=0;
		while( (x[i]==0) | (((unsigned int)(x[i]>>32))>=(fora-1)) | (((unsigned int)x[i])>=0xfffffffful))
		{
			//generate a random number
			xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);

			//calculate c and store in the upper 32 bits of x[i]
			x[i]=(unsigned int) floor((((double)((unsigned int)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
			x[i]=x[i]<<32;

			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
			x[i]+=(unsigned int) xinit;
		}
		//if(i<10)printf("%llu\n",x[i]);
    }
    fclose(fp);

	return 0;
}
// /RANDOM