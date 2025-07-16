//#This file is part of PyTransport.

//#PyTransport is free software: you can redistribute it and/or modify
//#it under the terms of the GNU General Public License as published by
//#the Free Software Foundation, either version 3 of the License, or
//#(at your option) any later version.

//#PyTransport is distributed in the hope that it will be useful,
//#but WITHOUT ANY WARRANTY; without even the implied warranty of
//#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#GNU General Public License for more details.

//#You should have received a copy of the GNU General Public License
//#along with PyTransport.  If not, see <http://www.gnu.org/licenses/>.

// This file contains a prototype of the potential.h file of PyTransport -- it is edited by the PyTransScripts module

#ifndef POTENTIAL_H  // Prevents the class being re-defined
#define POTENTIAL_H


#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;

// #Rewrite
// Potential file rewriten at Fri Jul 11 15:15:23 2025

class potential
{
private:
	int nF; // field number
	int nP; // params number which definFs potential
    
    
public:
	// flow constructor
	potential()
	{
// #FP
nF=1;
nP=3;

//        p.resize(nP);
        
// pdef

    }
	
    //void setP(vector<double> pin){
    //    p=pin;
    //}
	//calculates V()
	double V(vector<double> f, vector<double> p)
	{
		double sum ;
        
// Pot
    sum=p[0]*std::pow(1 - std::exp(f[0]*p[1]), p[2]);
         return sum;
	}
	
	//calculates V'()
	vector<double> dV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF,0.0);
	
// dPot
  double x0 = std::exp(f[0]*p[1]);

 sum[0]=-p[0]*p[1]*p[2]*x0*std::pow(1 - x0, p[2] - 1);
        
		return sum;
	}
    
	// calculates V''
	vector<double> dVV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF*nF,0.0);
		
// ddPot
  double x0 = std::exp(f[0]*p[1]);
  double x1 = x0 - 1;
  double x2 = -x1;

 sum[0]=-p[0]*std::pow(p[1], 2)*p[2]*x0*(x0*std::pow(x2, p[2] + 1)*(p[2] - 1) - std::pow(x2, p[2] + 2))/std::pow(x1, 3);
     
        return sum;
	}
    
	// calculates V'''
	vector<double> dVVV(vector<double> f, vector<double> p)
	{
        vector<double> sum(nF*nF*nF,0.0);
// dddPot
  double x0 = f[0]*p[1];
  double x1 = std::exp(x0);
  double x2 = 1 - x1;

 sum[0]=p[0]*std::pow(p[1], 3)*p[2]*x1*(3*x1*std::pow(x2, p[2] + 4)*(p[2] - 1) - std::pow(x2, p[2] + 3)*(std::pow(p[2], 2) - 3*p[2] + 2)*std::exp(2*x0) - std::pow(x2, p[2] + 5))/std::pow(x2, 6);
       
        return sum;
	}
    
    int getnF()
    {
        return nF;
    }
    
    int getnP()
    {
        return nP;
    }

};
#endif