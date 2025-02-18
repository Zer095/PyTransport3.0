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
// Potential file rewriten at Fri Feb 14 13:18:35 2025

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
nP=6;

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
  sum=p[0]*(-p[1]*std::tanh((f[0] - p[2])/p[3]) + 1.0)*(-std::pow(f[0], 2)*p[4]/(f[0]/p[5] + 1) + 1.0);
         return sum;
	}
	
	//calculates V'()
	vector<double> dV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF,0.0);
	
// dPot
  double x0 = 1.0/p[3];
  double x1 = std::tanh(x0*(f[0] - p[2]));
  double x2 = 1.0/p[5];
  double x3 = f[0]*x2 + 1;
  double x4 = p[4]/x3;
  double x5 = std::pow(f[0], 2);

 sum[0]=-p[0]*p[1]*x0*(1 - std::pow(x1, 2))*(-x4*x5 + 1.0) + p[0]*(-2*f[0]*x4 + p[4]*x2*x5/std::pow(x3, 2))*(-p[1]*x1 + 1.0);
        
		return sum;
	}
    
	// calculates V''
	vector<double> dVV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF*nF,0.0);
		
// ddPot
  double x0 = 1.0/p[3];
  double x1 = std::tanh(x0*(f[0] - p[2]));
  double x2 = p[1]*x1;
  double x3 = 1.0/p[5];
  double x4 = f[0]*x3 + 1;
  double x5 = p[4]/x4;
  double x6 = 2*x5;
  double x7 = std::pow(x4, -2);
  double x8 = std::pow(f[0], 2);
  double x9 = p[4]*x8;
  double x10 = 2*p[0]*(1 - std::pow(x1, 2));

 sum[0]=p[0]*(1.0 - x2)*(4*f[0]*p[4]*x3*x7 - x6 - 2*x9/(std::pow(p[5], 2)*std::pow(x4, 3))) - p[1]*x0*x10*(-f[0]*x6 + x3*x7*x9) + x10*x2*(-x5*x8 + 1.0)/std::pow(p[3], 2);
     
        return sum;
	}
    
	// calculates V'''
	vector<double> dVVV(vector<double> f, vector<double> p)
	{
        vector<double> sum(nF*nF*nF,0.0);
// dddPot
  double x0 = 1.0/p[3];
  double x1 = std::tanh(x0*(f[0] - p[2]));
  double x2 = p[1]*x1;
  double x3 = 6*p[4];
  double x4 = 1.0/p[5];
  double x5 = f[0]*x4 + 1;
  double x6 = std::pow(x5, -2);
  double x7 = x4*x6;
  double x8 = 1/(std::pow(p[5], 2)*std::pow(x5, 3));
  double x9 = std::pow(f[0], 2);
  double x10 = p[4]/x5;
  double x11 = 2*x10;
  double x12 = p[4]*x9;
  double x13 = std::pow(x1, 2);
  double x14 = 1 - x13;
  double x15 = p[0]*p[1];
  double x16 = x14*x15;
  double x17 = (-x10*x9 + 1.0)/std::pow(p[3], 3);

 sum[0]=p[0]*(1.0 - x2)*(-12*f[0]*p[4]*x8 + x3*x7 + x3*x9/(std::pow(p[5], 3)*std::pow(x5, 4))) + 6*p[0]*x14*x2*(-f[0]*x11 + x12*x7)/std::pow(p[3], 2) - 3*x0*x16*(4*f[0]*p[4]*x4*x6 - x11 - 2*x12*x8) - 4*x13*x16*x17 + 2*std::pow(x14, 2)*x15*x17;
       
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