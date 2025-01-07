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
// Potential file rewriten at Tue Jan  7 16:07:36 2025

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
nF=2;
nP=4;

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
  double x0 = std::pow(f[0], 2);
  sum=0.5*std::pow(f[1], 2)*p[2] + p[0]*x0/(std::pow(p[1], 2) + x0);
         return sum;
	}
	
	//calculates V'()
	vector<double> dV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF,0.0);
	
// dPot
  double x0 = std::pow(f[0], 2) + std::pow(p[1], 2);

 sum[0]=-2*std::pow(f[0], 3)*p[0]/std::pow(x0, 2) + 2*f[0]*p[0]/x0;

 sum[1]=1.0*f[1]*p[2];
        
		return sum;
	}
    
	// calculates V''
	vector<double> dVV(vector<double> f, vector<double> p)
	{
		vector<double> sum(nF*nF,0.0);
		
// ddPot
  double x0 = std::pow(f[0], 2);
  double x1 = std::pow(p[1], 2) + x0;
  double x2 = 1.0/x1;
  double x3 = 2*p[0];
  double x4 = std::pow(x1, -2);
  double x5 = 1.0*p[2];
  double x6 = -f[1]*p[3]*x5;

 sum[0]=8*std::pow(f[0], 4)*p[0]/std::pow(x1, 3) - 10*p[0]*x0*x4 + x2*x3;

 sum[2]=x6;

 sum[1]=x6;

 sum[3]=1.0*p[3]*(-std::pow(f[0], 3)*x3*x4 + 2*f[0]*p[0]*x2)*std::exp(2.0*f[0]*p[3]) + x5;
     
        return sum;
	}
    
	// calculates V'''
	vector<double> dVVV(vector<double> f, vector<double> p)
	{
        vector<double> sum(nF*nF*nF,0.0);
// dddPot
  double x0 = std::pow(f[0], 3);
  double x1 = std::pow(f[0], 2);
  double x2 = std::pow(p[1], 2) + x1;
  double x3 = std::pow(x2, -3);
  double x4 = std::pow(x2, -2);
  double x5 = p[0]*x4;
  double x6 = 1.0*p[2];
  double x7 = std::pow(p[3], 2);
  double x8 = f[1]*x7;
  double x9 = x6*x8;
  double x10 = 1.0/x2;
  double x11 = 2*p[0];
  double x12 = 2*f[0]*p[0]*x10 - x0*x11*x4;
  double x13 = 2.0*p[3];
  double x14 = std::exp(f[0]*x13);
  double x15 = 1.0*p[3];
  double x16 = x14*x15;
  double x17 = x12*x16 + x6;
  double x18 = 8*std::pow(f[0], 4)*p[0]*x3 - 10*x1*x5 + x10*x11;
  double x19 = 2.0*x14;
  double x20 = p[2]*x8;
  double x21 = 1.0*p[3]*x14*x18 - p[3]*x6 - x15*x17;

 sum[0]=-48*std::pow(f[0], 5)*p[0]/std::pow(x2, 4) - 24*f[0]*x5 + 72*p[0]*x0*x3;

 sum[4]=2.0*x20;

 sum[2]=x9;

 sum[6]=x21;

 sum[1]=x9;

 sum[5]=x21;

 sum[3]=x12*x19*x7 - x13*x17 + x16*x18;

 sum[7]=-x19*x20;
       
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