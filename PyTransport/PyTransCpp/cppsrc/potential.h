/**
 * @file potentialProto.h
 * @brief Prototype header for automatically generated scalar field potential and derivative calculations.
 *
 * This file serves as a template that `PyTransSetup.py` uses to generate `potential.h`.
 * It defines the function signatures for evaluating the potential, its gradient,
 * Hessian, and third derivatives. The actual implementations are
 * symbolically derived and inserted by the Python setup script.
 *
 * @copyright GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * @see http://www.gnu.org/licenses/
 */

#ifndef POTENTIAL_H  // Prevents the class being re-defined
#define POTENTIAL_H


#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;

// #Rewrite
// Potential file rewriten at Mon Oct 27 11:47:36 2025

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
  sum=0.5*std::pow(f[0], 2)*p[1] + 0.5*p[4]*std::pow(f[1] - p[5], 2);
         return sum;
    }
    
    //calculates V'()
    vector<double> dV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF,0.0);
    
// dPot

 sum[0]=1.0*f[0]*p[1];

 sum[1]=0.5*p[4]*(2*f[1] - 2*p[5]);
        
        return sum;
    }
    
    // calculates V''
    vector<double> dVV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF*nF,0.0);
        
// ddPot
  double x0 = 1.0/p[0];
  double x1 = -0.5*f[0]*p[1]*x0;

 sum[0]=1.0*p[1] + 0.25*p[4]*x0*(2*f[1] - 2*p[5])*std::exp(f[1]*x0);

 sum[2]=x1;

 sum[1]=x1;

 sum[3]=1.0*p[4];
     
        return sum;
    }
    
    // calculates V'''
    vector<double> dVVV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF*nF*nF,0.0);
// dddPot
  double x0 = 1.0/p[0];
  double x1 = std::exp(f[1]*x0);
  double x2 = std::pow(p[0], -2);
  double x3 = f[0]*p[1]*x2;
  double x4 = 0.5*x3;
  double x5 = p[4]*x1;
  double x6 = 0.25*x5*(2*f[1] - 2*p[5]);
  double x7 = x0*(1.0*p[1] + x0*x6);
  double x8 = 0.5*x0;
  double x9 = -p[1]*x8 + 0.5*p[4]*x0*x1 - 1.0/2.0*x7;
  double x10 = 0.25*x3;

 sum[0]=-x1*x4;

 sum[4]=x2*x6 + x5*x8 - x7;

 sum[2]=x9;

 sum[6]=x10;

 sum[1]=x9;

 sum[5]=x10;

 sum[3]=x4;

 sum[7]=0;
       
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