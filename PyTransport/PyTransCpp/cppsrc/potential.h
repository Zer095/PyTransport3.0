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
// Potential file rewriten at Thu Feb  5 16:41:25 2026

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
  sum=0.5*std::pow(f[0], 2)*p[0] + p[1]*std::cos(f[0]/p[2]);
         return sum;
    }
    
    //calculates V'()
    vector<double> dV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF,0.0);
    
// dPot
  double x0 = 1.0/p[2];

 sum[0]=1.0*f[0]*p[0] - p[1]*x0*std::sin(f[0]*x0);
        
        return sum;
    }
    
    // calculates V''
    vector<double> dVV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF*nF,0.0);
        
// ddPot

 sum[0]=1.0*p[0] - p[1]*std::cos(f[0]/p[2])/std::pow(p[2], 2);
     
        return sum;
    }
    
    // calculates V'''
    vector<double> dVVV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF*nF*nF,0.0);
// dddPot

 sum[0]=p[1]*std::sin(f[0]/p[2])/std::pow(p[2], 3);
       
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