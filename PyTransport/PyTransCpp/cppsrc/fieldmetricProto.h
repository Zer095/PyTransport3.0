/**
 * @file fieldmetricProto.h
 * @brief Prototype header for automatically generated field space metric and curvature calculations.
 *
 * This file serves as a template that `PyTransSetup.py` uses to generate `fieldmetric.h`.
 * It defines the function signatures for computing the field space metric, Christoffel symbols,
 * Riemann tensor, and its covariant derivative. The actual implementations are
 * symbolically derived and inserted by the Python setup script.
 *
 * @copyright GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * @see http://www.gnu.org/licenses/
 */


#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

// #Rewrite
#ifndef FIELDMETRIC_H  // Prevents the class being re-defined
#define FIELDMETRIC_H


#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;


class fieldmetric
{
private:
	int nF; // field number
    int nP; // params number which definFs potential

public:
    fieldmetric()
   {
// #FP
	
   }
	
	
	//calculates fieldmetic()
	vector<double> fmetric(vector<double> f, vector<double> p)
	{
		vector<double> FM((2*nF)*(2*nF),0.0) ;
        
// metric

         return FM;
	}
	
	
	
	//calculates ChristoffelSymbole()
	vector<double> Chroff(vector<double> f, vector<double> p)
	{
		vector<double> CS((2*nF)*(2*nF)*(2*nF),0.0);
	
// Christoffel
        
		return CS;
	}
    

	
	// calculates RiemannTensor()
	vector<double> Riemn(vector<double> f, vector<double> p)
	{
		vector<double> RM((nF)*(nF)*(nF)*(nF),0.0);
		
// Riemann
     
        return RM;
	}

	// calculates RiemannTensor() covariant derivatives
	vector<double> Riemncd(vector<double> f, vector<double> p)
	{
		vector<double> RMcd((nF)*(nF)*(nF)*(nF)*(nF),0.0);
		
// Riemanncd
     
        return RMcd;
	}
    
    int getnF()
    {
        return nF;
    }
    


};
#endif