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

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef back_H  // Prevents the class being re-definFd
#define back_H 
#include "model.h"
#include <iostream>
#include "../fieldmetric.h"
#include <math.h>
#include <cmath>
#include <vector>
using namespace std;
/**
 * Implementation of the inflationary background i.e. field and fields' derivatives values.
 */
class back 
{	
private:
	int nF;
	vector<double> f;
    fieldmetric fmet;

public:
	//constructor for background store
	back(int nFi, vector<double> f )
	{
    nF=nFi;
        f.resize(2*nF);
    }
    //back accessor methods
	vector<double> getB()
	{
		return f;
	}
	
    double getB(int i)
	{
		return f[i];
	}
    //back modifier methods
	void setB(int i, double value)
	{
		f[i]=value;
	}
	void setB(vector<double> y)
	{
		f=y;
	}
    //print back to screen
	void printB()
	{
		for(int i=0;i<2*nF;i++){std::cout << f[i] << '\t';}
		std::cout << std::endl;
	}
};
#endif 

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef sigma_H  // Prevents the class being re-definFd
#define sigma_H 
#include "model.h"
#include <iostream>
#include <math.h>
using namespace std;

class sigma 
{	
private:
	int nF;
	vector<double> sig;
	double k;

public:

    /*-----------------------------------------------------------------------------------------------------------------------------------
        Constructor methods for Sigma. Define the initial conditions for Sigma deep inside the horizon, or at horizon crossing. 

        Arguments
        --------
        nFi : int
            Number of fields.
        k1 : double
            Value of the k mode. If given, it returns the initial conditions deep inside the horizon.
        N0 : double
            initial time.
        f : vector<double>
            Values of the background.
        p : vector<double>
            Values of the model's parameters

        Returns
        -------
        None

        Description
        -----------

        The constructor methods populates the Sigma class with initial conditions.
        If k1 is given, the constructor computes the initial conditions deep inside the horizon.
        If k1 is NOT given, the contructor computes the initial conditions at horizon crossing (for canonical light fields).
    ------------------------------------------------------------------------------------------------------------------------------------*/

	//constructor for sig deep inside horizon
	sigma(int nFi, double k1, double N0, vector<double> f, vector<double> p)
	{
		nF=nFi;
		k=k1;
		
		fieldmetric fmet;
		vector<double> FMi;
		FMi = fmet.fmetric(f,p);
		
        double Hi;
		model m;
		double a = exp(N0);
		double s = m.scale(f,p,N0);
		sig.resize(2*nFi*2*nFi);
		
		for(int i=0; i<2*nFi*2*nFi;i++){sig[i]=0.;}
		Hi=m.H(f,p);
    	double ff = 0.5*Hi*Hi/k * 1./((a*Hi)*(a*Hi));
		double fp = -ff*Hi*s;
		double pp = ff*k*k/(a*a)*s*s;
		
        for(int i = 0; i<nFi; i++){
			for(int j=0; j<nFi;j++){ 
				sig[i + 2*nFi*j]=FMi[((2*nF)*(i)+j)]*ff;
			}
		}
		for(int i = nFi; i<2*nFi; i++){
			for(int j=nFi; j<2*nFi;j++){ 
				sig[i + 2*nFi*j]=FMi[((2*nF)*(i-nFi)+j-nFi)]*pp;
			}
		}
		for(int i = 0; i<nFi; i++){
			for(int j=nFi; j<2*nFi;j++){ 
				sig[i + 2*nFi*(j)]=FMi[((2*nF)*(i)+j-nFi)]*fp; 
			}
		}
		for(int i = nFi; i<2*nFi; i++){
			for(int j=0; j<nFi;j++){ 
				sig[(i) + 2*nFi*(j)]=FMi[((2*nF)*(i-nFi)+j)]*fp;
			}
		}
    }
    //constuctor for sig at horizon crossing (for canonical light field)
    sigma(int nFi, double N0, vector<double> f,vector<double> p)
	{
    	nF=nFi;
	    double Hi;
		model m;
        potential pot;
		fieldmetric fmet;

		//double a = exp(N0);
        sig.resize(2*nF*2*nF);
		for(int i=0; i<2*nFi*2*nFi;i++){sig[i]=0.;}
		Hi=m.H(f,p);
        vector<double> dVVi, dVi;
        double Vi ;
        Vi = pot.V(f,p);
        dVi = pot.dV(f,p);
        dVVi = pot.dVV(f,p);
        //double Hdi = m.Hdot(f,p);
    	double ff = 0.5*0.5*Hi*Hi/(3.142)/3.142;
		
        
        for(int i=0;i<2*nFi*2*nFi;i++){sig[i]=0;}
        
        for(int i = 0; i<nFi; i++){for(int j = 0; j<nFi; j++){
            if (i==j){sig[i + 2*nFi*j]=ff;
                double sum1=0;
                for(int k=0;k<nFi;k++){ sum1 = sum1 + (-dVVi[i +k*nFi]/Vi + dVi[i]*dVi[i]/Vi/Vi );}
                sig[i+nF + 2*nFi*(j)] = sum1*ff*Hi;
                sig[i+ 2*nFi*(j+nFi)] = sum1*ff*Hi;
                sig[i+nFi +2*nF*(j+nFi)] = sum1*sum1*ff*Hi*Hi;}}}
    }
    /**
     *  Sigma accessor methods. 
     *  It takes 0 arguments, and returns the vector sig.
     *  It takes 2 integer, it returns the (i,j) element of sigma.
     */
    vector<double>  getS()
	{
		return sig;
	}
	
    double getS(int i, int j)
	{
		return sig[i+j*2*nF];
	}
	/**
     *  Sigma modifier methods. 
     *  It takes two integer (i,j) and a double (value), sets sig(i,j) = value
     *  It takes 1 vector (value), it sets sig = value.
     */
	void setS(int i, int j, double value)
	{
		sig[i+j*2*nF]=value;
	}
	
    void setS(vector<double> value)
	{
		sig=value;
	}
    //print sig to screen
	void printS()
	{
		for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++) {std::cout << sig[i+j*2*nF] << '\t';}}
		std::cout << std::endl;
	}
};
#endif 

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef sigmaI_H  // Prevents the class being re-definFd
#define sigmaI_H 
#include "model.h"
#include <iostream>
#include <math.h>
using namespace std;
/**
 * Implementation of the IMAGINARY phase-space two-point correlation function.
*/
class sigmaI 
{	
private:
	int nF;
	vector<double> sigI;
	double k;

public:
    /*-----------------------------------------------------------------------------------------------------------------------------------
        Constructor methods for Sigma. Define the initial conditions for Sigma deep inside the horizon.

        Arguments
        --------
        nFi : int
            Number of fields.
        k1 : double
            Value of the k mode.
        N0 : double
            initial time.
        f : vector<double>
            Values of the background.
        p : vector<double>
            Values of the model's parameters

        Returns
        -------
        None

        Description
        -----------

        The constructor methods populates the Sigma class with initial conditions.
    ------------------------------------------------------------------------------------------------------------------------------------*/
	//constructor for sigI
	sigmaI(int nFi, double k1, double N0, vector<double> f,vector<double> p)
	{
		nF=nFi;
		fieldmetric fmet;

		vector<double> FMi;
		FMi = fmet.fmetric(f,p);
		double Hi;
        k=k1;
		model m;
		double a = exp(N0);
		double s=m.scale(f,p,N0);

        sigI.resize(2*nF*2*nF);
        for(int i=0; i<2*nFi*2*nFi;i++){sigI[i]=0.;}
		Hi=m.H(f,p);
		
		double fpI = 0.5*Hi*Hi * 1./(a*Hi)/(a*Hi)/a *s;
		
        for(int i = 0; i<nF; i++){
			for(int j = 0; j<nF; j++){
				sigI[i+(nF+j)*2*nF] = +FMi[((2*nF)*(i)+j)]*fpI;
				sigI[i+nF+(j)*2*nF] = -FMi[((2*nF)*(i)+j)]*fpI;
			}
        }	
	}
    /**
     *  Sigma accessor methods. 
     *  It takes 0 arguments, and returns the vector sig.
     *  It takes 2 integer, it returns the (i,j) element of sigma.
     */
	vector<double>  getSI()
	{
		return sigI;	
	}
	double getS(int i, int j)
	{
		return sigI[i+j*2*nF];
	}
	/**
     *  Sigma modifier methods. 
     *  It takes two integer (i,j) and a double (value), sets sig(i,j) = value
     *  It takes 1 vector (value), it sets sig = value.
     */
	void setS(int i, int j, double value)
	{
		sigI[i+j*2*nF]=value;
	}
	void setS(vector<double> value)
	{
		sigI=value;
	}
	//print sig to screen 
	void printS()
	{
		for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++) {std::cout << sigI[i+j*2*nF] << '\t';}}
		std::cout << std::endl;
	}
};
#endif 

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef Gamma_H  // Prevents the class being re-definFd
#define Gamma_H 
#include "model.h"
#include <iostream>
#include <math.h>
using namespace std;
/**
 * Implementation of the Tensor two-point correlation function.
*/
class Gamma
{	
private:
	int nF;
	vector<double> gam;
	double k;

public:
    /*-----------------------------------------------------------------------------------------------------------------------------------
        Constructor methods for Gamma. Define the initial conditions for Gamma deep inside the horizon.

        Arguments
        --------
        nFi : int
            Number of fields.
        k1 : double
            Value of the k mode.
        N0 : double
            initial time.
        f : vector<double>
            Values of the background.
        p : vector<double>
            Values of the model's parameters

        Returns
        -------
        None

        Description
        -----------

        The constructor methods populates the Gamma class with initial conditions.
    ------------------------------------------------------------------------------------------------------------------------------------*/
	//constructor for gam deep inside horizon
	Gamma(int nFi, double k1, double N0, vector<double> f, vector<double> p)
	{
		int nT = 1;
		k=k1;
        double Hi;
		model m;
        gam.resize(2*nT*2*nT);
		double a = exp(N0);


        for(int i=0; i<2*nT*2*nT;i++){gam[i]=0.;}
        Hi=m.H(f,p);
        double ff = 0.5*Hi*Hi/k * 1./((a*Hi)*(a*Hi));
        double fp = -ff*Hi;
        double pp = ff*k*k/(a*a);


        gam[0 + 2*nT*0]=ff;
		gam[1 + 2*nT*1]=pp;
		gam[0 + 2*nT*1]=fp; 
		gam[1 + 2*nT*(0)]=fp;
    }
    /**
     *  Gamma accessor methods. 
     *  It takes 0 arguments, and returns the vector gam.
     *  It takes 2 integer, it returns the (i,j) element of gam.
     */
    vector<double>  getG()
	{
		return gam;
	}
	
    double getG(int i, int j)
	{
		return gam[i+j*2];
	}
	/**
     *  Gamma modifier methods. 
     *  It takes two integer (i,j) and a double (value), sets gam(i,j) = value
     *  It takes 1 vector (value), it sets gam = value.
     */
	void setG(int i, int j, double value)
	{
		gam[i+j*2]=value;
	}
	
    void setG(vector<double> value)
	{
		gam=value;
	}
    //print gam to screen
	void printG()
	{
		for(int i=0;i<2;i++){for(int j=0;j<2;j++) {std::cout << gam[i+j*2] << '\t';}}
		std::cout << std::endl;
	}
};
#endif 

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef alpha_H  // Prevents the class being re-definFd
#define alpha_H 
#include "model.h"
#include <iostream>
#include <math.h>
using namespace std;
/**
 * Implementation of the REAL phase-space three-point correlation function.
*/
class alpha
{	
private:
	int nF;
	vector<double> alp;
	double k1, k2, k3;
         
    public:
    /*-----------------------------------------------------------------------------------------------------------------------------------
        Constructor methods for Alpha. Define the initial conditions for Alpha deep inside the horizon.

        Arguments
        --------
        nFi : int
            Number of fields.
        k1 : double
            Value of the first k mode.
        k2 : double
            Value of the second k mode.
        k3 : double
            Value of the third k mode.
        N0 : double
            initial time.
        f : vector<double>
            Values of the background.
        p : vector<double>
            Values of the model's parameters

        Returns
        -------
        None

        Description
        -----------

        The constructor methods populates the Alpha class with initial conditions.
        If the k modes aren't given, the constructor initialize Alpha = 0.
    ------------------------------------------------------------------------------------------------------------------------------------*/

    //constructor for alp on subhorizon scales
	alpha(int nFi, double k1, double k2, double k3, double N0, vector<double> f,vector<double> p)
    {
        vector<double>  fff, pff, fpf, ffp, ppf, fpp,pfp,ppp;
        double a;
        nF=nFi;
        model m;
        double s=m.scale(f,p,N0);
        a = exp(N0);

        fff = fffCalc(f,p, k1, k2, k3,N0);
		
        pff = pffCalc(f,p,k1,k2,k3,N0);
        fpf = pffCalc(f,p,k2,k1,k3,N0);
        ffp = pffCalc(f,p,k3,k1,k2,N0);

        ppf = ppfCalc(f,p,k1,k2,k3,N0);
        pfp = ppfCalc(f,p,k1,k3,k2,N0);
        fpp = ppfCalc(f,p,k2,k3,k1,N0);
        
        ppp = pppCalc(f,p,k1,k2,k3,N0);
            


        alp.resize(2*nF*2*nF*2*nF);
        
        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
          alp[i + j*2*nF + k*2*nF*2*nF] = fff[i + j*nF + k*nF*nF] ;
            alp[nF+i + j*2*nF + k*2*nF*2*nF] = pff[i + j*nF + k*nF*nF]/a   *s;// *a;
            alp[i + (nF+j)*2*nF + k*2*nF*2*nF] = fpf[j + i*nF + k*nF*nF]/a   *s;// *a;
            alp[i + j*2*nF + (nF+k)*2*nF*2*nF ] = ffp[k + i*nF + j*nF*nF ]/a   *s;// *a;
            alp[i + (nF+j)*2*nF + (nF+k)*2*nF*2*nF] = fpp[j + k*nF + i*nF*nF]/a/a  *s*s;// *a*a;
            alp[(nF+i) + (nF+j)*2*nF + (k)*2*nF*2*nF] = ppf[i + j*nF + k*nF*nF ]/a/a  *s*s;// *a*a;
            alp[(nF+i) + (j)*2*nF + (nF+k)*2*nF*2*nF] = pfp[i + k*nF + j*nF*nF ]/a/a   *s*s;// *a*a;
            alp[(nF+i) + (nF+j)*2*nF + (nF+k)*2*nF*2*nF ] = ppp[i + j*nF + k*nF*nF ]/a/a/a  *s*s*s;// *a*a*a;
        }}}

    }
	
    alpha(int nFi, double N0, vector<double> f,vector<double> p)
    {
        k1=0.;k2=0;k3=3;
        nF=nFi;
        alp.resize(2*nF*2*nF*2*nF);
        for(int i=0;i<2*nF*2*nF*2*nF;i++){
            alp[i]=0;
        }
		
    }
    /**
     * Auxiliary functions to compute the initial conditions.
     * 
     */
    vector<double> fffCalc(vector<double> f, vector<double> p, double k1, double k2, double k3, double N0)
    {
        double  ks, K2;
        vector<double> C123, C132, C231, B123, B132, B231, AS123, AS132, AS231;
		fieldmetric fmet;

		vector<double> FMi;
		FMi = fmet.fmetric(f,p);
        model m;
        C123 = m.Ccalcuuu(f,p, k1, k2, k3, N0);
        C132 = m.Ccalcuuu(f,p, k1, k3, k2, N0);
        C231 = m.Ccalcuuu(f,p, k2, k3, k1, N0);

        AS123 = m.AScalcuuu(f, p, k1, k2, k3, N0);
        AS132 = m.AScalcuuu(f, p, k1, k3, k2, N0);
        AS231 = m.AScalcuuu(f, p, k2, k3, k1, N0);

        B123 = m.Bcalcuuu(f, p, k1, k2, k3, N0);
        B132 = m.Bcalcuuu(f, p, k1, k3, k2, N0);
        B231 = m.Bcalcuuu(f, p, k2, k3, k1, N0);
        
        vector<double> fff(nF*nF*nF);
        double Hi= m.H(f,p);
        
        double a =exp(N0);
        ks = k1 + k2 + k3;
        K2 = k1*k2 + k1*k3 + k2*k3;
        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
                        fff[i+j*nF+k*nF*nF] = 1./(a*a*a*a)/4. / (k1*k2*k3)/ks*(-C123[i+nF*j+nF*nF*k]*k1*k2 - C132[i+nF*k+nF*nF*j]*k1*k3 - C231[j+nF*k+nF*nF*i]*k2*k3
                                                                   + a*a*AS123[i+nF*j+nF*nF*k] + a*a*AS132[i+nF*k+nF*nF*j] + a*a*AS231[j+nF*k+nF*nF*i]
                                                                   + a*a*Hi*B123[i+nF*j+nF*nF*k]*((k1+k2)*k3/k1/k2 - K2/k1/k2)
                                                                   + a*a*Hi*B132[i+nF*k+nF*nF*j]*((k1+k3)*k1/k1/k3 - K2/k1/k3)
                                                                   + a*a*Hi*B231[j+nF*k+nF*nF*i]*((k2+k3)*k1/k2/k3 - K2/k2/k3)); 
            fff[i+j*nF+k*nF*nF] = fff[i+j*nF+k*nF*nF] + FMi[((2*nF)*(j)+k)]*1./(a*a*a*a)/4./(k1*k2*k3)/ks*f[nF+i]/2./Hi*(-k2*k2-k3*k3+k1*k1)/2.;
            fff[i+j*nF+k*nF*nF] = fff[i+j*nF+k*nF*nF] + FMi[((2*nF)*(i)+k)]*1./(a*a*a*a)/4./(k1*k2*k3)/ks*f[nF+j]/2./Hi*(-k1*k1-k3*k3+k2*k2)/2.;
            fff[i+j*nF+k*nF*nF] = fff[i+j*nF+k*nF*nF] + FMi[((2*nF)*(i)+j)]*1./(a*a*a*a)/4./(k1*k2*k3)/ks*f[nF+k]/2./Hi*(-k1*k1-k2*k2+k3*k3)/2.;
        }}}
        return fff;
    }

    vector<double> pffCalc(vector <double> f,vector<double> p, double k1, double k2, double k3, double N0)
    {
        double a, Hi, ks, k3s, K2;
        vector<double> pff(nF*nF*nF);
        vector<double> C123, C132, C231, B123, B132, B231, AS123, AS132, AS231;
		fieldmetric fmet;

        vector<double> FMi;
		FMi = fmet.fmetric(f,p);
		
        model m;
        
        C123 = m.Ccalcuuu(f, p, k1, k2, k3, N0);
        C132 = m.Ccalcuuu(f, p, k1, k3, k2, N0);
        C231 = m.Ccalcuuu(f, p, k2, k3, k1, N0);
        
        B123 = m.Bcalcuuu(f, p, k1, k2, k3, N0);
        B132 = m.Bcalcuuu(f, p, k1, k3, k2, N0);
        B231 = m.Bcalcuuu(f, p, k2, k3, k1, N0);
        
		AS123 = m.AScalcuuu(f, p, k1, k2, k3, N0);
        AS132 = m.AScalcuuu(f, p, k1, k3, k2, N0);
        AS231 = m.AScalcuuu(f, p, k2, k3, k1, N0);
        
        Hi=m.H(f,p);
        
        a=exp(N0);
        k3s = k1*k1*k1 * k2*k2*k2 * k3*k3*k3;
        ks = k1 + k2 + k3;
        K2 = k1*k2 + k1*k3 + k2*k3;
        
        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
            pff[i+j*nF+k*nF*nF] = - 1./(a*a*a)/4./k3s * Hi * (-k1*k1*(k2+k3)/ks* k1*k2*k3) * (-C123[i+nF*j+nF*nF*k]*k1*k2 - C132[i+nF*k+nF*nF*j]*k1*k3 - C231[j+nF*k+nF*nF*i]*k2*k3
                                                                                              + a*a*AS123[i+nF*j+nF*nF*k] + a*a*AS132[i+nF*k+nF*nF*j] + a*a*AS231[j+nF*k+nF*nF*i]
                                                                                              );
			pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - FMi[((2*nF)*(j)+k)]*1./(a*a*a)/4./k3s * Hi * (-k1*k1*(k2+k3)/ks* k1*k2*k3) * f[nF+i]/2./Hi*(-k2*k2-k3*k3+k1*k1)/2.;
            pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+k)]*1./(a*a*a)/4./k3s * Hi * (-k1*k1*(k2+k3)/ks* k1*k2*k3) * f[nF+j]/2./Hi*(-k1*k1-k3*k3+k2*k2)/2.;
            pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+j)]*1./(a*a*a)/4./k3s * Hi * (-k1*k1*(k2+k3)/ks* k1*k2*k3) * f[nF+k]/2./Hi*(-k1*k1-k2*k2+k3*k3)/2.;
		}}}
            
        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
            pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - 1./(a*a*a)/4./k3s * Hi * (-k1*k1*k2*k3/ks) * (C123[i+j*nF+k*nF*nF]*k1*k1*k2*k2*(1.+k3/ks)
                                                                                                      + C132[i+k*nF+j*nF*nF]*k1*k1*k3*k3*(1.+k2/ks)
                                                                                                      + C231[j+k*nF+i*nF*nF]*k3*k3*k2*k2*(1.+k1/ks)
                                                                                                      - a*a*AS123[i+nF*j+nF*nF*k]*(K2 - k1*k2*k3/ks)
                                                                                                      - a*a*AS132[i+nF*k+nF*nF*j]*(K2 - k1*k2*k3/ks)
                                                                                                      - a*a*AS231[j+nF*k+nF*nF*i]*(K2 - k1*k2*k3/ks)
                                                                                                      );
            pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - 1./(a*a*a)/4./k3s * Hi * (-k1*k1*k2*k3/ks) * (B123[i+j*nF+k*nF*nF]/Hi*k1*k2*k3*k3
                                                                                                      + B132[i+k*nF+j*nF*nF]/Hi*k1*k3*k2*k2
                                                                                                      + B231[j+k*nF+i*nF*nF]/Hi*k2*k3*k1*k1);
			pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - FMi[((2*nF)*(j)+k)]*1./(a*a*a)/4./k3s * Hi * (-k1*k1*k2*k3/ks) * f[nF+i]/2./Hi*(-1.)*(-k2*k2-k3*k3+k1*k1)/2.*(K2 +k1*k2*k3/ks);
            pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+k)]*1./(a*a*a)/4./k3s * Hi * (-k1*k1*k2*k3/ks) * f[nF+j]/2./Hi*(-1.)*(-k1*k1-k3*k3+k2*k2)/2.*(K2 +k1*k2*k3/ks);
            pff[i+j*nF+k*nF*nF] = pff[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+j)]*1./(a*a*a)/4./k3s * Hi * (-k1*k1*k2*k3/ks) * f[nF+k]/2./Hi*(-1.)*(-k1*k1-k2*k2+k3*k3)/2.*(K2 +k1*k2*k3/ks);
        }}}
        return pff;
    }
    
    vector<double> ppfCalc(vector<double> f,vector<double> p,  double k1, double k2, double k3,double N0)
    {
        double a, H, ks, k3s;
        vector<double> ppf(nF*nF*nF);
        vector<double> C123, C132, C231, B123, B132, B231, AS123, AS132, AS231;
        model m;
        fieldmetric fmet;

        vector<double> FMi;
		FMi = fmet.fmetric(f,p);
        
        C123 = m.Ccalcuuu(f,p, k1, k2, k3, N0);
        C132 = m.Ccalcuuu(f,p, k1, k3, k2, N0);
        C231 = m.Ccalcuuu(f,p, k2, k3, k1, N0);
        
        B123 = m.Bcalcuuu(f, p, k1, k2, k3, N0);
        B132 = m.Bcalcuuu(f, p, k1, k3, k2, N0);
        B231 = m.Bcalcuuu(f, p, k2, k3, k1, N0);
        
        AS123 = m.AScalcuuu(f, p, k1, k2, k3, N0);
        AS132 = m.AScalcuuu(f, p, k1, k3, k2, N0);
        AS231 = m.AScalcuuu(f, p, k2, k3, k1, N0);
        
        
        H=m.H(f,p);
        
        a=exp(N0);
        k3s = k1*k1*k1 * k2*k2*k2 * k3*k3*k3;
        ks = k1 + k2 + k3;
        
        
        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
            ppf[i+j*nF+k*nF*nF] = -1./(a*a*a*a)/4./k3s * (k1*k2*k3)*(k1*k2*k3)/ks*k1*k2*(-C123[i+j*nF+k*nF*nF]*k1*k2 - C132[i+k*nF+j*nF*nF]*k1*k3 - C231[j+k*nF+i*nF*nF]*k2*k3
                                                                                         + a*a*AS123[i+nF*j+nF*nF*k] + a*a*AS132[i+nF*k+nF*nF*j] + a*a*AS231[j+nF*k+nF*nF*i]
                                                                                         + a*a*H*B123[i+nF*j+nF*nF*k]*(k1+k2)*k3/k1/k2
                                                                                         + a*a*H*B132[i+nF*k+nF*nF*j]*(k1+k3)*k2/k1/k3
                                                                                         + a*a*H*B231[j+nF*k+nF*nF*i]*(k2+k3)*k1/k2/k3
                                                                                         + k1*k1*k2*k2*a*a*H*B123[i+nF*j+nF*nF*k]*k1*k2*k3*k3
                                                                                         + k1*k1*k2*k2*a*a*H*B132[i+nF*k+nF*nF*j]*k1*k3*k2*k2
                                                                                         + k1*k1*k2*k2*a*a*H*B231[j+nF*k+nF*nF*i]*k2*k3*k1*k1);
            ppf[i+j*nF+k*nF*nF] = ppf[i+j*nF+k*nF*nF] - FMi[((2*nF)*(j)+k)]*1./(a*a*a*a)/4/k3s * (k1*k2*k3)*(k1*k2*k3)/ks*k1*k2*f[nF+i]/2./H*(-k2*k2-k3*k3+k1*k1)/2.;
            ppf[i+j*nF+k*nF*nF] = ppf[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+k)]*1./(a*a*a*a)/4/k3s * (k1*k2*k3)*(k1*k2*k3)/ks*k1*k2*f[nF+j]/2./H*(-k1*k1-k3*k3+k2*k2)/2.;
            ppf[i+j*nF+k*nF*nF] = ppf[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+j)]*1./(a*a*a*a)/4/k3s * (k1*k2*k3)*(k1*k2*k3)/ks*k1*k2*f[nF+k]/2./H*(-k1*k1-k2*k2+k3*k3)/2.;
            
        }}}
        return ppf;
    }
    
    vector<double> pppCalc(vector<double> f,vector<double> p,  double k1, double k2, double k3,double N0)
    {
        
        double a, H, ks, k3s, K2;
        vector<double> ppp(nF*nF*nF);
        vector<double> C123, C132, C231, B123, B132, B231, AS123, AS132, AS231;
        model m;
        fieldmetric fmet;

        vector<double> FMi;
		FMi = fmet.fmetric(f,p);
        C123 = m.Ccalcuuu(f,p, k1, k2, k3, N0);
        C132 = m.Ccalcuuu(f,p, k1, k3, k2, N0);
        C231 = m.Ccalcuuu(f,p, k2, k3, k1, N0);
        
        B123 = m.Bcalcuuu(f,p, k1, k2, k3, N0);
        B132 = m.Bcalcuuu(f,p, k1, k3, k2, N0);
        B231 = m.Bcalcuuu(f,p, k2, k3, k1, N0);
        
		AS123 = m.AScalcuuu(f, p, k1, k2, k3, N0);
        AS132 = m.AScalcuuu(f, p, k1, k3, k2, N0);
        AS231 = m.AScalcuuu(f, p, k2, k3, k1, N0);
        
        H=m.H(f,p);
        a=exp(N0);
        k3s = k1*k1*k1 * k2*k2*k2 * k3*k3*k3;
        ks = k1 + k2 + k3;
        K2 = k1*k2 + k1*k3 + k2*k3;
        
        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
            ppp[i+j*nF+k*nF*nF] = - 1./(a*a*a)/4./k3s * H * (k1*k1*k2*k2*k3*k3)/ks  * (C123[i+j*nF+k*nF*nF]*k1*k1*k2*k2*(1.+k3/ks)
                                                                                       + C132[i+k*nF+j*nF*nF]*k1*k1*k3*k3*(1.+k2/ks)
                                                                                       + C231[j+k*nF+i*nF*nF ]*k3*k3*k2*k2*(1.+k1/ks)
                                                                                       - a*a*AS123[i+nF*j+nF*nF*k]*(K2 - k1*k2*k3/ks)
                                                                                       - a*a*AS132[i+nF*k+nF*nF*j]*(K2 - k1*k2*k3/ks)
                                                                                       - a*a*AS231[j+nF*k+nF*nF*i]*(K2 - k1*k2*k3/ks)
                                                                                       );
            ppp[i+j*nF+k*nF*nF ] = ppp[i+j*nF+k*nF*nF] - 1./(a*a*a)/4./k3s * H * (k1*k1*k2*k2*k3*k3)/ks * (B123[i+j*nF+k*nF*nF]/H*k1*k2*k3*k3
                                                                                                           + B132[i+k*nF+j*nF*nF]/H*k1*k3*k2*k2
                                                                                                           + B231[j+k*nF+i*nF*nF]/H*k2*k3*k1*k1);
            ppp[i+j*nF+k*nF*nF] = ppp[i+j*nF+k*nF*nF] - FMi[((2*nF)*(j)+k)]*1./(a*a*a)/4./k3s * H * (k1*k1*k2*k2*k3*k3)/ks * f[nF+i]/2./H*(-1.)*(-k2*k2-k3*k3+k1*k1)/2.*(K2 +k1*k2*k3/ks);
            ppp[i+j*nF+k*nF*nF] = ppp[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+k)]*1./(a*a*a)/4./k3s * H * (k1*k1*k2*k2*k3*k3)/ks * f[nF+j]/2./H*(-1.)*(-k1*k1-k3*k3+k2*k2)/2.*(K2 +k1*k2*k3/ks);
            ppp[i+j*nF+k*nF*nF] = ppp[i+j*nF+k*nF*nF] - FMi[((2*nF)*(i)+j)]*1./(a*a*a)/4./k3s * H * (k1*k1*k2*k2*k3*k3)/ks * f[nF+k]/2./H*(-1.)*(-k1*k1-k2*k2+k3*k3)/2.*(K2 +k1*k2*k3/ks);
            
        }}}
        return ppp;
    }
	/**
     *  Alpha accessor methods. 
     *  It takes 0 arguments, and returns the vector alp.
     *  It takes 3 integer, it returns the (i,j,k) element of alp.
     */
	vector<double> getA()
	{
		return alp;
	}
	
	double getA(int i, int j, int k)
	{
        return alp[i+j*2*nF + k*2*2*nF*nF];
	}
	/**
    *  Alpha modifier methods. 
    *  It takes two integer (i,j,k) and a double (value), sets alp(i,j) = value
    *  It takes 1 vector (value), it sets alp = value.
    */
	void setA(int i, int j, int k, double value)
	{
		alp[i+j*2*nF + k*2*2*nF*nF]=value;
	}
	void setA(vector<double> value)
	{
		alp=value;
	}
	//print sig to screen 
    void printA()
	{
		for(int i=0;i<2*nF;i++){for(int j=0;j<2*nF;j++){for(int k=0;k<2*nF;k++) {std::cout << alp[i+j*2*nF+k*2*2*nF*nF] << '\t';}}}
		std::cout << std::endl;
	}
};
#endif 

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef Rho_1  // Prevents the class being re-definFd
#define Rho_1 
#include "model.h"
#include <iostream>
#include <math.h>
using namespace std;
/**
 * Instance of the first MPP matrix.
 */
class Rho1 
{	
private:
	int nF;               // Number of fields
	vector<double> rho;   // Vector storing elements of Rho_ij(N0,N0)
	double N0;            // Initial time 

public:
	//constructor for rho at initial time (it's a delta function)
	Rho1(int nFi, double k1, double N0, vector<double> f, vector<double> p)
	{
		nF = nFi;  
        N0  = N0;
        rho.resize(2*nF*2*nF);
        int count = 0 ; 
        for (int i = 0; i < 2*nF*2*nF; i++)
        {
            count = i % (2*nF+1);
            if (count == 0){
                rho[i] = 1. ;
            } else {
                rho[i] = 0. ;
            }
        }
    }
	//Rho1 accessors
    vector<double>  getR()
	{
		return rho;
	}
	
    double getR(int i, int j)
	{
		return rho[i+j*2*nF];
	}
	//Rho1 modifiers
	void setR(int i, int j, double value)
	{
		rho[i + 2*nF*j] = value;
	}
	
    void setR(vector<double>value)
	{
		rho = value;
	}
    //print Rho1 to screen
	void printR()
	{
		for(int i=0;i<nF;i++){for(int j=0;j<nF;j++) {std::cout << rho[nF*i + j] << '\t';}}
		std::cout << std::endl;
	}
};
#endif 

//-------------------------------------------------------------------------------------------------------------------------------------------

#ifndef Rho_2  // Prevents the class being re-definFd
#define Rho_2 
#include "model.h"
#include <iostream>
#include <math.h>
using namespace std;
/**
 * Instance of the second MPP matrix.
 */
class Rho2 
{	
private:
	int nF;               // Number of fields
	vector<double> rho;   // Vector storing elements of Rho_abc(N0,N0)
	double N0;            // Initial time 

public:
	//constructor for rho at initial time (it's a delta function)
	Rho2(int nFi, double k1, double k2, double k3, double N0, vector<double> f, vector<double> p)
	{
		nF = nFi;  
        N0  = N0;
        rho.resize(2*nF*2*nF*2*nF);
        for (int i = 0; i < 2*nF*2*nF*2*nF; i++)
        {
            rho[i] = 0;
        }
    }
	//Rho2 accessors
    vector<double>  getR()
	{
		return rho;
	}
	
    double getR(int i, int j, int k)
	{
		return rho[i + 2*nF*j + 2*nF*2*nF*k];
	}

	//Rho2 modifiers
	void setR(int i, int j, int k, double value)
	{
		rho[i + 2*nF*j + 2*nF*2*nF*k]=value;
	}
	
    void setR(vector<double> value)
	{
		rho=value;
	}
	
    //print Rho2 to screen
	void printR()
	{
		for(int i=0;i<nF;i++){for(int j=0;j<nF;j++) {for(int k =0; i<2*nF; k++){std::cout << rho[nF*i + j] << '\t';}}}
		std::cout << std::endl;
	}
};
#endif 