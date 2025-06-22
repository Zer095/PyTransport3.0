/**
 * @file model.h
 * @brief Defines the base model interface and related functions for inflationary cosmology.
 *
 * This header file contains the declaration of the `model` class, which encapsulates
 * the defining features of an inflationary model, including the flow tensors (u1, u2, u3)
 * and the action-related tensors (A, B, C). It also provides methods for calculating
 * cosmological parameters like the Hubble rate (H), slow-roll parameters (epsilon, eta),
 * and gauge transformation tensors (N1, N2).
 *
 * The class interacts with the `potential` class to retrieve potential values and
 * their derivatives.
 *
 * @copyright GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * @see http://www.gnu.org/licenses/
 */

#ifndef MODEL_H  // Prevents the class being re-defined
#define MODEL_H

#include "potential.h"
#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;

/**
 * @class model
 * @brief Represents an inflationary model, encapsulating its key physical properties and calculations.
 *
 * This class provides methods to compute various cosmological quantities relevant
 * to inflationary models, such as the Hubble parameter, slow-roll parameters,
 * and the flow tensors (u1, u2, u3) and action-related tensors (A, B, C)
 * required by the Transport formalism. It relies on a `potential` object
 * to define the scalar field potential and its derivatives.
 */
class model
{
private:
	int nF;  //!< @brief Number of fields in the model.
	int nP;  //!< @brief Number of parameters defining the potential.
	potential pot; //!< @brief Instance of the potential class, defining the model's potential.

public:
	/**
	 * @brief Constructor for the model class.
	 *
	 * Initializes the model by setting the number of fields (`nF`) and
	 * parameters (`nP`) based on the default `potential` object.
	 */
	model()
	{
	    potential pot_temp; // Use a temporary object to get nP and nF
        nP = pot_temp.getnP();
        nF = pot_temp.getnF();
        // The member 'pot' is automatically default-constructed here,
        // so no need to explicitly initialize it in the constructor body
        // unless you want to pass specific arguments to its constructor.
    }

    /**
     * @brief Returns the number of fields (`nF`) in the current model.
     * @return An integer representing the number of fields.
     */
    int getnF()
    {
        return nF;
    }

    /**
     * @brief Returns the number of parameters (`nP`) for the current model's potential.
     * @return An integer representing the number of parameters.
     */
    int getnP()
    {
        return nP;
    }

    /**
     * @brief Calculates the Hubble expansion rate H.
     *
     * Computes the Hubble parameter $H$ given the field values and model parameters.
     * The Hubble rate is calculated from the kinetic and potential energy of the fields.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @return The value of the Hubble rate, $H$.
     */
	double H(vector<double> f, vector<double> p)
	{
		double Hi2=0;
		double Vi;
		Vi=pot.V(f,p);
		for(int i=0; i<nF; i++)
		{
			Hi2 = Hi2 + 1./3.*(f[nF+i]*f[nF+i]/2.); // Contribution from kinetic energy
		}
		Hi2=Hi2 + 1./3.*Vi; // Contribution from potential energy
        return sqrt(Hi2);
	}

    /**
     * @brief Calculates the time derivative of the Hubble rate, $\dot{H}$.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @return The value of $\dot{H}$.
     */
    double Hdot(vector<double> f)
	{
        double sum=0.;
		for(int i=0; i<nF; i++)
		{
			sum= sum - 1./2.*(f[nF+i]*f[nF+i]); // Sum over kinetic energy terms
		}
		return sum;
	}

    /**
     * @brief Calculates the second time derivative of the scale factor, $\ddot{a}$.
     *
     * This function returns $\ddot{a}$ (normalized by $a$) which is related to the
     * acceleration of the universe's expansion.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @return The value of $\ddot{a}/a$.
     */
    double addot(vector<double> f, vector<double> p)
    {
        double sum=0.;
        double Vi;
        Vi=pot.V(f,p);
        for(int i=0; i<nF; i++)
        {
            sum= sum - 1./2.*(f[nF+i]*f[nF+i]); // Sum over kinetic energy terms
        }
        return -1./3.*(sum+Vi); // Related to acceleration equation
    }

    /**
     * @brief Calculates the first slow-roll parameter, epsilon ($\epsilon$).
     *
     * $\epsilon = -\dot{H}/H^2 = \frac{1}{2} \sum_i \dot{\phi}_i^2 / H^2 $.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @return The value of the first slow-roll parameter, $\epsilon$.
     */
	double Ep(vector<double> f,vector<double> p)
	{
		double Hi = H(f,p);
		double mdotH=0.;
		for(int i=0; i<nF; i++)
		{
			mdotH= mdotH + 1./2.*(f[nF+i]*f[nF+i]); // Sum of kinetic energy densities
		}
  		return mdotH/(Hi*Hi);
	}

    /**
     * @brief Calculates the second slow-roll parameter, eta ($\eta$).
     *
     * $\eta = V''/V$. More generally, in multi-field, this captures curvature
     * of the potential along the field trajectory.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @return The value of the second slow-roll parameter, $\eta$.
     */
	double Eta(vector<double> f,vector<double> p)
	{
		double V = pot.V(f,p);
        vector<double> Vp = pot.dV(f,p); // First derivative of the potential
        double Et = 0;
        double Hi = H(f,p);
        for(int i = 0; i < nF; i++){
            // This is a specific definition of Eta relevant to the Transport formalism or particular models.
            // It might differ from standard single-field definitions.
            Et += (3*Hi*f[nF+i] + Vp[i])/ f[nF+i];
        }

  		return Et/Hi;
	}

    /**
     * @brief Calculates a scaling factor for the field perturbation $\delta\dot{\phi}$.
     *
     * This function returns a specific scaling factor often used in the context
     * of cosmological perturbations or transfer functions to improve numerical stability
     * or normalize quantities.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param N Value of the e-fold time.
     * @return The calculated scaling factor.
     */
    double scale(vector<double> f, vector<double> p,  double N)
    {
        double k = 1.0; // Reference wavenumber, typically set to 1 for dimensionless calculations
        double a = exp(N); // Scale factor 'a'
        double Hi = H(f,p); // Hubble rate
        return  a/(1.+a*Hi/k)/Hi;
    }

    /**
     * @brief Calculates the time derivative of the `scale` function, normalized by the `scale` function itself.
     *
     * This function returns $ \dot{(\text{scale})} / \text{scale} $. This quantity
     * is often used in the evolution equations of cosmological perturbations.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param N Value of the e-fold time.
     * @return The scaled derivative of the scale function.
     */
    double dscale(vector<double> f, vector<double> p, double N)
    {
        double k = 1.0; // Reference wavenumber
        double a = exp(N); // Scale factor 'a'
        double Hi = H(f,p); // Hubble rate
        double Hdi = Hdot(f); // Derivative of Hubble rate
        return  -Hdi/Hi/Hi*a/(1.+a*Hi/k) + a/(1.+a*Hi/k) -a*(a*Hi*Hi/k + a*Hdi/k)/(1.+a*Hi/k)/(1.+a*Hi/k)/Hi;
    }

    /**
     * @brief Calculates the first flow tensor $u_i^{(1)}$.
     *
     * This tensor represents the background trajectory's velocity in field space,
     * normalized by the Hubble rate.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @return A vector containing the components of the first flow tensor, $u_i^{(1)}$.
     */
	vector<double> u(vector<double> f,vector<double> p)
	{
		vector<double> u1out(2*nF);
		vector<double> dVi; // First derivative of potential
		double Hi;
		Hi=H(f,p);
		// First nF-elements of u1 (related to field velocities)
		for(int i=0; i<nF; i++)
		{
			u1out[i]  = f[nF+i]/Hi;
		}

		dVi=pot.dV(f,p); // Calculate dV
        // Second nF-elements of u1 (related to field accelerations)
		for(int i=0; i<nF; i++)
		{
			u1out[nF+i]  = -3.*Hi*f[nF+i]/Hi-dVi[i]/Hi;
		}
		return u1out;
	}

	/**
	 * @brief Calculates the second flow tensor $u_{ij}^{(2)}$.
	 *
	 * This tensor describes the second-order deviations from the background trajectory
	 * in field space, relevant for 2-point correlation functions.
	 *
	 * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
	 * @param p A vector containing the model parameters.
	 * @param k1 The value of the wavenumber.
	 * @param N Value of the e-fold time.
	 * @return A vector representing the flattened 2D tensor $u_{ij}^{(2)}$ (size $2nF \times 2nF$).
	 */
	vector<double> u(vector<double> f,vector<double> p, double k1, double N)
	{
		vector<double> u2out(2*nF*2*nF); // Size (2*nF) x (2*nF)
        double a = exp(N); // Scale factor
		double ep = Ep(f,p); // Epsilon slow-roll parameter
		double Hi=H(f,p); // Hubble rate
        double s=scale(f,p,N); // Scaling factor
        double ds=dscale(f,p,N); // Derivative of scaling factor
		vector<double> dVVi; dVVi = pot.dVV(f,p); // Second derivative of potential
		vector<double> dVi; dVi =  pot.dV(f,p); // First derivative of potential

		for(int i = 0; i<nF; i++)
            {for(int j = 0; j<nF; j++){
                // Off-diagonal terms related to field-velocity mixing
                u2out[i+ j*2*nF]=0.;
                u2out[i+(j+nF)*2*nF]=0.;
                // Terms involving potential derivatives and field velocities
                u2out[i+nF+(j)*2*nF]=(-dVVi[i + nF*j] + (-3.+ep)*f[nF+i]*f[nF+j] + 1./Hi*(-dVi[i])*f[nF+j] + 1./Hi*f[nF+i]*(-dVi[j]) )/Hi *s;
                u2out[i+nF+(j+nF)*2*nF]=0.;
                // Diagonal terms modified for specific components
                if(i==j){
                    u2out[i+nF+(j)*2*nF]=u2out[i+nF+(j)*2*nF]-1.0*(k1*k1)/(a*a)/Hi  * s ;
                    u2out[i+(j+nF)*2*nF]=u2out[i+(j+nF)*2*nF] + 1./Hi/s;
                    u2out[i+nF+(j+nF)*2*nF]= u2out[i+nF+(j+nF)*2*nF] - 3.0*Hi/Hi  + ds/s/Hi;
                }
            }
        }
		return u2out;
	}

    /**
     * @brief Calculates the $w$ tensor for tensor perturbations.
     *
     * This tensor describes the evolution of tensor (gravitational wave) perturbations.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param k1 The value of the wavenumber.
     * @param N Value of the e-fold time.
     * @return A vector representing the flattened 2D tensor $w$ (size $2 \times 2$ for single tensor mode).
     */
	vector<double> w(vector<double> f,vector<double> p, double k1, double N)
	{
		int nT = 1; // Number of tensor modes, typically 1 for standard models
		double a = exp(N); // Scale factor
		vector<double> w2out(2*nT*2*nT); // Size (2*nT) x (2*nT)
		double Hi=H(f,p); // Hubble rate
        double s=scale(f,p,N); // Scaling factor
        double ds=dscale(f,p,N); // Derivative of scaling factor
		vector<double> dVVi;
		dVVi = pot.dVV(f,p); // Second derivative of potential
		vector<double> dVi;
		dVi =  pot.dV(f,p); // First derivative of potential

        w2out[0+ 0*2*nT]=0.; // (0,0) component
        w2out[0+1*2*nT]=+ 1./Hi; // (0,1) component
		w2out[1+(0)*2*nT] = -1.0*(k1*k1)/(a*a)/Hi; // (1,0) component (related to momentum and wavenumber)
		w2out[1+(1)*2*nT]= - 3.0*Hi/Hi ; // (1,1) component (related to damping)

		return w2out;
	}

    /**
     * @brief Calculates the A-tensor, a three-index coupling term in the action.
     *
     * This tensor represents the cubic coupling of three scalar fields in the action,
     * relevant for 3-point correlation functions.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param k1 First wavenumber.
     * @param k2 Second wavenumber.
     * @param k3 Third wavenumber.
     * @param N Value of the e-fold time.
     * @return A vector representing the flattened 3D tensor $A_{ijk}$ (size $nF \times nF \times nF$).
     */
    vector<double> Acalc(vector<double> f, vector<double> p, double k1, double k2, double k3,double N)
	{
		double a = exp(N); // Scale factor
        double Vi=pot.V(f,p); // Potential value
		double Hi=H(f,p); // Hubble rate

        vector<double> dVVi;
		dVVi=pot.dVV(f,p); // Second derivative of potential
		vector<double> dVi;
		dVi =  pot.dV(f,p); // First derivative of potential
		vector<double> dVVVi;
		dVVVi=pot.dVVV(f,p); // Third derivative of potential
        vector<double> Xi(nF); // Auxiliary vector
        vector<double> A(nF*nF*nF); // Resultant A tensor (flattened)

        double sum1=0;
		for(int i=0;i<nF;i++){sum1=sum1+f[nF+i]*f[nF+i];} // Sum of squared velocities
		for(int i=0;i<nF;i++){Xi[i] = 2.*(-dVi[i]-3.*Hi*f[nF+i])+f[nF+i]/Hi*sum1;} // Populate Xi

		for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
			A[i + j*nF +k* nF*nF] = -1./3. * dVVVi[i + j*nF +k* nF*nF] // Terms from third derivative of potential
			- 1./3.*f[nF + i]/2./Hi* dVVi[j + k*nF]
            - 1./3.*f[nF + j]/2./Hi* dVVi[i + k*nF]
            - 1./3.*f[nF + k]/2./Hi* dVVi[i + j*nF] // Terms from second derivative of potential
			+ 1./3.*f[nF + i] * f[nF + j]/8./Hi/Hi * Xi[k]
            + 1./3.*f[nF + i] * f[nF + k]/8./Hi/Hi * Xi[j]
            + 1./3.*f[nF + k] * f[nF + j]/8./Hi/Hi * Xi[i] // Terms from Xi and velocities
			+ 1./3.*f[nF + i]/32./Hi/Hi/Hi * Xi[j] *Xi[k]
            + 1./3.*f[nF + j]/32./Hi/Hi/Hi * Xi[i] *Xi[k]
            + 1./3.*f[nF + k]/32./Hi/Hi/Hi * Xi[i] *Xi[j]
			+ 1.*f[nF + i]*f[nF + j]*f[nF + k]/8./Hi/Hi/Hi*2.*Vi // Terms involving field products and potential
			- 1./3.*f[nF + i]/32./Hi/Hi/Hi * Xi[j] * Xi[k] * (k2*k2+k3*k3 - k1*k1)*(k2*k2+k3*k3 - k1*k1)/k2/k2/k3/k3/4.
            - 1./3.*f[nF + j]/32./Hi/Hi/Hi * Xi[i] * Xi[k] * (k1*k1+k3*k3 - k2*k2)*(k1*k1+k3*k3 - k2*k2)/k1/k1/k3/k3/4.
            - 1./3.*f[nF + k]/32./Hi/Hi/Hi * Xi[i] * Xi[j] * (k1*k1+k2*k2 - k3*k3)*(k1*k1+k2*k2 - k3*k3)/k1/k1/k2/k2/4.; // Wavenumber-dependent terms
    		if(j==k){A[i + j*nF +k* nF*nF] = A[i + j*nF +k* nF*nF] + 1./3.*f[nF+i]/2./Hi*(-k2*k2-k3*k3+k1*k1)/a/a/2.;}
			if(i==k){A[i + j*nF +k* nF*nF] = A[i + j*nF +k* nF*nF] + 1./3.*f[nF+j]/2./Hi*(-k1*k1-k3*k3+k2*k2)/a/a/2.;}
			if(i==j){A[i + j*nF +k* nF*nF] = A[i + j*nF +k* nF*nF] + 1./3.*f[nF+k]/2./Hi*(-k2*k2-k1*k1+k3*k3)/a/a/2.;}
            }}}

        return A;
    }

    /**
     * @brief Calculates the "slow" part of the A-tensor, used for initial conditions.
     *
     * This function computes a specific component of the A-tensor that is only relevant
     * for setting up initial conditions for 3-point correlation functions. It has
     * similarities to `Acalc` but might omit terms that rapidly oscillate.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param k1 First wavenumber.
     * @param k2 Second wavenumber.
     * @param k3 Third wavenumber.
     * @param N Value of the e-fold time.
     * @return A vector representing the flattened 3D tensor $AS_{ijk}$ (size $nF \times nF \times nF$).
     */
    vector<double> AScalc(vector<double> f, vector<double> p, double k1, double k2, double k3,double N)
    {
        double Vi=pot.V(f,p); // Potential value
        double Hi=H(f,p); // Hubble rate

        vector<double> dVVi;
        dVVi=pot.dVV(f,p); // Second derivative of potential
        vector<double> dVi;
        dVi =  pot.dV(f,p); // First derivative of potential
        vector<double> dVVVi;
        dVVVi=pot.dVVV(f,p); // Third derivative of potential
        vector<double> Xi(nF); // Auxiliary vector
        vector<double> AS(nF*nF*nF); // Resultant AS tensor (flattened)

        double sum1=0;
        for(int i=0;i<nF;i++){sum1=sum1+f[nF+i]*f[nF+i];} // Sum of squared velocities
        for(int i=0;i<nF;i++){Xi[i] = 2.*(-dVi[i]-3.*Hi*f[nF+i])+f[nF+i]/Hi*sum1;} // Populate Xi

        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
            AS[i + j*nF +k* nF*nF] = -1./3. * dVVVi[i + j*nF +k* nF*nF]
            - 1./3.*f[nF + i]/2./Hi* dVVi[j + k*nF]
            - 1./3.*f[nF + j]/2./Hi* dVVi[i + k*nF]
            - 1./3.*f[nF + k]/2./Hi* dVVi[i + j*nF]
            + 1./3.*f[nF + i] * f[nF + j]/8./Hi/Hi * Xi[k]
            + 1./3.*f[nF + i] * f[nF + k]/8./Hi/Hi * Xi[j]
            + 1./3.*f[nF + k] * f[nF + j]/8./Hi/Hi * Xi[i]
            + 1./3.*f[nF + i]/32./Hi/Hi/Hi * Xi[j] *Xi[k]
            + 1./3.*f[nF + j]/32./Hi/Hi/Hi * Xi[i] *Xi[k]
            + 1./3.*f[nF + k]/32./Hi/Hi/Hi * Xi[i] *Xi[j]
            + 1.*f[nF + i]*f[nF + j]*f[nF + k]/8./Hi/Hi/Hi*2.*Vi
            - 1./3.*f[nF + i]/32./Hi/Hi/Hi * Xi[j] * Xi[k] * (k2*k2+k3*k3 - k1*k1)*(k2*k2+k3*k3 - k1*k1)/k2/k2/k3/k3/4.
            - 1./3.*f[nF + j]/32./Hi/Hi/Hi * Xi[i] * Xi[k] * (k1*k1+k3*k3 - k2*k2)*(k1*k1+k3*k3 - k2*k2)/k1/k1/k3/k3/4.
            - 1./3.*f[nF + k]/32./Hi/Hi/Hi * Xi[i] * Xi[j] * (k1*k1+k2*k2 - k3*k3)*(k1*k1+k2*k2 - k3*k3)/k1/k1/k2/k2/4.;
        }}}

        return AS;
    }


    /**
     * @brief Calculates the B-tensor, a three-index coupling term in the action.
     *
     * This tensor represents the cubic coupling of two scalar fields and one momentum
     * in the action, relevant for 3-point correlation functions.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param k1 First wavenumber.
     * @param k2 Second wavenumber.
     * @param k3 Third wavenumber.
     * @param N Value of the e-fold time.
     * @return A vector representing the flattened 3D tensor $B_{ijk}$ (size $nF \times nF \times nF$).
     */
   vector<double> Bcalc(vector<double> f,vector<double> p, double k1, double k2, double k3,double N)
	{
        double Hi=H(f,p); // Hubble rate

		vector<double> dVVi;
		dVVi=pot.dVV(f,p); // Second derivative of potential
		vector<double> dVi;
		dVi =  pot.dV(f,p); // First derivative of potential
		vector<double> dVVVi;
		dVVVi=pot.dVVV(f,p); // Third derivative of potential
        vector<double> Xi(nF); // Auxiliary vector
        vector<double> B(nF*nF*nF); // Resultant B tensor (flattened)

        double sum1=0;
		for(int i=0;i<nF;i++){sum1=sum1+f[nF+i]*f[nF+i];} // Sum of squared velocities
		for(int i=0;i<nF;i++){Xi[i] = 2.0*(-dVi[i]-3.0*Hi*f[nF+i])+f[nF+i]/Hi*sum1;} // Populate Xi


        for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
			B[i + j*nF +k* nF*nF] = 1.*f[nF + i]*f[nF+j]*f[nF+k]/4./Hi/Hi // Terms with product of velocities
			- 1./2.*f[nF + i] * f[nF + k]/8./Hi/Hi/Hi * Xi[j]
            - 1./2.*f[nF + j] * f[nF + k]/8./Hi/Hi/Hi * Xi[i] // Terms with Xi and velocities
			+ 1./2.*f[nF + i] * f[nF + k]/8./Hi/Hi/Hi * Xi[j]*(k2*k2+k3*k3 - k1*k1)*(k2*k2+k3*k3 - k1*k1)/k2/k2/k3/k3/4.
            + 1./2.*f[nF + j] * f[nF + k]/8./Hi/Hi/Hi * Xi[i]*(k1*k1+k3*k3 - k2*k2)*(k1*k1+k3*k3 - k2*k2)/k1/k1/k3/k3/4.; // Wavenumber-dependent terms
			if(j==k){B[i + j*nF +k* nF*nF] = B[i + j*nF +k* nF*nF] - 1.*Xi[i]/4./Hi*(-k1*k1-k2*k2+k3*k3)/k1/k1/2.;}
			if(i==k){B[i + j*nF +k* nF*nF] = B[i + j*nF +k* nF*nF] - 1.*Xi[j]/4./Hi*(-k1*k1-k2*k2+k3*k3)/k2/k2/2.;}
		}}}
        return B;
    }

    /**
     * @brief Calculates the C-tensor, a three-index coupling term in the action.
     *
     * This tensor represents the cubic coupling of one scalar field and two momenta
     * in the action, relevant for 3-point correlation functions.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param k1 First wavenumber.
     * @param k2 Second wavenumber.
     * @param k3 Third wavenumber.
     * @param N Value of the e-fold time.
     * @return A vector representing the flattened 3D tensor $C_{ijk}$ (size $nF \times nF \times nF$).
     */
    vector<double> Ccalc(vector<double> f, vector<double> p, double k1, double k2, double k3,double N)
	{
		double Hi=H(f,p); // Hubble rate

     	vector<double> dVVi;
		dVVi=pot.dVV(f,p); // Second derivative of potential
		vector<double> dVi;
		dVi =  pot.dV(f,p); // First derivative of potential
		vector<double> dVVVi;
		dVVVi=pot.dVVV(f,p); // Third derivative of potential
        vector<double> Xi(nF); // Auxiliary vector
        vector<double> C(nF*nF*nF); // Resultant C tensor (flattened)

        double sum1=0;
		for(int i=0;i<nF;i++){sum1=sum1+f[nF+i]*f[nF+i];} // Sum of squared velocities
		for(int i=0;i<nF;i++){Xi[i] = 2.*(-dVi[i]-3.*Hi*f[nF+i])+f[nF+i]/Hi*sum1;} // Populate Xi

		for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
			C[i + j*nF +k* nF*nF] = 1.*f[nF + i]*f[nF+j]*f[nF+k]/8./Hi/Hi/Hi // Terms with product of velocities
			- 1.*f[nF + i] * f[nF+j] *f[nF+k]/8./Hi/Hi/Hi *(k1*k1+k2*k2 - k3*k3)*(k1*k1+k2*k2 - k3*k3)/k1/k1/k2/k2/4. ; // Wavenumber-dependent terms
			if(i==j){C[i + j*nF +k* nF*nF] = C[i + j*nF +k* nF*nF] - 1.*f[nF+k]/2./Hi;} // Diagonal terms
			if(j==k){C[i + j*nF +k* nF*nF] = C[i + j*nF +k* nF*nF] + f[nF+i]/2./Hi*(-k1*k1-k3*k3+k2*k2)/k1/k1/2.;}
			if(i==k){C[i + j*nF +k* nF*nF] = C[i + j*nF +k* nF*nF] + f[nF+j]/2./Hi*(-k2*k2-k3*k3+k1*k1)/k2/k2/2.;}
		}}}
        return C;
    }

	/**
	 * @brief Calculates the third flow tensor $u_{ijk}^{(3)}$.
	 *
	 * This tensor describes the third-order deviations from the background trajectory
	 * in field space, coupling three fields, and is crucial for 3-point correlation functions.
	 *
	 * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
	 * @param p A vector containing the model parameters.
	 * @param k1 First wavenumber.
	 * @param k2 Second wavenumber.
	 * @param k3 Third wavenumber.
	 * @param N Value of the e-fold time.
	 * @return A vector representing the flattened 3D tensor $u_{ijk}^{(3)}$ (size $2nF \times 2nF \times 2nF$).
	 */
	vector<double> u(vector<double> f, vector<double> p, double k1, double k2, double k3,double N)
	{
        vector<double>  A_tensor, B_tensor, B2_tensor, B3_tensor, C_tensor, C2_tensor, C3_tensor;
        double Hi;
		Hi=H(f,p); // Hubble rate
        double s=scale(f,p,N); // Scaling factor

        A_tensor = Acalc(f,p, k1, k2, k3 ,N);
        B_tensor = Bcalc(f,p, k2, k3, k1 ,N);
        B2_tensor =  Bcalc(f,p, k1, k2, k3 ,N);
        B3_tensor = Bcalc(f,p, k1, k3, k2 ,N);
        C_tensor =  Ccalc(f,p, k1, k2, k3 ,N);
        C2_tensor =  Ccalc(f,p, k1, k3, k2 ,N);
        C3_tensor = Ccalc(f,p, k3, k2, k1 ,N);

        vector<double> u3out(2*nF*2*nF*2*nF); // Resultant u3 tensor (flattened)

		for(int i=0;i<nF;i++){for(int j=0;j<nF;j++){for(int k=0;k<nF;k++){
			u3out[i+j*2*nF+k*2*nF*2*nF]= -B_tensor[j+k*nF+i*nF*nF]/Hi; // Term related to B tensor permutations

            u3out[(i)+(nF+j)*2*nF+k*2*nF*2*nF]= -C_tensor[i+j*nF+k*nF*nF]/Hi  /s;
            u3out[(i)+j*2*nF+(k+nF)*2*nF*2*nF]= -C2_tensor[i+k*nF+j*nF*nF]/Hi /s;

			u3out[(i)+(j+nF)*2*nF+(k+nF)*2*nF*2*nF]= 0.; // Zero component

            u3out[(nF+i) + j*2*nF + k*2*nF*2*nF]= 3.*A_tensor[i+j*nF+k*nF*nF]/Hi  *s;

			u3out[(nF+i)+(nF+j)*2*nF+k*2*nF*2*nF]=B3_tensor[i+k*nF+j*nF*nF]/Hi ;
			u3out[(nF+i)+(j)*2*nF+(k+nF)*2*nF*2*nF]=B2_tensor[i+j*nF+k*nF*nF]/Hi ;

            u3out[(nF+i)+(j+nF)*2*nF + (k+nF)*2*nF*2*nF]=C3_tensor[k+j*nF+i*nF*nF]/Hi  /s;

		}}}
        return u3out;
	}


    /**
     * @brief Calculates the first gauge transformation tensor $N_i^{(1)}$.
     *
     * This tensor transforms quantities from the field perturbation basis to the
     * curvature perturbation ($\zeta$) basis at first order.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param N Value of the e-fold time.
     * @return A vector containing the components of the first gauge tensor, $N_i^{(1)}$.
     */
    vector<double> N1(vector<double> f,vector<double> p, double N)
    {
        double Hd=Hdot(f); // Derivative of Hubble rate
        double Hi=H(f,p); // Hubble rate
        vector<double> dVi; // First derivative of potential
        vector<double> Ni(2*nF); // Resultant N1 tensor (flattened)
        dVi=pot.dV(f,p); // Calculate dV

        for(int i=0;i<nF;i++){
            Ni[i] = 1./2.*Hi/Hd * f[nF+i]; // Components related to field velocities

            Ni[nF+i] = 0. ; // Components related to field momenta (typically zero for N1)
        }

        return Ni;
    }

    /**
     * @brief Calculates the second gauge transformation tensor $N_{ij}^{(2)}$.
     *
     * This tensor transforms quantities from the field perturbation basis to the
     * curvature perturbation ($\zeta$) basis at second order. It is dependent on
     * the wavenumbers of the three-point function.
     *
     * @param f A vector containing the field values and their velocities ($ \phi_i, \dot{\phi}_i $).
     * @param p A vector containing the model parameters.
     * @param k1 First wavenumber.
     * @param k2 Second wavenumber.
     * @param k3 Third wavenumber.
     * @param N Value of the e-fold time.
     * @return A vector representing the flattened 2D tensor $N_{ij}^{(2)}$ (size $2nF \times 2nF$).
     */
    vector<double> N2(vector<double> f, vector<double> p, double k1, double k2, double k3, double N)
    {
        double Hd=Hdot(f); // Derivative of Hubble rate
        double Hin=H(f,p); // Hubble rate
        vector<double> dVi, dVVi;
        vector<double> Nii(2*nF*2*nF); // Resultant N2 tensor (flattened)
        double s = scale(f,p,N); // Scaling factor
        dVi=pot.dV(f,p); // First derivative of potential
        dVVi=pot.dVV(f,p); // Second derivative of potential

        double sum3 = 0.0;
        for(int i=0;i<nF;i++){sum3=sum3+dVi[i]*f[nF+i]/Hin/Hin/Hin;} // Auxiliary sum

        double ep = -Hd/Hin/Hin; // Epsilon slow-roll parameter
        for(int i=0;i<nF;i++){for(int j=0; j<nF; j++){
        Nii[i + (j) * 2*nF]= 2./ep/Hin/Hin/6. * (f[nF+i]*f[nF+j] *(-3./2. + 9./2./ep + 3./4.*sum3/ep/ep)); // Components for N_ij
        Nii[i + (j+nF) * 2*nF]=2./ep/Hin/Hin/6.*3./2.*f[i+nF]*f[j+nF]/Hin/ep  /s;
        Nii[i+nF + (j) * 2*nF]=2./ep/Hin/Hin/6.*3./2.*f[i+nF]*f[j+nF]/Hin/ep  /s;
        Nii[i+nF + (j+nF) * 2*nF]=0.; // Zero component
            if(i==j){Nii[i+nF+(j)*2*nF] = Nii[i+nF + (j) * 2*nF] - 2./ep/Hin/Hin/6. * 3./2.*Hin/k1/k1*((-k2*k2-k3*k3+k1*k1)/2. + k3*k3)  /s;
                    Nii[i+(j+nF)*2*nF] = Nii[i + (j+nF) * 2*nF] - 2./ep/Hin/Hin/6. * 3./2.*Hin/k1/k1*((-k2*k2-k3*k3+k1*k1)/2. + k2*k2)  /s;}
        }}

        return Nii;

    }
};
#endif
