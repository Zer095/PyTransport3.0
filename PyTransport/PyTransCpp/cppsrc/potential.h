/**
 * @file potential.h
 * @brief Defines the `potential` class for evaluating and managing scalar field potentials and their derivatives.
 *
 * This file declares the `potential` class, which is responsible for providing
 * the value of the scalar field potential and its first, second, and third
 * derivatives with respect to the fields. The actual mathematical expressions
 * for these functions are automatically generated and inserted by `PyTransSetup.py`
 * based on symbolic definitions.
 *
 * @copyright GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * @see http://www.gnu.org/licenses/
 */

#ifndef POTENTIAL_H
#define POTENTIAL_H

#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include "fieldmetric.h"

using namespace std;

/**
 * @class potential
 * @brief Manages the scalar field potential and its derivatives.
 *
 * This class provides an interface to query the potential value, its gradient,
 * Hessian (second derivatives), and third derivatives with respect to the fields.
 * The underlying mathematical expressions are generated dynamically from Python.
 */
class potential
{
private:
    int nF; //!< @brief Number of fields in the potential.
    int nP; //!< @brief Number of parameters for the potential.

public:
    /**
     * @brief Constructor for the potential class.
     *
     * Initializes the potential by retrieving the number of fields (`nF`) and
     * parameters (`nP`) from the automatically generated definitions.
     */
    potential()
    {
        // These are set by the preprocessor based on the Python setup
        nF = 0; // Default or placeholder, will be set by preprocessor
        nP = 0; // Default or placeholder, will be set by preprocessor
    }

    /**
     * @brief Returns the number of fields (`nF`) associated with the potential.
     * @return An integer representing the number of fields.
     */
    int getnF()
    {
        return nF;
    }

    /**
     * @brief Returns the number of parameters (`nP`) associated with the potential.
     * @return An integer representing the number of parameters.
     */
    int getnP()
    {
        return nP;
    }

    /**
     * @brief Evaluates the potential function.
     *
     * This function returns the value of the potential for given field values
     * and model parameters. The implementation is automatically generated.
     *
     * @param f A vector containing the field values.
     * @param p A vector containing the model parameters.
     * @return The value of the potential.
     */
    double V(vector<double> f, vector<double> p)
    {
        double sum = 0.;
        // Pot
        return sum;
    }

    /**
     * @brief Evaluates the first derivatives (gradient) of the potential.
     *
     * This function returns a vector containing the gradient of the potential
     * with respect to each field, for given field values and model parameters.
     * The implementation is automatically generated.
     *
     * @param f A vector containing the field values.
     * @param p A vector containing the model parameters.
     * @return A vector with the components of the potential's gradient.
     */
    vector<double> dV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF);
        // dPot
        return sum;
    }

    /**
     * @brief Evaluates the second derivatives (Hessian matrix) of the potential.
     *
     * This function returns a flattened vector representing the Hessian matrix
     * of the potential (second derivatives with respect to pairs of fields),
     * for given field values and model parameters. The implementation is automatically generated.
     *
     * @param f A vector containing the field values.
     * @param p A vector containing the model parameters.
     * @return A flattened vector representing the Hessian matrix (size `nF*nF`).
     */
    vector<double> dVV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF * nF);
        // ddPot
        return sum;
    }

    /**
     * @brief Evaluates the third derivatives of the potential.
     *
     * This function returns a flattened vector representing the third derivatives
     * of the potential with respect to triplets of fields, for given field values
     * and model parameters. The implementation is automatically generated.
     *
     * @param f A vector containing the field values.
     * @param p A vector containing the model parameters.
     * @return A flattened vector representing the third derivative tensor (size `nF*nF*nF`).
     */
    vector<double> dVVV(vector<double> f, vector<double> p)
    {
        vector<double> sum(nF * nF * nF);
        // dddPot
        return sum;
    }
};

#endif
