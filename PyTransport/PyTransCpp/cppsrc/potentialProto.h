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

#ifndef POTENTIALPROTO_H
#define POTENTIALPROTO_H

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

// #Rewrite

// #FP
static int nF;
static int nP;

/**
 * @brief Prototype function for evaluating the potential.
 * @param f Vector of field values.
 * @param p Vector of model parameters.
 * @return The potential value.
 */
double V(double f[], double p[]); // Pot

/**
 * @brief Prototype function for evaluating the gradient of the potential.
 * @param f Vector of field values.
 * @param p Vector of model parameters.
 * @param sum Output array for the gradient components.
 */
void dV(double f[], double p[], double sum[]); // dPot

/**
 * @brief Prototype function for evaluating the Hessian (second derivatives) of the potential.
 * @param f Vector of field values.
 * @param p Vector of model parameters.
 * @param sum Output array for the flattened Hessian matrix.
 */
void dVV(double f[], double p[], double sum[]); // ddPot

/**
 * @brief Prototype function for evaluating the third derivatives of the potential.
 * @param f Vector of field values.
 * @param p Vector of model parameters.
 * @param sum Output array for the flattened third derivative tensor.
 */
void dVVV(double f[], double p[], double sum[]); // dddPot

#endif
