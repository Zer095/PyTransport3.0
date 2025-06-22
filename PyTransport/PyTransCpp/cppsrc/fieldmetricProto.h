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

#ifndef FIELDMETRICPROTO_H
#define FIELDMETRICPROTO_H

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

// #Rewrite

// #FP
static int nF;
static int nP;

/**
 * @brief Prototype function for computing the field space metric.
 * @param f Array of field values.
 * @param p Array of model parameters.
 * @param sum Output array for the flattened metric tensor.
 */
void FM(double f[],double p[],double sum[]); // metric

/**
 * @brief Prototype function for computing the Christoffel symbols.
 * @param f Array of field values.
 * @param p Array of model parameters.
 * @param sum Output array for the flattened Christoffel symbols.
 */
void CS(double f[],double p[],double sum[]); // Christoffel

/**
 * @brief Prototype function for computing the Riemann curvature tensor.
 * @param f Array of field values.
 * @param p Array of model parameters.
 * @param sum Output array for the flattened Riemann tensor.
 */
void RM(double f[],double p[],double sum[]); // Riemann

/**
 * @brief Prototype function for computing the covariant derivative of the Riemann curvature tensor.
 * @param f Array of field values.
 * @param p Array of model parameters.
 * @param sum Output array for the flattened covariant Riemann tensor.
 */
void RMcd(double f[],double p[],double sum[]); // Riemanncd

#endif
