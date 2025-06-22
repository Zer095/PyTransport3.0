/**
 * @file fieldmetric.h
 * @brief Defines functions for calculating field space metric and related curvature quantities.
 *
 * This file provides functions to compute the field space metric (g_ij),
 * Christoffel symbols (Gamma^k_ij), Riemann curvature tensor (R_ijkl),
 * and covariant derivatives of the Riemann tensor. These quantities are
 * essential for models with a curved field space.
 *
 * @copyright GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 * @see http://www.gnu.org/licenses/
 */

#ifndef FIELDMETRIC_H
#define FIELDMETRIC_H

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

/**
 * @brief Computes the field space metric.
 *
 * This function is generated automatically by `PyTransSetup.py` based on the
 * symbolic definition of the metric in Python. It evaluates the metric tensor
 * components for given field values and parameters.
 *
 * @param f An array of field values.
 * @param p An array of model parameters.
 * @param sum An output array where the flattened 2D metric tensor (g_ij) will be stored.
 */
void FM(double f[],double p[],double sum[])
{
    // #FP
    // metric
}

/**
 * @brief Computes the Christoffel symbols.
 *
 * This function is generated automatically by `PyTransSetup.py`. It evaluates the
 * Christoffel symbols (Gamma^k_ij) for given field values and parameters.
 *
 * @param f An array of field values.
 * @param p An array of model parameters.
 * @param sum An output array where the flattened 3D Christoffel symbol tensor will be stored.
 */
void CS(double f[],double p[],double sum[])
{
    // Christoffel
}

/**
 * @brief Computes the Riemann curvature tensor.
 *
 * This function is generated automatically by `PyTransSetup.py`. It evaluates the
 * Riemann curvature tensor components for given field values and parameters.
 *
 * @param f An array of field values.
 * @param p An array of model parameters.
 * @param sum An output array where the flattened 4D Riemann tensor will be stored.
 */
void RM(double f[],double p[],double sum[])
{
    // Riemann
}

/**
 * @brief Computes the covariant derivative of the Riemann curvature tensor.
 *
 * This function is generated automatically by `PyTransSetup.py`. It evaluates the
 * covariant derivative of the Riemann tensor for given field values and parameters.
 *
 * @param f An array of field values.
 * @param p An array of model parameters.
 * @param sum An output array where the flattened 5D covariant Riemann tensor will be stored.
 */
void RMcd(double f[],double p[],double sum[])
{
    // Riemanncd
}

#endif
