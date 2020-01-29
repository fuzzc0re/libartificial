/* Copyright (c) 2020 Fuzznets P.C. and Dim Karoukis. All rights reserved.
 * 
 * Licensed under the Affero General Public License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   https://www.gnu.org/licenses/agpl-3.0.en.html
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

#ifndef libartificial_h__
#define libartificial_h__

#ifdef __WIN32__
#if defined(COMPILING_DLL)
#define PUBLIC_API __declspec(dllexport)
#else
#define PUBLIC_API __declspec(dllimport)
#endif
#else
#define PUBLIC_API
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "libartificial_utils.h"

#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define RESET "\033[0m"

// Converts names into array of ints
static inline void name2int(const int *restrict layers,
                            int f[(*layers)],
                            const char funcs[(*layers) + 1][30]);
// Activations
static inline void activate(const int *restrict rows,
                            const int *restrict cols,
                            const int *restrict f,
                            double Y[(*rows) * (*cols)],
                            const double X[(*rows) * (*cols)]);
// Gradients
static inline void gradient(const int *restrict rows,
                            const int *restrict cols,
                            const int *restrict f,
                            double Y[(*rows) * (*cols)],
                            const double X[(*rows) * (*cols)]);
// Losses
static inline double mse(const int *restrict rows,
                         const int *restrict cols_Y,
                         const double Y[(*rows) * (*cols_Y)],
                         const double Z[(*rows) * (*cols_Y)]);

static inline double xentropy(const int *restrict rows,
                              const int *restrict cols_Y,
                              const double Y[(*rows) * (*cols_Y)],
                              const double Z[(*rows) * (*cols_Y)]);

// Feedforward pass
static inline void cpu_mm_notrans(const int *restrict rows,
                                  const int *restrict cols,
                                  const int *restrict coms,
                                  const double A[(*rows) * (*coms)],
                                  const double B[(*coms) * (*cols)],
                                  double C[(*rows) * (*cols)]);

static inline void cpu_mm_notrans_trans(const int *restrict rows,
                                        const int *restrict cols,
                                        const int *restrict coms,
                                        const double A[(*rows) * (*coms)],
                                        const double B[(*cols) * (*coms)],
                                        double C[(*rows) * (*cols)]);

static inline void cpu_gd_delta(const int *restrict rows,
                                const int *restrict cols_Y,
                                const int *restrict layers,
                                const int *restrict nodes,
                                const int f[(*layers)],
                                const double *restrict Y,
                                double ***restrict Z,
                                double **restrict w,
                                double ***helpers);
// returns new wb
static inline void cpu_threaded_update(const int *restrict rows,
                                       const int *restrict cols_Y,
                                       const int *restrict cols_X,
                                       const int *restrict layers,
                                       const int *restrict nodes,
                                       const double X[(*rows) * (*cols_X)],
                                       double ***restrict Z,
                                       double **restrict w,
                                       double ***restrict helpers,
                                       const double *restrict correction);
//

static inline void PUBLIC_API cpu_feedforward_update(const int *restrict rows,
                                                     const int *restrict cols_Y,
                                                     const int *restrict cols_X,
                                                     const int *restrict layers,
                                                     const int *restrict nodes,
                                                     const int f[(*layers)],
                                                     const double X[(*rows) * (*cols_X)],
                                                     double ***restrict Z,
                                                     double **restrict w)
{
  // l is for layers
  int l;
  for (l = 0; l < (*layers) + 1; l++)
  {
    if (l == 0)
    {
      cpu_mm_notrans(rows, &nodes[l], cols_X, X, w[l], Z[0][l]);
      activate(rows, &nodes[l], &f[l], Z[1][l], Z[0][l]);
    }
    else if (l == (*layers))
    {
      cpu_mm_notrans(rows, cols_Y, &nodes[l - 1], Z[1][l - 1], w[l], Z[0][l]);
      activate(rows, cols_Y, &f[l], Z[1][l], Z[0][l]);
    }
    else
    {
      cpu_mm_notrans(rows, &nodes[l], &nodes[l - 1], Z[1][l - 1], w[l], Z[0][l]);
      activate(rows, &nodes[l], &f[l], Z[1][l], Z[0][l]);
    }
  }
}

// Actual perceptron
static inline double *PUBLIC_API cpu_feedforward_predict(const int *restrict rows,
                                                         const int *restrict cols_Y,
                                                         const int *restrict cols_X,
                                                         const int *restrict layers,
                                                         const int *restrict nodes,
                                                         const int f[(*layers)],
                                                         const double X[(*rows) * (*cols_X)],
                                                         double **restrict w)
{
  // l is for layers
  // i is for each row * column of X, Y
  int i, r_times_col = (*rows) * (*cols_Y);
  double ***Z = init_Z(rows, cols_Y, cols_X, layers, nodes);
  double *Z_pred = malloc(sizeof(double[r_times_col]));
  assert(Z_pred != NULL);

  // Directly manipulates Z
  cpu_feedforward_update(rows, cols_Y, cols_X, layers, nodes, f, X, Z, w);

  for (i = 0; i < r_times_col; i++)
  {
    Z_pred[i] = Z[1][(*layers)][i];
  }

  delete_Z(Z, layers);
  return Z_pred;
}

static inline double PUBLIC_API cpu_gd_train(const int *restrict rows,
                                             const int *restrict cols_Y,
                                             const int *restrict cols_X,
                                             const int *restrict layers,
                                             const int *restrict nodes,
                                             const int f[(*layers)],
                                             const double Y[(*rows) * (*cols_Y)],
                                             const double X[(*rows) * (*cols_X)],
                                             double ***restrict Z,
                                             double **restrict w,
                                             double ***restrict helpers,
                                             const double *restrict learning_rate,
                                             const double *restrict correction)
{
  // // The values to be subtracted from weights
  // register double **deltas = (*helpers)[0];
  // register double **grad_w = (*helpers)[1];
  // // Gradient of layer's unactivated output
  // register double **help_1 = (*helpers)[2];
  // // Product of next layer's transposed weights and deltas
  // register double **help_2 = (*helpers)[3];

  // Find deltas
  cpu_gd_delta(rows, cols_Y, layers, nodes, f, Y, Z, w, helpers);

  // Update the weights
  cpu_threaded_update(rows, cols_Y, cols_X, layers, nodes, X, Z, w, helpers, correction);

  // Update Zs with the new wb's
  cpu_feedforward_update(rows, cols_Y, cols_X, layers, nodes, f, X, Z, w);

  // Save weights
  // save_w(w, layers, nodes, cols_Y, cols_X);

  switch (f[(*layers)])
  {
  case 4: // If softmax then cross entropy
    return xentropy(rows, cols_Y, Y, Z[1][(*layers)]);
  case 6: // softplus
    return xentropy(rows, cols_Y, Y, Z[1][(*layers)]);
  case 7: // softsign
    return xentropy(rows, cols_Y, Y, Z[1][(*layers)]);
  default:
    return mse(rows, cols_Y, Y, Z[1][(*layers)]);
  }
}

static inline void name2int(const int *restrict layers,
                            int f[(*layers) + 1],
                            const char funcs[(*layers) + 1][30])
{
  int l;
  for (l = 0; l < (*layers) + 1; l++)
  {
    switch (strcmp(funcs[l], "relu"))
    {
    case 0:
      f[l] = 0;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "logistic"))
    {
    case 0:
      f[l] = 1;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "linear"))
    {
    case 0:
      f[l] = 2;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "tanh"))
    {
    case 0:
      f[l] = 3;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "softmax"))
    {
    case 0:
      f[l] = 4;
      continue;
    default:
      break;
    }
    // Leaky relu
    switch (strcmp(funcs[l], "lrelu"))
    {
    case 0:
      f[l] = 5;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "softplus"))
    {
    case 0:
      f[l] = 6;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "softsign"))
    {
    case 0:
      f[l] = 7;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "arctan"))
    {
    case 0:
      f[l] = 8;
      continue;
    default:
      break;
    }
    //Inverse square root with a = 1
    switch (strcmp(funcs[l], "isru"))
    {
    case 0:
      f[l] = 9;
      continue;
    default:
      break;
    }
    //Inverse sqrt linear unit \w a=1
    switch (strcmp(funcs[l], "isrlu"))
    {
    case 0:
      f[l] = 10;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "bent"))
    {
    case 0:
      f[l] = 11;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "sinus"))
    {
    case 0:
      f[l] = 12;
      continue;
    default:
      break;
    }
    switch (strcmp(funcs[l], "sinusc"))
    {
    case 0:
      f[l] = 13;
      continue;
    default:
      // Gaussian if nothing else
      f[l] = 14;
      break;
    }
  }
  printf(KGRN "\nSuccessfully converted function names to numbers!\n" RESET);
}

static inline void activate(const int *restrict rows,
                            const int *restrict cols,
                            const int *restrict f,
                            double Y[(*rows) * (*cols)],
                            const double X[(*rows) * (*cols)])
{
  int i, c = (*rows) * (*cols);
  switch ((*f))
  {
  case 0: // Relu
    for (i = 0; i < c; i++)
    {
      if (X[i] < 0.0)
      {
        Y[i] = 0.0;
      }
      else
      {
        Y[i] = X[i];
      }
    }
    return;
  case 1: // Logistic
    for (i = 0; i < c; i++)
    {
      Y[i] = 1 / (1 + exp(-X[i]));
    }
    return;
  case 2: // Linear
    for (i = 0; i < c; i++)
    {
      Y[i] = X[i];
    }
    return;
  case 3: // Tanh
    for (i = 0; i < c; i++)
    {
      Y[i] = tanh(X[i]);
    }
    return;
  case 4: // Softmax
  {
    double *e = malloc(sizeof(double[(*cols)]));
    double e_X;
    int j;
    for (i = 0; i < (*rows); i++)
    {
      e_X = 0.0;
      for (j = 0; j < (*cols); j++)
      {
        e[j] = exp(X[i * (*cols) + j]);
        e_X += e[j];
      }
      for (j = 0; j < (*cols); j++)
      {
        Y[i * (*cols) + j] = e[j] / e_X;
      }
    }
    free(e);
    return;
  }
  case 5: // Lrelu
    for (i = 0; i < c; i++)
    {
      if (X[i] < 0.0)
      {
        Y[i] = 0.01 * X[i];
      }
      else
      {
        Y[i] = X[i];
      }
    }
    return;
  case 6: // Softplus
    for (i = 0; i < c; i++)
    {
      Y[i] = log(1 + exp(X[i]));
    }
    return;
  case 7: // Softsign
    for (i = 0; i < c; i++)
    {
      Y[i] = X[i] / (1 + fabs(X[i]));
    }
    return;
  case 8: // Arctan
    for (i = 0; i < c; i++)
    {
      Y[i] = atan(X[i]);
    }
    return;
  case 9: // Isru
    for (i = 0; i < c; i++)
    {
      Y[i] = X[i] / sqrt(1 + X[i] * X[i]);
    }
    return;
  case 10: // Isrlu
    for (i = 0; i < c; i++)
    {
      if (X[i] < 0.0)
      {
        Y[i] = X[i] / sqrt(1 + X[i] * X[i]);
      }
      else
      {
        Y[i] = X[i];
      }
    }
    return;
  case 11: // Bent
    for (i = 0; i < c; i++)
    {
      Y[i] = (sqrt(X[i] * X[i] + 1.0) - 1.0) / 2.0 + X[i];
    }
    return;
  case 12: // Sinus
    for (i = 0; i < c; i++)
    {
      Y[i] = sin(X[i]);
    }
    return;
  case 13: // Sinusc
    for (i = 0; i < c; i++)
    {
      if (X[i] == 0.0)
      {
        Y[i] = 1.0;
      }
      else
      {
        Y[i] = sin(X[i]) / X[i];
      }
    }
    return;
  default: // Gauss
    for (i = 0; i < c; i++)
    {
      Y[i] = exp(-(X[i] * X[i]));
    }
    return;
  }
}

static inline void gradient(const int *restrict rows,
                            const int *restrict cols,
                            const int *restrict f,
                            double Y[(*rows) * (*cols)],
                            const double X[(*rows) * (*cols)])
{
  int i, c = (*rows) * (*cols);
  switch ((*f))
  {
  case 0: // Relu
    for (i = 0; i < c; i++)
    {
      if (X[i] < 0.0)
      {
        Y[i] = 0.0;
      }
      else
      {
        Y[i] = 1.0;
      }
    }
    return;
  case 1: // Logistic
  {
    double y;
    for (i = 0; i < c; i++)
    {
      y = 1 / (1 + exp(-X[i]));
      Y[i] = y * (1 - y);
    }
    return;
  }
  case 2: // Linear
    for (i = 0; i < c; i++)
    {
      Y[i] = 1.0;
    }
    return;
  case 3: // Tanh
  {
    double e_X, e_mX, y;
    for (i = 0; i < c; i++)
    {
      e_X = exp(X[i]);
      e_mX = exp(-X[i]);
      y = (e_X - e_mX) / (e_X + e_mX);
      Y[i] = 1 - y * y;
    }
    return;
  }
  case 4: // Softmax
  {
    double *e = malloc((*cols) * sizeof(double));
    int j;
    double e_X, e_mX;
    for (i = 0; i < (*rows); i++)
    {
      e_X = 0.0;
      for (j = 0; j < (*cols); j++)
      {
        e[j] = exp(X[i * (*cols) + j]);
        e_X += e[j];
      }
      for (j = 0; j < (*cols); j++)
      {
        e_mX = e[j] / e_X;
        if (i == j)
        {
          Y[i * (*cols) + j] = e_mX * (1 - e_mX);
        }
        else
        {
          Y[i * (*cols) + j] = -e_mX * e_mX;
        }
      }
    }
    free(e);
    return;
  }
  case 5: // Lrelu
    for (i = 0; i < c; i++)
    {
      if (X[i] < 0.0)
      {
        Y[i] = 0.01;
      }
      else
      {
        Y[i] = 1.0;
      }
    }
    return;
  case 6: // Softplus
    for (i = 0; i < c; i++)
    {
      Y[i] = 1 / (1 + exp(-X[i]));
    }
    return;
  case 7: // Softsign
  {
    double y;
    for (i = 0; i < c; i++)
    {
      y = 1 + fabs(X[i]);
      Y[i] = 1 / (y * y);
    }
    return;
  }
  case 8: // Arctan
    for (i = 0; i < c; i++)
    {
      Y[i] = 1 / (X[i] * X[i] + 1);
    }
    return;
  case 9: // Isru
  {
    double sq, y;
    for (i = 0; i < c; i++)
    {
      sq = sqrt(1 + X[i] * X[i]);
      y = X[i] / sq;
      Y[i] = y * y * y;
    }
    return;
  }
  case 10: // Isrlu
  {
    double sq, y;
    for (i = 0; i < c; i++)
    {
      if (X[i] < 0.0)
      {
        sq = sqrt(1 + X[i] * X[i]);
        y = X[i] / sq;
        Y[i] = y * y * y;
      }
      else
      {
        Y[i] = 1.0;
      }
    }
    return;
  }
  case 11: // Bent
  {
    double y, add;
    for (i = 0; i < c; i++)
    {
      add = X[i] + 1;
      y = sqrt(add * add);
      Y[i] = X[i] / (2 * y) + 1;
    }
    return;
  }
  case 12: // Sinus
    for (i = 0; i < c; i++)
    {
      Y[i] = cos(X[i]);
    }
    return;
  case 13: // Sinusc
    for (i = 0; i < c; i++)
    {
      if (X[i] == 0.0)
      {
        Y[i] = 0.0;
      }
      else
      {
        Y[i] = cos(X[i]) / X[i] - sin(X[i]) / (X[i] * X[i]);
      }
    }
    return;
  default: // Gauss
    for (i = 0; i < c; i++)
    {
      Y[i] = -2.0 * X[i] * exp(-(X[i] * X[i]));
    }
    return;
  }
}

static inline double mse(const int *restrict rows,
                         const int *restrict cols_Y,
                         const double Y[(*rows) * (*cols_Y)],
                         const double Z[(*rows) * (*cols_Y)])
{
  int i, c = (*rows) * (*cols_Y);
  double loss = 0.0;
  double d = 0.0;
  for (i = 0; i < c; i++)
  {
    d = Z[i] - Y[i];
    loss += (d * d) / (double)(*cols_Y);
  }
  loss /= (double)(*rows);
  return loss;
}

static inline double xentropy(const int *restrict rows,
                              const int *restrict cols_Y,
                              const double Y[(*rows) * (*cols_Y)],
                              const double Z[(*rows) * (*cols_Y)])
{
  int i, c = (*rows) * (*cols_Y);
  double loss = 0.0;
  for (i = 0; i < c; i++)
  {
    loss += Y[i] * log(Z[i]) / (double)(*cols_Y);
  }
  loss /= (double)(*rows);
  return -loss;
}

static inline void cpu_mm_notrans(const int *restrict rows,
                                  const int *restrict cols,
                                  const int *restrict coms,
                                  const double A[(*rows) * (*coms)],
                                  const double B[(*coms) * (*cols)],
                                  double C[(*rows) * (*cols)])
{
  double A_i;
  int i, j, k = (*rows) * (*cols);
  for (i = 0; i < k; i++)
  {
    C[i] = 0.0;
  }
  for (i = 0; i < (*rows); i++)
  {
    for (k = 0; k < (*coms); k++)
    {
      A_i = A[i * (*coms) + k];
      for (j = 0; j < (*cols); j++)
      {
        C[i * (*cols) + j] += A_i * B[k * (*cols) + j];
      }
    }
  }
}

static inline void cpu_mm_notrans_trans(const int *restrict rows,
                                        const int *restrict cols,
                                        const int *restrict coms,
                                        const double A[(*rows) * (*coms)],
                                        const double B[(*cols) * (*coms)],
                                        double C[(*rows) * (*cols)])
{
  int i, j, k;
  for (i = 0; i < (*rows); i++)
  {
    for (j = 0; j < (*cols); j++)
    {
      C[i * (*cols) + j] = 0.0;
      for (k = 0; k < (*coms); k++)
      {
        C[i * (*cols) + j] += A[i * (*coms) + k] * B[j * (*coms) + k];
      }
    }
  }
}

static inline void cpu_gd_delta(const int *restrict rows,
                                const int *restrict cols_Y,
                                const int *restrict layers,
                                const int *restrict nodes,
                                const int f[(*layers)],
                                const double *restrict Y,
                                double ***restrict Z,
                                double **restrict w,
                                double ***helpers)
{
  int i, l = (*layers), c = (*rows) * (*cols_Y);
  do
  {
    if (l == (*layers))
    {
      switch (f[l])
      {
      case 2: // Linear
        for (i = 0; i < c; i++)
        {
          helpers[0][l][i] = Z[1][l][i] - Y[i];
        }
        break;
      case 4: // Softmax crossentropy
        for (i = 0; i < c; i++)
        {
          helpers[0][l][i] = Z[1][l][i] - Y[i];
        }
        break;
      default:
        gradient(rows, cols_Y, &f[l], helpers[0][l], Z[0][l]);
        for (i = 0; i < c; i++)
        {
          helpers[0][l][i] *= Z[1][l][i] - Y[i];
        }
        break;
      }
    }
    else if (l == (*layers) - 1)
    {
      gradient(rows, &nodes[l], &f[l], helpers[2][l], Z[0][l]);
      cpu_mm_notrans_trans(rows, &nodes[l], cols_Y, helpers[0][l + 1], w[l + 1], helpers[3][l]);
      // Hadamard product
      c = (*rows) * nodes[l];
      for (i = 0; i < c; i++)
      {
        helpers[0][l][i] = helpers[2][l][i] * helpers[3][l][i];
      }
    }
    else
    {
      gradient(rows, &nodes[l], &f[l], helpers[2][l], Z[0][l]);
      cpu_mm_notrans_trans(rows, &nodes[l], &nodes[l + 1], helpers[0][l + 1], w[l + 1], helpers[3][l]);
      c = (*rows) * nodes[l];
      for (i = 0; i < c; i++)
      {
        helpers[0][l][i] = helpers[2][l][i] * helpers[3][l][i];
      }
    }
  } while (--l >= 0);
}

// returns new wb
static inline void cpu_threaded_update(const int *restrict rows,
                                       const int *restrict cols_Y,
                                       const int *restrict cols_X,
                                       const int *restrict layers,
                                       const int *restrict nodes,
                                       const double X[(*rows) * (*cols_X)],
                                       double ***restrict Z,
                                       double **restrict w,
                                       double ***restrict helpers, // need deltas [0] and grad_w [1]
                                       const double *restrict correction)
{
  int l, i, c, com, row, col;
  for (l = 0; l < (*layers) + 1; l++)
  {
    if (l == 0)
    {
      c = (*cols_X) * nodes[l];
      for (i = 0; i < c; i++)
      {
        helpers[1][l][i] = 0.0;
      }
      double A_i;
      for (row = 0; row < (*rows); row++)
      {
        for (com = 0; com < (*cols_X); com++)
        {
          A_i = X[row * (*cols_X) + com];
          for (col = 0; col < nodes[l]; col++)
          {
            helpers[1][l][com * nodes[l] + col] += A_i * helpers[0][l][row * nodes[l] + col];
          }
        }
      }
      for (i = 0; i < c; i++)
      {
        w[l][i] -= (*correction) * helpers[1][l][i];
      }
    }
    else if (l > 0 && l < (*layers))
    {
      c = nodes[l - 1] * nodes[l];
      for (i = 0; i < c; i++)
      {
        helpers[1][l][i] = 0.0;
      }
      double A_i;
      for (row = 0; row < (*rows); row++)
      {
        for (com = 0; com < nodes[l - 1]; com++)
        {
          A_i = Z[1][l - 1][row * nodes[l - 1] + com];
          for (col = 0; col < nodes[l]; col++)
          {
            helpers[1][l][com * nodes[l] + col] += A_i * helpers[0][l][row * nodes[l] + col];
          }
        }
      }
      for (i = 0; i < c; i++)
      {
        w[l][i] -= (*correction) * helpers[1][l][i];
      }
    }
    else
    {
      c = nodes[l - 1] * (*cols_Y);
      for (i = 0; i < c; i++)
      {
        helpers[1][l][i] = 0.0;
      }
      double A_i;
      for (row = 0; row < (*rows); row++)
      {
        for (com = 0; com < nodes[l - 1]; com++)
        {
          A_i = Z[1][l - 1][row * nodes[l - 1] + com];
          for (col = 0; col < (*cols_Y); col++)
          {
            helpers[1][l][com * (*cols_Y) + col] += A_i * helpers[0][l][row * (*cols_Y) + col];
          }
        }
      }
      for (i = 0; i < c; i++)
      {
        w[l][i] -= (*correction) * helpers[1][l][i];
      }
    }
  }
}

#endif // libartificial_h__
