/* Copyright (c) 2020 Fuzznets P.C. All rights reserved.
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

#ifndef libartificial_utils_h__
#define libartificial_utils_h__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define RESET "\033[0m"

static inline void mallocInt1d(const int *restrict i,
                               int (**A)[(*i)])
{
  *A = malloc(sizeof(int[(*i)]));
  assert(*A != NULL);
}

static inline void mallocDouble1d(const int *restrict i,
                                  double (**A)[(*i)])
{
  *A = malloc(sizeof(double[(*i)]));
  assert(*A != NULL);
}

static inline double rand_normal(const double mu,
                                 const double sigma)
{
  static double n2 = 0.0;
  static double n2_cached = 0.0;
  if (!n2_cached)
  {
    double x, y, r;
    do
    {
      x = 2.0 * (double)rand() / RAND_MAX - 1;
      y = 2.0 * (double)rand() / RAND_MAX - 1;

      r = x * x + y * y;
    } while (r == 0.0 || r > 1.0);
    double d = sqrt(-2.0 * log(r) / r);
    double n1 = x * d;
    n2 = y * d;
    double result = n1 * sigma + mu;
    n2_cached = 1.0;
    return result;
  }
  else
  {
    n2_cached = 0.0;
    return n2 * sigma + mu;
  }
}

// Definitions of forward declarated functions

static inline void randomize(const int *restrict rows,
                             const int *restrict cols_Y,
                             const int *restrict cols_X,
                             const int *restrict batch,
                             double Y_batch[(*batch) * (*cols_Y)],
                             double X_batch[(*batch) * (*cols_X)],
                             const double Y[(*rows) * (*cols_Y)],
                             const double X[(*rows) * (*cols_X)])
{
  int i, k, j;
  if ((*batch) > (*rows))
  {
    printf(KRED "\nBatch number bigger than number of rows. Aborting...\n" RESET);
    abort();
  }
  for (i = 0; i < (*batch); i++)
  {
    k = rand() % (*rows);
    for (j = 0; j < (*cols_X); j++)
    {
      X_batch[i * (*cols_X) + j] = X[k * (*cols_X) + j];
    }
    for (j = 0; j < (*cols_Y); j++)
    {
      Y_batch[i * (*cols_Y) + j] = Y[k * (*cols_Y) + j];
    }
  }
}

static inline double mean(const int *restrict rows,
                          const double col[(*rows)])
{
  int i;
  double sum = 0.0;
  for (i = 0; i < (*rows); i++)
  {
    sum += col[i];
  }
  sum /= (double)(*rows);
  return sum;
}

static inline double stdev(const int *restrict rows,
                           const double col[(*rows)],
                           const double *restrict mean)
{
  int i;
  double sumsq = 0.0, subtr;
  for (i = 0; i < (*rows); i++)
  {
    subtr = col[i] - (*mean);
    sumsq += subtr * subtr;
  }
  sumsq /= (double)((*rows) - 1);
  return sqrt(sumsq);
}

static inline void normalize(const int *restrict rows,
                             const int *restrict cols_X,
                             double X[(*rows) * (*cols_X)])
{
  int i, j;
  double m = 0.0, sd = 0.0, (*col)[(*rows)];
  mallocDouble1d(rows, &col);

  for (j = 1; j < (*cols_X); j++) // First column is ones (for bias)
  {
    for (i = 0; i < (*rows); i++)
    {
      (*col)[i] = X[i * (*cols_X) + j];
    }
    m = mean(rows, *col);
    sd = stdev(rows, *col, &m);
    for (i = 0; i < (*rows); i++)
    {
      X[i * (*cols_X) + j] = (X[i * (*cols_X) + j] - m) / sd;
    }
  }
  free(col);
}

static inline double ***init_Z(const int *restrict rows,
                               const int *restrict cols_Y,
                               const int *restrict cols_X,
                               const int *restrict layers,
                               const int *restrict nodes)
{
  // l is for layers
  // c is for each row * column of X, Y
  int l, c, i;
  double ***Z;
  Z = malloc(2 * sizeof(**Z));
  assert(Z != NULL);

  // unactivated
  Z[0] = malloc(((*layers) + 1) * sizeof(*Z));
  assert(Z[0] != NULL);
  // activated
  Z[1] = malloc(((*layers) + 1) * sizeof(*Z));
  assert(Z[1] != NULL);

  for (l = 0; l < (*layers) + 1; l++)
  {
    c = l != (*layers) ? (*rows) * nodes[l] : (*rows) * (*cols_Y);
    Z[0][l] = malloc(sizeof(double[c]));
    assert(Z[0][l] != NULL);
    Z[1][l] = malloc(sizeof(double[c]));
    assert(Z[1][l] != NULL);

    for (i = 0; i < c; i++)
    {
      Z[0][l][i] = 0.0;
      Z[1][l][i] = 0.0;
    }
  }
  printf(KGRN "\nZs initialized successfully!\n" RESET);
  return Z;
}

static inline double ***init_helpers(const int *restrict rows,
                                     const int *restrict cols_Y,
                                     const int *restrict cols_X,
                                     const int *restrict layers,
                                     const int *restrict nodes)
{
  int l, i, for_helper_w, for_helper_batch;
  double ***helpers;
  helpers = malloc(4 * sizeof(**helpers));
  assert(helpers != NULL);

  // The values to be subtracted from weights (deltas)
  helpers[0] = malloc(((*layers) + 1) * sizeof(*helpers));
  assert(helpers[0] != NULL);
  // grad_w
  helpers[1] = malloc(((*layers) + 1) * sizeof(*helpers));
  assert(helpers[1] != NULL);
  // Gradient of layer's unactivated output (help_1)
  helpers[2] = malloc((*layers) * sizeof(*helpers));
  assert(helpers[2] != NULL);
  // Product of next layer's transposed weights and deltas (help_2)
  helpers[3] = malloc((*layers) * sizeof(*helpers));
  assert(helpers[3] != NULL);

  for (l = 0; l < (*layers) + 1; l++)
  {
    if (l == 0)
    {
      for_helper_batch = (*rows) * nodes[l];
      for_helper_w = (*cols_X) * nodes[l];
      helpers[2][l] = malloc(sizeof(double[for_helper_batch]));
      assert(helpers[2][l] != NULL);
      helpers[3][l] = malloc(sizeof(double[for_helper_batch]));
      assert(helpers[3][l] != NULL);

      for (i = 0; i < for_helper_batch; i++)
      {
        helpers[2][l][i] = 0.0;
        helpers[3][l][i] = 0.0;
      }
    }
    else if (l == (*layers))
    {
      for_helper_batch = (*rows) * (*cols_Y);
      for_helper_w = nodes[l - 1] * (*cols_Y);
    }
    else
    {
      for_helper_batch = (*rows) * nodes[l];
      for_helper_w = nodes[l - 1] * nodes[l];
      helpers[2][l] = malloc(sizeof(double[for_helper_batch]));
      assert(helpers[2][l] != NULL);
      helpers[3][l] = malloc(sizeof(double[for_helper_batch]));
      assert(helpers[3][l] != NULL);

      for (i = 0; i < for_helper_batch; i++)
      {
        helpers[2][l][i] = 0.0;
        helpers[3][l][i] = 0.0;
      }
    }

    helpers[0][l] = malloc(sizeof(double[for_helper_batch]));
    assert(helpers[0][l] != NULL);
    helpers[1][l] = malloc(sizeof(double[for_helper_w]));
    assert(helpers[1][l] != NULL);

    for (i = 0; i < for_helper_batch; i++)
    {
      helpers[0][l][i] = 0.0;
    }
    for (i = 0; i < for_helper_w; i++)
    {
      helpers[1][l][i] = 0.0;
    }
  }
  printf(KGRN "\nHelpers initialized successfully!\n" RESET);
  return helpers;
}

//	Depending on the data, tanh/relu may give nans.
//	Variance < 1 and close to 0.01 if data range too large
static inline double **init_w(const double *restrict variance,
                              const int *restrict cols_Y,
                              const int *restrict cols_X,
                              const int *restrict layers,
                              const int *restrict nodes,
                              const int *restrict f)
{
  // l layers
  int l, c, i;
  double **weights;
  // For the heuristics of weight initialization
  double correction;
  // wb[0] is weights;
  // wb[1] is biases;
  weights = malloc(((*layers) + 1) * sizeof(*weights));
  assert(weights != NULL);

  for (l = 0; l < (*layers) + 1; l++)
  {
    int isRelu = 0, isTanh = 0;
    if (f[l] == 0 || f[l] == 5)
    {
      isRelu = 1;
    }
    else if (f[l] == 3)
    {
      isTanh = 1;
    }

    if (l == 0)
    {
      c = (*cols_X) * nodes[l];
      weights[l] = malloc(sizeof(double[c]));
      assert(weights[l] != NULL);

      if (isRelu == 1)
      { // He et al.
        correction = sqrt(2.0 / (double)(*cols_X));
      }
      else if (isTanh == 1)
      { // Xavier
        correction = sqrt(1.0 / (double)(*cols_X));
      }
      else
      {
        correction = sqrt(2.0 / (double)(c));
      }
    }
    else if (l == (*layers))
    {
      c = nodes[l - 1] * (*cols_Y);
      weights[l] = malloc(sizeof(double[c]));
      assert(weights[l] != NULL);

      if (isRelu == 1)
      { // He et al.
        correction = sqrt(2.0 / (double)nodes[l - 1]);
      }
      else if (isTanh == 1)
      { // Xavier
        correction = sqrt(1.0 / (double)nodes[l - 1]);
      }
      else
      {
        correction = sqrt(2.0 / (double)(c));
      }
    }
    else
    {
      c = nodes[l - 1] * nodes[l];
      weights[l] = malloc(sizeof(double[c]));
      assert(weights[l] != NULL);

      if (isRelu == 1)
      { // He et al.
        correction = sqrt(2.0 / (double)nodes[l - 1]);
      }
      else if (isTanh == 1)
      { // Xavier
        correction = sqrt(1.0 / (double)nodes[l - 1]);
      }
      else
      {
        correction = sqrt(2.0 / (double)(c));
      }
    }
    for (i = 0; i < c; i++)
    {
      weights[l][i] = correction * rand_normal(0.0, (*variance));
    }
  }
  printf(KGRN "\nWeights and biases initialized successfully!\n" RESET);
  return weights;
}

// Free wb
static inline void delete_w(double **restrict w,
                            const int *restrict layers)
{
  int l;
  for (l = 0; l < (*layers) + 1; l++)
    free(w[l]);
  free(w);
}

static inline void delete_Z(double ***restrict Z,
                            const int *restrict layers)
{
  int l;
  for (l = 0; l < (*layers) + 1; l++)
  {
    free(Z[0][l]);
    free(Z[1][l]);
  }
  free(Z[0]);
  free(Z[1]);
  free(Z);
}

static inline void delete_helpers(double ***restrict helpers,
                                  const int *restrict layers)
{
  int l;
  for (l = 0; l < (*layers) + 1; l++)
  {
    free(helpers[0][l]);
    free(helpers[1][l]);
    if (l < (*layers))
    {
      free(helpers[2][l]);
      free(helpers[3][l]);
    }
  }
  free(helpers[0]);
  free(helpers[1]);
  free(helpers[2]);
  free(helpers[3]);
  free(helpers);
}

#endif // libartificial_utils_h__
