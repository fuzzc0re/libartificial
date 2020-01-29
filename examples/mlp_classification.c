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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../src/libartificial.h"
#include "../src/libartificial_utils.h"

void model(const int *restrict rows,
           const int *restrict cols_Y,
           const int *restrict cols_X,
           double Y[(*rows) * (*cols_Y)],
           double X[(*rows) * (*cols_X)])
{
  int i, j, random_prob_one;

  for (i = 0; i < (*rows); i++)
  {
    for (j = 0; j < (*cols_X); j++)
    {
      if (j == 0)
      {
        X[i * (*cols_X) + j] = 1;
      }
      else if (j == 1)
      {
        X[i * (*cols_X) + j] = rand_normal(1.0, 1.0);
      }
      else
      {
        X[i * (*cols_X) + j] = rand_normal(2.0, 2.0);
      }
    }
  }

  for (i = 0; i < (*rows); i++)
  {
    random_prob_one = rand() % (((*cols_Y) - 1) + 1 - 0) + 0; // range(0, labels)
    for (j = 0; j < (*cols_Y); j++)
    {
      if (j == random_prob_one)
      {
        Y[i * (*cols_Y) + j] = 1.0;
      }
      else
      {
        Y[i * (*cols_Y) + j] = 0.0;
      }
    }
  }
}

int main(void)
{
  srand(time(NULL));

  // The model
  ///////////////////////////////////////////////////////////////////////////
  const int cols_X = 3;
  const int cols_Y = 30;
  const int rows = 1024;

  int c = rows * cols_X;
  double(*X)[c];
  mallocDouble1d(&c, &X);
  c = rows * cols_Y;
  double(*Y)[c];
  mallocDouble1d(&c, &Y);

  model(&rows, &cols_Y, &cols_X, *Y, *X);
  ////////////////////////////////////////////////////////////////////////////////////////////////

  // Hyperparameters
  const int batch = 128;       // Divisor of 1024
  const double w_variance = 1; // For the weight initialization
  const double learning_rate = 0.0001;
  const int epochs = 1000;

  const int layers = 1;
  const int nodes[1] = {200};
  char funcs[2][30] = {
      "relu",
      "softmax"};
  //

  // First we normalize X for the gradients
  normalize(&rows, &cols_X, *X);

  // Initialize helpers, weights and prediction
  c = layers + 1;
  int(*f)[c];
  mallocInt1d(&c, &f);
  name2int(&layers, *f, funcs);

  double ***Z = init_Z(&batch, &cols_Y, &cols_X, &layers, nodes);
  double ***helpers = init_helpers(&batch, &cols_Y, &cols_X, &layers, nodes);
  // We initialize weights and biases at every layer (if we do not already have them)
  // wb[l][i * columns_X + j] weights at layer l=0,...,layers, i'th row j'th column
  // wb[l][j] biases at layer l=0,...,layers always 1 row and j'th column
  double **weights = init_w(&w_variance, &cols_Y, &cols_X, &layers, nodes, *f);

  // For the averaging of deltas in batch/mini-batch
  double correction = learning_rate * 1.0 / (double)rows;

  c = batch * cols_Y;
  double(*Y_batch)[c];
  mallocDouble1d(&c, &Y_batch);
  c = batch * cols_X;
  double(*X_batch)[c];
  mallocDouble1d(&c, &X_batch);

  double loss = 0.0;
  for (int e = 0; e < epochs; e++)
  {
    randomize(&rows, &cols_Y, &cols_X, &batch, *Y_batch, *X_batch, *Y, *X);
    loss = cpu_gd_train(&batch, &cols_Y, &cols_X, &layers, nodes, *f, *Y_batch, *X_batch, Z, weights, helpers, &learning_rate, &correction);
    if (loss != loss) // Loss is NaN
    {
      free(X);
      free(Y);
      free(X_batch);
      free(Y_batch);
      free(f);
      delete_Z(Z, &layers);
      delete_helpers(helpers, &layers);
      delete_w(weights, &layers);
      printf(KRED "\nLoss is NaN due to exploding gradient. Fix learning rate. Aborting...\n" RESET);
      abort();
    }
    printf("\nLoss = %.10lf at epoch = %i\n", loss, e);
  }
  free(Y_batch);
  free(X_batch);
  delete_Z(Z, &layers);
  delete_helpers(helpers, &layers);

  c = rows * cols_Y;

  double *Z_pred = cpu_feedforward_predict(&rows, &cols_Y, &cols_X, &layers, nodes, *f, *X, weights);

  // for (i = 0; i < rows; i++)
  // {
  //   printf("\nZ_pred is %.10lf and Y is %.10lf", Z_pred[i], Y[i]);
  // }
  // printf("\n");
  delete_w(weights, &layers);
  free(X);
  free(Y);
  free(f);
  free(Z_pred);

  return 0;
}
