/**
 *  @file sat.cuh
 *  @brief CUDA device code for GPU-Efficient Summed-Area Tables
 *  @author Andre Maximo
 *  @date September, 2011
 */

#ifndef SAT_CUH
#define SAT_CUH

//== NAMESPACES ===============================================================

namespace gpufilter {

//== EXTERNS ==================================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 1
 *
 *  This function computes the algorithm stage S.1 following:
 *
 *  In parallel for all \f$m\f$ and \f$n\f$, compute and store the
 *  \f$P_{m,n}(\bar{Y})\f$ and \f$P^T_{m,n}(\hat{V})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5() and figure in algorithmSAT()
 *  @param[in] g_in Input image
 *  @param[out] g_ybar All \f$P_{m,n}(\bar{Y})\f$
 *  @param[out] g_vhat All \f$P^T_{m,n}(\hat{V})\f$
 */
__global__
void algorithmSAT_stage1( const float *g_in,
                          float *g_ybar,
                          float *g_vhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 2
 *
 *  This function computes the algorithm stage S.2 following:
 *
 *  Sequentially for each \f$m\f$, but in parallel for each \f$n\f$,
 *  compute and store the \f$P_{m,n}(Y)\f$ and using the previously
 *  computed \f$P_{m,n}(\bar{Y})\f$.  Compute and store
 *  \f$s(P_{m,n}(Y))\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5() and figure in algorithmSAT()
 *  @param[in,out] g_ybar All \f$P_{m,n}(\bar{Y})\f$ fixed to \f$P_{m,n}(Y)\f$
 *  @param[out] g_ysum All \f$s(P_{m,n}(Y))\f$
 */
__global__
void algorithmSAT_stage2( float *g_ybar,
                          float *g_ysum );

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 3
 *
 *  This function computes the algorithm stage S.3 following:
 *
 *  Sequentially for each \f$n\f$, but in parallel for each \f$m\f$,
 *  compute and store the \f$P^T{m,n}(V)\f$ using the previously
 *  computed \f$P_{m-1,n}(Y)\f$, \f$P^T_{m,n}(\hat{V})\f$ and
 *  \f$s(P_{m,n}(Y))\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5() and figure in algorithmSAT()
 *  @param[in] g_ysum All \f$s(P_{m,n}(Y))\f$
 *  @param[in,out] g_vhat All \f$P^T_{m,n}(\hat{V})\f$ fixed to \f$P^T_{m,n}(V)\f$
 */
__global__
void algorithmSAT_stage3( const float *g_ysum,
                          float *g_vhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 4
 *
 *  This function computes the algorithm stage S.4 following:
 *
 *  In parallel for all \f$m\f$ and \f$n\f$, compute \f$B_{m,n}(Y)\f$
 *  then compute and store \f$B_{m,n}(V)\f$ and using the previously
 *  computed \f$P_{m,n}(Y)\f$ and \f$P^T_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5() and figure in algorithmSAT()
 *  @param[in] g_inout The input and output image
 *  @param[out] g_y All \f$P_{m,n}(Y)\f$
 *  @param[out] g_v All \f$P^T_{m,n}(V)\f$
 */
__global__
void algorithmSAT_stage4( float *g_inout,
                          const float *g_y,
                          const float *g_v );

__host__
void algorithmSAT( dvector<float>& d_img,
                   dvector<float>& d_xbar,
                   dvector<float>& d_ybar,
                   dvector<float>& d_xsum,
                   const dim3& cg_img,
                   const dim3& cg_xbar,
                   const dim3& cg_ybar );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // SAT_CUH
//=============================================================================
