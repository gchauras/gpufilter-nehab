/**
 *  @file gpufilter.cuh
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering
 *  @author Rodolfo Lima
 *  @date September, 2011
 */

#ifndef GPUFILTER_CUH
#define GPUFILTER_CUH

//== NAMESPACES ===============================================================

namespace gpufilter {

//== EXTERNS ==================================================================

/**
 *  @ingroup gpu
 */
__global__
void algorithm4_stage1( float *g_inout,
                        float2 *g_transp_ybar,
                        float2 *g_transp_zhat );

/**
 *  @ingroup gpu
 */
__global__
void algorithm4_stage2_3_or_5_6( float2 *g_transp_ybar,
                                 float2 *g_transp_zhat );

/**
 *  @ingroup gpu
 */
__global__
void algorithm4_stage4( float *g_inout,
                        float2 *g_transp_y,
                        float2 *g_transp_z,
                        float2 *g_ubar,
                        float2 *g_vhat );

/**
 *  @ingroup gpu
 */
__global__
void algorithm4_stage7( float *g_inout,
                        float2 *g_u,
                        float2 *g_v );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 1
 *
 *  This function computes the algorithm stage 5.1 following:
 *
 *  In parallel for all \f$m\f$ and \f$n\f$, compute and store each
 *  \f$P_{m,n}(\bar{Y})\f$, \f$E_{m,n}(\hat{Z})\f$,
 *  \f$P^T_{m,n}(\check{U})\f$, and \f$E^T_{m,n}(\tilde{V})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5()
 *  @param[in] g_in Input image
 *  @param[out] g_transp_ybar All \f$P_{m,n}(\bar{Y})\f$
 *  @param[out] g_transp_zhat All \f$E_{m,n}(\hat{Z})\f$
 *  @param[out] g_ucheck All \f$P^T_{m,n}(\check{U})\f$
 *  @param[out] g_vtilde All \f$E^T_{m,n}(\tilde{V})\f$
 */
__global__
void algorithm5_stage1( const float *g_in,
                        float *g_transp_ybar,
                        float *g_transp_zhat,
                        float *g_ucheck,
                        float *g_vtilde );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 2 and 3 (fusioned)
 *
 *  This function computes the algorithm stages 5.2 and 5.3 following:
 *
 *  In parallel for all \f$n\f$, sequentially for each \f$m\f$, compute and
 *  store the \f$P_{m,n}(Y)\f$ according to (37) and using the previously
 *  computed \f$P_{m-1,n}(\bar{Y})\f$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  In parallel for all \f$n\f$, sequentially for each \f$m\f$, compute and
 *  store \f$E_{m,n}(Z)\f$ according to (45) using the previously computed
 *  \f$P_{m-1,n}(Y)\f$ and \f$E_{m+1,n}(\hat{Z})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5()
 *  @param[in,out] g_transp_ybar All \f$P_{m,n}(\bar{Y})\f$
 *  @param[in,out] g_transp_zhat All \f$E_{m,n}(\hat{Z})\f$
 */
__global__
void algorithm5_stage2_3( float *g_transp_ybar,
                          float *g_transp_zhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 4 and 5 (fusioned) step 1
 *
 *  This function computes the first part of the algorithm stages 5.4
 *  and 5.5 following:
 *
 *  In parallel for all \f$m\f$, sequentially for each \f$n\f$, compute and
 *  store \f$P^T_{m,n}(U)\f$ according to (48) and using the previously
 *  computed \f$P^T_{m,n}(\check{U})\f$, \f$P_{m-1,n}(Y)\f$, and
 *  \f$E_{m+1,n}(Z)\f$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  In parallel for all \f$m\f$, sequentially for each \f$n\f$, compute and
 *  store \f$E^T_{m,n}(V)\f$ according to (50) and using the previously
 *  computed \f$E^T_{m,n}(\tilde{V})\f$, \f$P^T_{m,n-1}(U)\f$,
 *  \f$P_{m-1,n}(Y)\f$, and \f$E_{m+1,n}(Z)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5()
 *  @param[in,out] g_ucheck All \f$P^T_{m,n}(\check{U})\f$
 *  @param[in,out] g_vtilde All \f$E^T_{m,n}(\tilde{V})\f$
 *  @param[in] g_transp_y All \f$P_{m,n}(Y)\f$
 *  @param[in] g_transp_z All \f$E_{m,n}(Z)\f$
 */
__global__
void algorithm5_stage4_5_step1( float *g_ucheck,
                                float *g_vtilde,
                                const float *g_transp_y,
                                const float *g_transp_z );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 4 and 5 (fusioned) step 2
 *
 *  This function computes the second part of the algorithm stages 5.4
 *  and 5.5 following:
 *
 *  In parallel for all \f$m\f$, sequentially for each \f$n\f$, compute and
 *  store \f$P^T_{m,n}(U)\f$ according to (48) and using the previously
 *  computed \f$P^T_{m,n}(\check{U})\f$, \f$P_{m-1,n}(Y)\f$, and
 *  \f$E_{m+1,n}(Z)\f$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  In parallel for all \f$m\f$, sequentially for each \f$n\f$, compute and
 *  store \f$E^T_{m,n}(V)\f$ according to (50) and using the previously
 *  computed \f$E^T_{m,n}(\tilde{V})\f$, \f$P^T_{m,n-1}(U)\f$,
 *  \f$P_{m-1,n}(Y)\f$, and \f$E_{m+1,n}(Z)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5()
 *  @param[in,out] g_ubar All \f$P^T_{m,n}(\bar{U})\f$ (half-way fixed \f$P^T_{m,n}(U)\f$)
 *  @param[in,out] g_vcheck All \f$E^T_{m,n}(\check{V})\f$ (half-way fixed \f$E^T_{m,n}(V)\f$)
 */
__global__
void algorithm5_stage4_5_step2( float *g_ubar,
                                float *g_vcheck );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 6
 *
 *  This function computes the algorithm stage 5.6 following:
 *
 *  In parallel for all \f$m\f$ and \f$n\f$, compute one after the other
 *  \f$B_{m,n}(Y)\f$, \f$B_{m,n}(Z)\f$, \f$B_{m,n}(U)\f$, and \f$B_{m,n}(V)\f$
 *  according to (18) and using the previously computed
 *  \f$P_{m-1,n}(Y)\f$, \f$E_{m+1,n}(Z)\f$, \f$P^T_{m,n-1}(U)\f$, and
 *  \f$E^T_{m,n+1}(V)\f$. Store \f$B_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in algorithm5()
 *  @param[in,out] g_inout The input and output image
 *  @param[in] g_transp_y All \f$P_{m,n}(Y)\f$
 *  @param[in] g_transp_z All \f$E_{m,n}(Z)\f$
 *  @param[in] g_u All \f$P^T_{m,n}(U)\f$
 *  @param[in] g_v All \f$E^T_{m,n}(V)\f$
 */
__global__
void algorithm5_stage6( float *g_inout,
                        const float *g_transp_y,
                        const float *g_transp_z,
                        const float *g_u,
                        const float *g_v );
/**
 *  @ingroup gpu
 */
__global__
void algorithm5_stage6_fusion_algorithm4_stage1( float *g_inout,
                                                 const float *g_transp_y,
                                                 const float *g_transp_z,
                                                 const float *g_u,
                                                 const float *g_v,
                                                 float2 *g_transp_ybar,
                                                 float2 *g_transp_zhat );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUFILTER_CUH
//=============================================================================