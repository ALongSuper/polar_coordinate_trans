/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2023-08-14 10:08:13
 * @LastEditors: SJL
 * @LastEditTime: 2024-12-31 09:43:11
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#pragma once

#define PI 2 * std::acos(0.0)

namespace cuda_fun_api
{
	//cuda加速
	void polar2cart_cuda(cv::Mat& mat1, cv::Mat& mat2,
		const cv::Point2f& center, const float& min_radius,
		const float& max_radius, const float& min_theta, const float& max_theta);


	__global__ void process(uchar* mat1Data, int mat1Rows, int mat1Cols, uchar* mat2Data, int mat2Rows, int mat2Cols,int channels1,int channels2,float min_radius,float min_theta,float thetaStep,float center_polar_x,float center_polar_y);
   
	//cv::parallel_for加速	
	void polar2cart_cv(cv::Mat& mat1, cv::Mat& mat2,
		const cv::Point2f& center, const float& min_radius,
		const float& max_radius, const float& min_theta, const float& max_theta);

}







