/*
 * @Descripttion: 
 * @version: 
 * @Author: SJL
 * @Date: 2023-08-14 10:14:17
 * @LastEditors: SJL
 * @LastEditTime: 2024-12-31 14:15:49
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
using namespace std::chrono;
#define PI 2 * std::acos(0.0)
#include "cuda_fun_api/polar2cart.h" 

int main()
{
	cv::Mat ori_image = cv::imread("..\\..\\1.png");
	cv::Mat ori_gray;
	cv::cvtColor(ori_image, ori_gray, cv::COLOR_BGR2GRAY);
	int i = 0;
	cv::Mat output_image;
	float sum_time = 0;
	int min_radius = 680;
	int max_radius = 1330;
	cv::Point2f circle_center = cv::Point2f(2102,1539);
	while(i < 20)
	{
		double t1 = cv::getTickCount();
		// opencv的多线程
		// cuda_fun_api::polar2cart_cv(ori_gray.clone(), output_image, circle_center, min_radius, max_radius, 0, 2 * PI);

		// cuda
		cuda_fun_api::polar2cart_cuda(ori_gray.clone(), output_image, circle_center, min_radius, max_radius, 0, 2 * PI);
		double t2 = cv::getTickCount();
		//剔除第一张的运行时间（在gpu开辟内存会比较耗时）
		std::cout <<  (t2 - t1) * 1000 / cv::getTickFrequency() << std::endl;
		if (i > 0)
			sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();
		i++;
	}
	std::cout << "average:" <<sum_time/19 << std::endl;

	// cv::namedWindow("ori_image",0);
	// cv::circle(ori_image,circle_center,min_radius,cv::Scalar(0,0,255),10);
	// cv::circle(ori_image,circle_center,max_radius,cv::Scalar(0,0,255),10);
	// cv::resizeWindow("ori_image", ori_image.cols/5, ori_image.rows/5);
	// cv::imshow("ori_image", ori_image);

	// cv::namedWindow("output_image",0);
	// cv::resizeWindow("output_image", output_image.cols/5, output_image.rows/5);
	// cv::imshow("output_image", output_image);
	// cv::waitKey(0);

	return 0;
}