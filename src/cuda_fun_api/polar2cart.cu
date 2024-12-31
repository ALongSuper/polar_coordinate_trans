#include "polar2cart.h"

namespace cuda_fun_api
{
    void polar2cart_cuda(cv::Mat& mat1, cv::Mat& mat2,
        const cv::Point2f& center, const float& min_radius,
        const float& max_radius, const float& min_theta, const float& max_theta)
    {
        int rows_c = static_cast<int>(std::ceil(max_radius - min_radius));
        int cols_c = static_cast<int>(std::ceil((max_theta - min_theta) * max_radius));

        mat2 = cv::Mat::zeros(rows_c, cols_c, CV_8UC1);
        float rstep = 1.0;
        float thetaStep = (max_theta - min_theta) / cols_c;

        float center_polar_x = center.x;
        float center_polar_y = center.y;

        uchar* deviceMat1Data;
        cudaMalloc((void**)&deviceMat1Data, mat1.rows * mat1.cols * sizeof(uchar));

        uchar* deviceMat2Data;
        cudaMalloc((void**)&deviceMat2Data, mat2.rows * mat2.cols * sizeof(uchar));

        //图像 cpu -> gpu
        cudaMemcpy(deviceMat1Data, mat1.data, mat1.rows * mat1.cols * sizeof(uchar), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceMat2Data, mat2.data, mat2.rows * mat2.cols * sizeof(uchar), cudaMemcpyHostToDevice);

        int maxRows = std::max(mat1.rows, mat2.rows);
        int maxCols = std::max(mat1.cols, mat2.cols);

        dim3 blockSize(32, 32);
        dim3 gridSize((maxCols + blockSize.x - 1) / blockSize.x, (maxRows + blockSize.y - 1) / blockSize.y);

        process<<<gridSize, blockSize>>>(deviceMat1Data, mat1.rows, mat1.cols, deviceMat2Data, mat2.rows, mat2.cols,1,1, min_radius,min_theta,thetaStep,center_polar_x,center_polar_y);

        //结果图像 gpu -> cpu
        cudaMemcpy(mat2.data, deviceMat2Data, mat2.rows * mat2.cols * sizeof(uchar), cudaMemcpyDeviceToHost);

        cudaFree(deviceMat1Data);
        cudaFree(deviceMat2Data);

    }

    void polar2cart_cv(cv::Mat& mat1, cv::Mat& mat2,
        const cv::Point2f& center, const float& min_radius,
        const float& max_radius, const float& min_theta, const float& max_theta)
    {
        //转换后图像的行即圆环的半径差，列数即最大半径对应的弧长
        int rows_c = static_cast<int>(std::ceil(max_radius - min_radius));
        int cols_c = static_cast<int>(std::ceil((max_theta - min_theta) * max_radius));

        mat2 = cv::Mat::zeros(rows_c, cols_c, CV_8UC1);
        // 极坐标转换的角度步长
        float rstep = 1.0;
        float thetaStep = (max_theta - min_theta) / cols_c;

        float center_polar_x = center.x;
        float center_polar_y = center.y;

        cv::parallel_for_(cv::Range(0, rows_c * cols_c), [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; r++)
            {
                int row = r / cols_c;
                int col = r % cols_c;
                float temp_r = min_radius + row * rstep;

                float theta_p = min_theta + col * thetaStep;
                float sintheta = std::sin(theta_p);
                float costheta = std::cos(theta_p);

                int polar_x = static_cast<int>(center_polar_x + temp_r * sintheta);
                int polar_y = static_cast<int>(center_polar_y + temp_r * costheta);

                if (polar_x >= 0 && polar_x < mat1.cols && polar_y >= 0 && polar_y < mat1.rows)
                {
                    mat2.at<uchar>(rows_c - 1 - row, col) = mat1.at<uchar>(polar_y, polar_x);
                }
            }
            });
    }

    __global__ void process(uchar* mat1Data, int mat1Rows, int mat1Cols, uchar* mat2Data, int mat2Rows, int mat2Cols,int channels1,int channels2,float min_radius,float min_theta,float thetaStep,float center_polar_x,float center_polar_y)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < mat2Rows && col < mat2Cols) {
            
            float theta_p = min_theta + col * thetaStep;
            float sintheta = std::sin(theta_p);
            float costheta = std::cos(theta_p);

            //每个极坐标系下的点，在原图(直角坐标系)下的坐标
            float temp_r = min_radius + row*1;
            int src_col = static_cast<int>(center_polar_x + temp_r * sintheta);
            int src_row = static_cast<int>(center_polar_y + temp_r * costheta);
            
            // 结果图排放时的坐标(row的定义是沿着半径增加的方向增加，与图像的索引dst_row方向相反。col则与dst_col一致)
            int dst_row = mat2Rows-1-row;
            int dst_col = col;
            if (src_col >= 0 && src_col < mat1Cols && src_row >= 0 && src_row < mat1Rows)
            {
                //每个点对应在原图(极坐标系)的索引
                int srcidex= (src_row*mat1Cols+ src_col)*channels1;
                //每个点对应在结果图(直角坐标系)的索引
                int dstIdex =  (dst_row*mat2Cols+ dst_col)*channels2;
                mat2Data[dstIdex] =  mat1Data[srcidex];     
            }
        }
    }

}



