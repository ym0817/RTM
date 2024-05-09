#include "rtmpose_onnxruntime.h"

#include <iostream>
#include <thread>

// #include "characterset_convert.h"
#include "rtmpose_utils.h"

#undef max

RTMPoseOnnxruntime::RTMPoseOnnxruntime()
{
}

RTMPoseOnnxruntime::~RTMPoseOnnxruntime()
{
}

std::vector<PosePoint> RTMPoseOnnxruntime::Inference(const cv::Mat& input_mat, const DetectBox& box)
{
	std::vector<PosePoint> pose_result;

	if (!box.IsValid())
		return pose_result;

	std::pair<cv::Mat, cv::Mat> crop_result_pair = CropImageByDetectBox(input_mat, box);

	cv::Mat crop_mat = crop_result_pair.first;
	cv::Mat affine_transform_reverse = crop_result_pair.second;

	// deep copy
	// ���
	cv::Mat crop_mat_copy;
	crop_mat.copyTo(crop_mat_copy);

	// BGR to RGB
	// BGRתRGB
	cv::Mat input_mat_copy_rgb;
	cv::cvtColor(crop_mat, input_mat_copy_rgb, cv::COLOR_BGR2RGB);

	// image data��HWC->CHW��image_data - mean / std normalize
	// ͼƬ���ݣ�HWC->CHW��image_data - mean / std��һ��
	int image_height = input_mat_copy_rgb.rows;
	int image_width = input_mat_copy_rgb.cols;
	int image_channels = input_mat_copy_rgb.channels();

	std::vector<float> input_image_array;
	input_image_array.resize(1 * image_channels * image_height * image_width);

	float* input_image = input_image_array.data();
	for (int h = 0; h < image_height; ++h)
	{
		for (int w = 0; w < image_width; ++w)
		{
			for (int c = 0; c < image_channels; ++c)
			{
				int chw_index = c * image_height * image_width + h * image_width + w;

				float tmp = input_mat_copy_rgb.ptr<uchar>(h)[w * 3 + c];

				input_image[chw_index] = (tmp - IMAGE_MEAN[c]) / IMAGE_STD[c];
			}
		}
	}

	// inference
	// ����
	std::vector<const char*> m_onnx_input_names{ "input" };
	std::vector<const char*> m_onnx_output_names{ "simcc_x","simcc_y" };
	std::array<int64_t, 4> input_shape{ 1, image_channels, image_height, image_width };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		input_image_array.data(),
		input_image_array.size(),
		input_shape.data(),
		input_shape.size()
	);

	assert(input_tensor.IsTensor());

	auto output_tensors = m_session.Run(
		Ort::RunOptions{ nullptr },
		m_onnx_input_names.data(),
		&input_tensor,
		1,
		m_onnx_output_names.data(),
		m_onnx_output_names.size()
	);

	// pose process
	// ����
	std::vector<int64_t> simcc_x_dims = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> simcc_y_dims = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

	assert(simcc_x_dims.size() == 3 && simcc_y_dims.size() == 3);

	int batch_size = simcc_x_dims[0] == simcc_y_dims[0] ? simcc_x_dims[0] : 0;
	int joint_num = simcc_x_dims[1] == simcc_y_dims[1] ? simcc_x_dims[1] : 0;
	int extend_width = simcc_x_dims[2];
	int extend_height = simcc_y_dims[2];

	float* simcc_x_result = output_tensors[0].GetTensorMutableData<float>();
	float* simcc_y_result = output_tensors[1].GetTensorMutableData<float>();

	
	for (int i = 0; i < joint_num; ++i)
	{
		// find the maximum and maximum indexes in the value of each Extend_width length
		// ��ÿһ��extend_width���ȵ���ֵ���ҵ����ֵ�Լ����ֵ������
		auto x_biggest_iter = std::max_element(simcc_x_result + i * extend_width, simcc_x_result + i * extend_width + extend_width);
		int max_x_pos = std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
		int pose_x = max_x_pos / 2;
		float score_x = *x_biggest_iter;

		// find the maximum and maximum indexes in the value of each exten_height length
		// ��ÿһ��extend_height���ȵ���ֵ���ҵ����ֵ�Լ����ֵ������
		auto y_biggest_iter = std::max_element(simcc_y_result + i * extend_height, simcc_y_result + i * extend_height + extend_height);
		int max_y_pos = std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
		int pose_y = max_y_pos / 2;
		float score_y = *y_biggest_iter;

		//float score = (score_x + score_y) / 2;
		float score = std::max(score_x, score_y);

		PosePoint temp_point;
		temp_point.x = int(pose_x);
		temp_point.y = int(pose_y);
		temp_point.score = score;
		pose_result.emplace_back(temp_point);
	}

	// anti affine transformation to obtain the coordinates on the original picture
	// ������任��ȡ��ԭʼͼƬ�ϵ�����
	for (int i = 0; i < pose_result.size(); ++i)
	{
		cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
		origin_point_Mat.at<double>(0, 0) = pose_result[i].x;
		origin_point_Mat.at<double>(1, 0) = pose_result[i].y;

		cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;

		pose_result[i].x = temp_result_mat.at<double>(0, 0);
		pose_result[i].y = temp_result_mat.at<double>(1, 0);
	}

	return pose_result;
}

std::pair<cv::Mat, cv::Mat> RTMPoseOnnxruntime::CropImageByDetectBox(const cv::Mat& input_image, const DetectBox& box)
{
	std::pair<cv::Mat, cv::Mat> result_pair;

	if (!input_image.data)
	{
		return result_pair;
	}

	if (!box.IsValid())
	{
		return result_pair;
	}

	// deep copy
	// ���
	cv::Mat input_mat_copy;
	input_image.copyTo(input_mat_copy);

	// calculate the width, height and center points of the human detection box
	// �����������Ŀ������Լ����ĵ�
	int box_width = box.right - box.left;
	int box_height = box.bottom - box.top;
	int box_center_x = box.left + box_width / 2;
	int box_center_y = box.top + box_height / 2;

	float aspect_ratio = 192.0 / 256.0;

	// adjust the width and height ratio of the size of the picture in the RTMPOSE input
	// ����rtmpose����ͼƬ��С�Ŀ��߱������е���
	if (box_width > (aspect_ratio * box_height))
	{
		box_height = box_width / aspect_ratio;
	}
	else if (box_width < (aspect_ratio * box_height))
	{
		box_width = box_height * aspect_ratio;
	}

	float scale_image_width = box_width * 1.25;
	float scale_image_height = box_height * 1.25;

	// get the affine matrix
	// ��ȡ�������
	cv::Mat affine_transform = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256
	);
	
	cv::Mat affine_transform_reverse = GetAffineTransform(
		box_center_x,
		box_center_y,
		scale_image_width,
		scale_image_height,
		192,
		256,
		true
	);

	// affine transform
	// ���з���任
	cv::Mat affine_image;
	cv::warpAffine(input_mat_copy, affine_image, affine_transform, cv::Size(192, 256), cv::INTER_LINEAR);
	//cv::imwrite("affine_img.jpg", affine_image);

	result_pair = std::make_pair(affine_image, affine_transform_reverse);

	return result_pair;
}

