#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
//#include "camodocal/camera_models/CameraFactory.h"
//#include "camodocal/camera_models/CataCamera.h"
//#include "camodocal/camera_models/PinholeCamera.h"
//#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;


class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);
  DVision::BRIEF m_brief;
};

class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, cv::Mat &_image);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();

	double time_stamp; 
	int index;
	int local_index;
	cv::Mat image;
	cv::Mat thumbnail;
	vector<cv::KeyPoint> keypoints; //第二部计算的特征点
	vector<BRIEF::bitset> brief_descriptors; //描述子
	bool has_fast_point;

	bool has_loop;
	int loop_index;
};

