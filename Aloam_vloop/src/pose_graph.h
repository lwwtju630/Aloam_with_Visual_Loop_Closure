#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include "keyframe.h"
#include "utility/utility.h"
//#include "utility/tic_toc.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"


#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph
{
public:
	PoseGraph();
	~PoseGraph();
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
	int detectLoop(KeyFrame* keyframe, int frame_index);
	void loadVocabulary(std::string voc_path);
	KeyFrame* getKeyFrame(int index);
	nav_msgs::Path path[10];
	nav_msgs::Path base_path;

private:
	list<KeyFrame*> keyframelist;
	std::mutex m_keyframelist;
	std::mutex m_path;
	std::mutex m_drift;

	map<int, cv::Mat> image_pool;

	BriefDatabase db;
	BriefVocabulary* voc;

	ros::Publisher pub_pg_path;
	ros::Publisher pub_base_path;
	ros::Publisher pub_pose_graph;
	ros::Publisher pub_path[10];
};

