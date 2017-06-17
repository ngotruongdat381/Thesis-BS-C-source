#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;
using namespace cv;
//using namespace dlib;

const clock_t begin_time = clock();
static clock_t current_time = clock();

const Scalar blue = Scalar(255, 0, 0);
const Scalar green = Scalar(0, 255, 0);
const Scalar red = Scalar(0, 0, 255);
const Scalar black = Scalar(0, 0, 0);
const Scalar yellow = Scalar(0, 255, 255);
const Scalar organe = Scalar(243, 97, 53);

const int RIGHT = 1;
const int LEFT = 0;

static int FACE_DOWNSAMPLE_RATIO = 3;
static int SKIP_FRAMES = 3;

double EuclideanDistance(Point2f p1, Point2f p2);
double ColourDistance(Vec3b e1, Vec3b e2);
double Angle(Point2f start, Point2f end);
float FindY_LineEquationThroughTwoPoint(float x_, Point2f p1, Point2f p2);
bool isSegmentsIntersecting(Point2f& p1, Point2f& p2, Point2f& q1, Point2f& q2);
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r);

class MYcppGui {
public:
	MYcppGui();
	~MYcppGui();

	Mat GetThumnail(string fileName);
	int myCppLoadAndShowRGB(string fileName);
	void MYcppGui::VideoProcessing(string fileName);
	//void MYcppGui::ImageProcessing(Mat &frame);
	void MYcppGui::ImageProcessing_WithUserInput(Mat &frame, bool isTesting, bool DebugLine);
	Mat MYcppGui::ImageProcessing(string fileName, vector<cv::Point2f> userInput);

	void MYcppGui::Morphology_Operations(Mat &src);
	void MYcppGui::CannyProcessing(Mat image, OutputArray edges);
	std::vector<dlib::full_object_detection> MYcppGui::face_detection_dlib_image(Mat frame);
	std::vector<dlib::full_object_detection> MYcppGui::face_detection_update(Mat frame);
	void MYcppGui::detectNecessaryPointsOfFace(std::vector<dlib::full_object_detection> shapes_face);
	//void MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, Point head_shoulder, Point end_shoulder, int angle, int distance);
	cv::vector<Point2f> MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color
										, bool checkColor, bool checkPreviousResult);
	void AddSticker(Mat &frame);

	cv::vector<Point2f> findPath(int index, int index_line, cv::vector<cv::vector<Point2f>> point_collection, double angle);
	void MYcppGui::ShowSampleShoulder();
	void AddUserInput(vector<vector<Point2f>> _userInput);
	bool MYcppGui::IsMatchToUserInput(Point2f point);
	bool IsMatchToColorCollectionInput(Vec3b color);
	void collectColorShoulder();
	Mat Preprocessing(Mat frame);
	void GetSticker(string name);
private:
	dlib::shape_predictor shape_predictor;
	Mat userInputFrame;
	vector<vector<Point2f>> userInput;
	vector<vector<Point2f>> simplifizedUserInput;
	vector<vector<Point2f>> current_shoulderLine;
	vector<vector<Point2f>> simplifized_current_shoulderLine;

	vector<int>	colorValueCollection;
	vector<Vec3b> colorCollection;
	
	vector<cv::Point2f> leftRefinedInput;
	vector<cv::Point2f> rightRefinedInput;

	vector<Mat> featureCollection;
	double checking_block;
	double distance_from_face_to_shouldersample;
	int nth = 1;	//nth frame
	std::vector<dlib::rectangle> cur_dets;
	
	//stickers variable
	Vector<Mat> stickerFrames;
	int index_stickerFrames = 0;
	double relativePostion_sticker;

	Point2f left_cheek = NULL;
	Point2f right_cheek = NULL;
	Point2f chin = NULL;
	Point2f top_nose = NULL;
	Point2f nose = NULL;
	Point2f symmetric_point = NULL;
	Point2f upper_symmetric_point = NULL;
};