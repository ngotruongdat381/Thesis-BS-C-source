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

static Scalar blue = Scalar(255, 0, 0);
static Scalar green = Scalar(0, 255, 0);
static Scalar red = Scalar(0, 0, 255);
static Scalar black = Scalar(0, 0, 0);

double ColourDistance(Vec3b e1, Vec3b e2);
double Angle(Point start, Point end);
float FindY_LineEquationThroughTwoPoint(float x_, Point p1, Point p2);

class MYcppGui {
public:
	MYcppGui();
	~MYcppGui();

	int myCppLoadAndShowRGB(string fileName);
	Mat face_detection_dlib(string fileName);
	void MYcppGui::VideoProcessing(string fileName);
	void MYcppGui::ImageProcessing(Mat &frame);
	void MYcppGui::ImageProcessing_WithUserInput(Mat &frame);
	Mat MYcppGui::ImageProcessing(string fileName, vector<cv::Point> userInput);

	void MYcppGui::Morphology_Operations(Mat &src);
	void MYcppGui::CannyProcessing(Mat image, OutputArray edges);
	std::vector<dlib::full_object_detection> MYcppGui::face_detection_dlib_image(Mat frame);
	void MYcppGui::detectNecessaryPointsOfFace(std::vector<dlib::full_object_detection> shapes_face);
	void MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, Point head_shoulder, Point end_shoulder, int angle, int distance);
	void MYcppGui::detectShoulderLine(Mat shoulder_detection_image, Mat detected_edges, bool leftHandSide, int angle, Scalar color, bool checkColor);

	cv::vector<Point> findPath(int index, int index_line, cv::vector<cv::vector<Point>> point_collection, double angle);
	void MYcppGui::ShowSampleShoulder();

	cv::vector<Point> MYcppGui::getFeatureFromUserInput(Mat shoulder_detection_image, Point head_shoulder, Point end_shoulder, int angle, int distance);
	void AddUserInput(vector<cv::Point> _userInput);
	bool MYcppGui::IsMatchToUserInput(Point point);
	bool IsMatchToColorCollectionInput(Vec3b color);
	void collectColorShoulder();

private:
	dlib::shape_predictor sp; //shape_predictor
	Mat userInputFrame;
	vector<cv::Point> userInput;
	vector<int>	colorValueCollection;
	vector<Vec3b> colorCollection;
	
	vector<cv::Point> leftRefinedInput;
	vector<cv::Point> rightRefinedInput;
	vector<Mat> featureCollection;
	double checking_block;
	double distance_from_face_to_shouldersample;

	Point left_cheek = NULL;
	Point right_cheek = NULL;
	Point chin = NULL;
	Point top_nose = NULL;
	Point symmetric_point = NULL;
};